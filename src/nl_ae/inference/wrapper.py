"""Qwen2.5 model wrapper: scoring + generation + (optional) hidden-state capture.

The wrapper imports ``torch`` and ``transformers`` lazily so unit tests that
only need configs and outputs can run without the model dependencies.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from types import TracebackType
from typing import TYPE_CHECKING, Any, Literal

from nl_ae.data.text_norm import nfc
from nl_ae.prompt.letter_tokens import LetterVariant, select_canonical_variant
from nl_ae.schema.models import (
    FirstTokenScoringMath,
    LetterStr,
    LetterTokenEntry,
    ModelFingerprint,
    QuantizationSpec,
    Sha256Hex,
)

from .decoding import DecodingConfig, hf_generate_kwargs, seed_all
from .extractor import AnswerExtractorProtocol, RegexLadderExtractor
from .loader import HookSpec, ModelConfig
from .outputs import (
    FirstTokenScore,
    ForwardOutput,
    FreeGenResult,
    GenerateOutput,
    LetterScore,
    ScoringOutputs,
)

if TYPE_CHECKING:  # pragma: no cover
    import torch

LOG = logging.getLogger(__name__)


class InferenceError(RuntimeError):
    """Raised on a model-level failure (OOM, tokenizer mismatch, etc.)."""


class ChatTemplateMismatch(RuntimeError):
    """Raised when the live ``tokenizer.chat_template`` hash differs from pinned."""


def _hash_chat_template(template_text: str) -> Sha256Hex:
    import hashlib

    return hashlib.sha256(nfc(template_text).encode("utf-8")).hexdigest()


def _select_block_outputs(
    hidden_states: Sequence[Any], layers: Sequence[int], n_layers: int
) -> dict[int, Any]:
    """Map block index ``N`` → block-``N`` output residual at the last prompt token.

    HF's ``output_hidden_states=True`` returns a tuple of length
    ``num_hidden_layers + 1``: ``hidden_states[0]`` is post-embed, and
    ``hidden_states[i]`` for ``i ≥ 1`` is the output of block ``i - 1``. The
    interpretability convention (and the hook path below) treats "layer N" as
    "block N output", so we read ``hidden_states[N + 1]`` for ``N ∈ [0, n_layers)``.
    """
    out: dict[int, Any] = {}
    for layer_idx in layers:
        if 0 <= layer_idx < n_layers and (layer_idx + 1) < len(hidden_states):
            out[layer_idx] = hidden_states[layer_idx + 1][0, -1, :].detach()
    return out


def _build_bnb_config(spec: QuantizationSpec) -> Any:
    """Lazily import ``bitsandbytes``-aware ``BitsAndBytesConfig`` only for ``int*``."""
    if not spec.kind.startswith("int"):
        return None
    try:
        import torch
        from transformers import BitsAndBytesConfig
    except ImportError as exc:  # pragma: no cover
        raise InferenceError(
            f"quantization kind={spec.kind!r} requires bitsandbytes + transformers"
        ) from exc

    compute_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[
        spec.compute_dtype
    ]
    if spec.kind == "int8":
        return BitsAndBytesConfig(load_in_8bit=True)
    quant_type = "nf4" if spec.kind == "int4-nf4" else "fp4"
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=quant_type,
        bnb_4bit_use_double_quant=spec.double_quant,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_quant_storage=spec.bnb_4bit_quant_storage,
    )


class Qwen25Wrapper:
    """Wrapper around a HF causal LM + tokenizer.

    Construction loads the model into memory; ``close()`` releases it. Use as
    a context manager for guaranteed cleanup.
    """

    def __init__(
        self,
        *,
        config: ModelConfig,
        extractor: AnswerExtractorProtocol | None = None,
        pinned_chat_template_hash: Sha256Hex | None = None,
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:  # pragma: no cover
            raise InferenceError(
                "torch + transformers are required to construct Qwen25Wrapper; "
                "install nl-ae[model]"
            ) from exc

        self._config = config
        self._extractor = extractor or RegexLadderExtractor()
        self._call_count = 0
        self._hidden_state_buffer: dict[int, torch.Tensor] = {}
        self._hook_handles: list[Any] = []

        bnb_config = _build_bnb_config(config.quantization)
        torch_dtype = {
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
            "fp32": torch.float32,
        }[config.quantization.compute_dtype]

        LOG.info(
            "loading tokenizer + model id=%s kind=%s dtype=%s",
            config.hf_model_id,
            config.quantization.kind,
            config.quantization.compute_dtype,
        )
        self._tokenizer = AutoTokenizer.from_pretrained(
            config.hf_model_id,
            revision=config.hf_revision,
            cache_dir=str(config.cache_dir) if config.cache_dir else None,
            trust_remote_code=False,
        )
        load_kwargs: dict[str, Any] = {
            "revision": config.hf_revision,
            "attn_implementation": config.attn_implementation,
            "device_map": config.device_map,
            "torch_dtype": torch_dtype,
        }
        if bnb_config is not None:
            load_kwargs["quantization_config"] = bnb_config
        if config.cache_dir is not None:
            load_kwargs["cache_dir"] = str(config.cache_dir)
        if config.max_memory_per_gpu_mb is not None:
            load_kwargs["max_memory"] = {0: f"{config.max_memory_per_gpu_mb}MiB"}

        self._model = AutoModelForCausalLM.from_pretrained(config.hf_model_id, **load_kwargs)
        self._model.eval()

        # Chat-template hash pinning (U2.10 defense).
        chat_template = getattr(self._tokenizer, "chat_template", None) or ""
        self._chat_template_hash: Sha256Hex = _hash_chat_template(chat_template)
        if pinned_chat_template_hash is not None and pinned_chat_template_hash != "0" * 64:
            if pinned_chat_template_hash != self._chat_template_hash:
                raise ChatTemplateMismatch(
                    f"chat_template hash mismatch: pinned={pinned_chat_template_hash} "
                    f"live={self._chat_template_hash}"
                )

        # Register hidden-state hooks if Phase 2 layers were requested.
        if config.hook_spec.record_layers:
            self._register_hooks(config.hook_spec)

    # --- properties -----------------------------------------------------

    @property
    def tokenizer(self) -> Any:
        return self._tokenizer

    @property
    def model_device(self) -> torch.device:
        return next(self._model.parameters()).device

    @property
    def extractor_id(self) -> str:
        return self._extractor.extractor_id

    @property
    def call_count(self) -> int:
        return self._call_count

    @property
    def chat_template_hash(self) -> Sha256Hex:
        return self._chat_template_hash

    @property
    def n_layers(self) -> int:
        return int(self._model.config.num_hidden_layers)

    @property
    def hidden_size(self) -> int:
        return int(self._model.config.hidden_size)

    def fingerprint(self) -> ModelFingerprint:
        commit = getattr(self._model.config, "_commit_hash", None)
        tok_commit = getattr(self._tokenizer, "_commit_hash", None)
        return ModelFingerprint(
            hf_model_id=self._config.hf_model_id,
            hf_model_commit=commit,
            hf_tokenizer_id=self._tokenizer.name_or_path,
            hf_tokenizer_commit=tok_commit,
            quantization=self._config.quantization,
        )

    # --- lifecycle ------------------------------------------------------

    def close(self) -> None:
        for h in self._hook_handles:
            try:
                h.remove()
            except Exception:  # pragma: no cover
                pass
        self._hook_handles.clear()
        try:
            import torch

            del self._model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:  # pragma: no cover
            pass

    def try_clear_cache(self) -> None:
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:  # pragma: no cover
            pass

    def __enter__(self) -> Qwen25Wrapper:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    # --- core operations ------------------------------------------------

    def forward(
        self,
        prompt: str,
        *,
        capture_hiddens: bool = False,
        record_layers_override: Sequence[int] | None = None,
    ) -> ForwardOutput:
        import torch

        layers = (
            tuple(record_layers_override)
            if record_layers_override is not None
            else (self._config.hook_spec.record_layers or ())
        )
        t0 = time.perf_counter()
        encoded = self._tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        input_ids = encoded["input_ids"].to(self.model_device)
        attention_mask = encoded.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.model_device)
        self._hidden_state_buffer.clear()
        with torch.inference_mode():
            outputs = self._model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                output_hidden_states=capture_hiddens,
            )
        logits = outputs.logits[0, -1, :].detach()
        hiddens: dict[int, torch.Tensor] | None = None
        if capture_hiddens and layers:
            hidden_states = getattr(outputs, "hidden_states", None) or ()
            hiddens = _select_block_outputs(hidden_states, layers, self.n_layers)
        elif capture_hiddens and self._hidden_state_buffer:
            hiddens = dict(self._hidden_state_buffer)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        call_id = self._call_count
        self._call_count += 1
        return ForwardOutput(
            last_token_logits=logits,
            prompt_token_count=int(input_ids.shape[-1]),
            hidden_states=hiddens,
            position_policy=self._config.hook_spec.position_policy if hiddens else None,
            wall_time_ms=elapsed_ms,
            engine_call_id=call_id,
        )

    def score_first_token(
        self,
        prompt: str,
        *,
        letter_token_table: Sequence[LetterTokenEntry],
        scoring_math: FirstTokenScoringMath = "full_vocab_softmax",
    ) -> FirstTokenScore:
        import torch

        forward = self.forward(prompt, capture_hiddens=False)
        logits = forward.last_token_logits
        letters = list(letter_token_table)
        if not letters:
            raise InferenceError("letter_token_table is empty")
        letter_ids = torch.tensor(
            [e.token_id for e in letters], dtype=torch.long, device=logits.device
        )
        letter_logits = logits[letter_ids]

        if scoring_math == "argmax_logits_only":
            argmax_idx = int(torch.argmax(letter_logits).item())
            per_letter = [
                LetterScore(
                    letter=e.letter,
                    token_id=e.token_id,
                    prob=None,
                    prob_valid=False,
                    logit=float(letter_logits[i].item()),
                )
                for i, e in enumerate(letters)
            ]
            return FirstTokenScore(
                argmax_letter=letters[argmax_idx].letter,
                per_letter=per_letter,
                scoring_math=scoring_math,
                total_letter_mass=0.0,
                prompt_token_count=forward.prompt_token_count,
                wall_time_ms=forward.wall_time_ms,
                engine_call_id=forward.engine_call_id,
            )

        if scoring_math == "renormalize_over_letters":
            probs = torch.softmax(letter_logits, dim=0)
            total_mass = 1.0
        else:  # full_vocab_softmax
            full = torch.softmax(logits, dim=0)
            probs = full[letter_ids]
            total_mass = float(probs.sum().item())

        argmax_idx = int(torch.argmax(probs).item())
        per_letter = [
            LetterScore(
                letter=e.letter,
                token_id=e.token_id,
                prob=float(probs[i].item()),
                prob_valid=True,
                logit=float(letter_logits[i].item()),
            )
            for i, e in enumerate(letters)
        ]
        return FirstTokenScore(
            argmax_letter=letters[argmax_idx].letter,
            per_letter=per_letter,
            scoring_math=scoring_math,
            total_letter_mass=min(max(total_mass, 0.0), 1.0),
            prompt_token_count=forward.prompt_token_count,
            wall_time_ms=forward.wall_time_ms,
            engine_call_id=forward.engine_call_id,
        )

    def generate(
        self,
        prompt: str,
        *,
        decoding: DecodingConfig,
    ) -> GenerateOutput:
        import torch

        if decoding.strategy == "sampled":
            assert decoding.seed is not None
            seed_all(decoding.seed)
        t0 = time.perf_counter()
        encoded = self._tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        input_ids = encoded["input_ids"].to(self.model_device)
        attention_mask = encoded.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.model_device)
        prompt_len = int(input_ids.shape[-1])
        kwargs = hf_generate_kwargs(decoding, tokenizer=self._tokenizer)
        try:
            with torch.inference_mode():
                output_ids = self._model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    **kwargs,
                )
        except RuntimeError as exc:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            call_id = self._call_count
            self._call_count += 1
            LOG.warning("model.generate failed: %r", exc)
            return GenerateOutput(
                text="",
                generated_token_ids=[],
                truncated=False,
                finish_reason="error",
                prompt_token_count=prompt_len,
                new_token_count=0,
                wall_time_ms=elapsed_ms,
                decode_strategy=decoding.strategy,
                seed=decoding.seed,
                generator_id="hf_generate_v1",
                engine_call_id=call_id,
            )

        new_ids = output_ids[0, prompt_len:].tolist()
        text = self._tokenizer.decode(new_ids, skip_special_tokens=True)

        truncated = len(new_ids) >= decoding.max_new_tokens
        finish_reason: Literal["eos", "stop_string", "max_new_tokens", "error"] = (
            "max_new_tokens" if truncated else "eos"
        )
        for stop in decoding.stop_strings:
            if stop and stop in text:
                truncated = False
                finish_reason = "stop_string"
                text = text.split(stop, 1)[0]
                break

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        call_id = self._call_count
        self._call_count += 1
        return GenerateOutput(
            text=text,
            generated_token_ids=new_ids,
            truncated=truncated,
            finish_reason=finish_reason,
            prompt_token_count=prompt_len,
            new_token_count=len(new_ids),
            wall_time_ms=elapsed_ms,
            decode_strategy=decoding.strategy,
            seed=decoding.seed,
            generator_id="hf_generate_v1",
            engine_call_id=call_id,
        )

    def generate_and_extract(
        self,
        prompt: str,
        *,
        letter_set: tuple[LetterStr, ...],
        decoding: DecodingConfig,
        max_free_text_chars: int = 2048,
    ) -> FreeGenResult:
        gen = self.generate(prompt, decoding=decoding)
        outcome = self._extractor.extract(
            gen.text, allowed_letters=frozenset(letter_set)
        )
        truncated_text = gen.text[:max_free_text_chars]
        return FreeGenResult(
            text=truncated_text,
            truncated=gen.truncated or len(gen.text) > max_free_text_chars,
            finish_reason=gen.finish_reason,
            extracted_letter=outcome.letter,
            match_rule=outcome.match_rule,
            extractor_id=outcome.extractor_id,
            seed=gen.seed,
            decode_strategy=gen.decode_strategy,
            wall_time_ms=gen.wall_time_ms,
            new_token_count=gen.new_token_count,
            generator_id=gen.generator_id,
            engine_call_id=gen.engine_call_id,
        )

    def score_and_generate(
        self,
        prompt: str,
        *,
        letter_token_table: Sequence[LetterTokenEntry],
        letter_set: tuple[LetterStr, ...],
        decoding: DecodingConfig,
        scoring_math: FirstTokenScoringMath = "full_vocab_softmax",
        variant: LetterVariant = "leading_space",
        max_free_text_chars: int = 2048,
    ) -> ScoringOutputs:
        """The C04 hot path: first-token scoring + free-gen + agreement.

        UR-R2.2: seed_all is called at the top so first-token scoring and
        free-gen share the same row seed under sampled decoding.
        """
        # UR-R2.2: seed everything before either path runs (sampled only).
        if decoding.strategy == "sampled" and decoding.seed is not None:
            seed_all(decoding.seed)

        t0 = time.perf_counter()
        filtered = select_canonical_variant(list(letter_token_table), chosen=variant)
        if len(filtered) != len(letter_set):
            # Fall back to using whichever variant rows happen to be present.
            filtered = [e for e in letter_token_table if e.letter in letter_set]
        first_token = self.score_first_token(
            prompt, letter_token_table=filtered, scoring_math=scoring_math
        )
        free = self.generate_and_extract(
            prompt,
            letter_set=letter_set,
            decoding=decoding,
            max_free_text_chars=max_free_text_chars,
        )

        agreement: bool | None
        if first_token.argmax_letter is None or free.extracted_letter is None:
            agreement = None
        else:
            agreement = first_token.argmax_letter == free.extracted_letter

        total_wall = (time.perf_counter() - t0) * 1000.0
        return ScoringOutputs(
            first_token_letter=first_token.argmax_letter,
            letter_softmax=list(first_token.per_letter),
            free_text_raw=free.text,
            free_text_truncated=free.truncated,
            free_text_letter=free.extracted_letter,
            extractor_id=free.extractor_id,
            agreement_flag=agreement,
            decode_strategy=decoding.strategy,
            decode_seed=decoding.seed,
            scoring_math=scoring_math,
            total_letter_mass=first_token.total_letter_mass,
            extractor_match_rule=free.match_rule,
            forward_wall_time_ms=first_token.wall_time_ms,
            generate_wall_time_ms=free.wall_time_ms,
            total_wall_time_ms=total_wall,
            provenance={
                "kv_shared": "false",
                "scorer_id": first_token.scorer_id,
                "generator_id": free.generator_id,
                "extractor_id": free.extractor_id,
                "extractor_version": self._extractor.extractor_version,
                "rules_content_hash": self._extractor.rules_content_hash,
                "chat_template_hash": self._chat_template_hash,
                "variant": variant,
            },
            engine_call_id=free.engine_call_id,
        )

    # --- hooks ----------------------------------------------------------

    def _register_hooks(self, spec: HookSpec) -> None:
        assert spec.record_layers is not None
        try:
            layers = self._resolve_block_modules()
        except (AttributeError, IndexError) as exc:  # pragma: no cover
            raise InferenceError(f"could not resolve transformer blocks for hooking: {exc}") from exc
        for layer_idx in spec.record_layers:
            if not (0 <= layer_idx < len(layers)):
                raise ValueError(
                    f"record_layers index {layer_idx} out of [0, {len(layers)})"
                )
            block = layers[layer_idx]

            def _make(idx: int) -> Any:
                def hook(_module: Any, _inputs: Any, output: Any) -> None:
                    tensor = output[0] if isinstance(output, tuple) else output
                    self._hidden_state_buffer[idx] = tensor.detach()

                return hook

            handle = block.register_forward_hook(_make(layer_idx))
            self._hook_handles.append(handle)

    def _resolve_block_modules(self) -> list[Any]:
        for path in ("model.layers", "transformer.h", "model.model.layers"):
            cur: Any = self._model
            try:
                for part in path.split("."):
                    cur = getattr(cur, part)
                return list(cur)
            except AttributeError:
                continue
        raise AttributeError(f"no transformer blocks found on {type(self._model).__name__}")


__all__ = [
    "ChatTemplateMismatch",
    "InferenceError",
    "Qwen25Wrapper",
]
