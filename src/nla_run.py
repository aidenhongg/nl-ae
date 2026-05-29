"""On-pod NLA driver: calibration gate + round-trip FVE sweep + NL verbalization.

Runs on the GPU pod against sglang servers for the AV (Activation Verbalizer) and AR
(Activation Reconstructor) from kitft/nla-inference. Drives NLAClient/NLACritic.

API (kitft/nla-inference, single-file nla_inference.py — verified on-pod 2026-05-28):
  client = NLAClient(actor_hf_path, sglang_url=AV_URL)     # AV: served by sglang (port 30000)
  critic = NLACritic(critic_hf_path, device="cuda:0")      # AR: PURE TORCH, in-process on GPU (no sglang)
  text   = client.generate(vec, temperature=..., max_new_tokens=...)  # vec: raw [d] f32, ANY norm
                                        #   (client rescales to injection_scale=150); returns <explanation> text
  mse,cos = critic.score(text, vec)     # -> (mse, cos) TUPLE; both L2-normed to mse_scale=sqrt(d); mse=2(1-cos)
  hhat   = critic.reconstruct(text)     # -> torch [d] activation (raw, unnormalized)

ALL normalization/scale/prompt/token conventions come from each checkpoint's
`nla_meta.yaml` sidecar (loaded by the client) — never hardcoded here (plan 00 §5.7).

Resumable: every AV call streams one JSON line to a .jsonl ledger; a re-run skips
(set, row, alpha) keys already present. Compacted to parquet/json at the end. This is
the cost-safety net — an interrupted/OOM pod resumes instead of re-paying.

Subcommands:
  calibrate          Phase C gate: FVE(orig) on the subset (+ optional indexing re-forward). STOP-on-fail.
  sweep              Phase D + Component-2(dense): subset x 11 alpha -> text + cos/mse + recon ĥ.
  verbalize-orig     Component-2: verbalize all 6536 original rows (store of record).
  verbalize-headline Component-2: verbalize full test (2615) at alpha in {2,10,30}.

Inputs (pulled from s3://iaxphg9saj/nla/inputs onto the pod, or sent from local):
  h_layer20_steered_a{tag}.npy, norms.parquet, subset_rows.json, steer_manifest.json
  + exp04 h_layer20_orig.npy (== a0) for the orig pass.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ALPHAS = [0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0]
HEADLINE_ALPHAS = [2.0, 10.0, 30.0]
CONFIG_HASH = "e80501525b6758e8a7c6f28556541bbbad1f268f92ae187972f83e69c075a55f"
CALIB_FVE_MIN, CALIB_FVE_MAX = 0.6, 0.8  # plan 01 §C.2 (card in-dist 0.752)
MAX_NEW_TOKENS = 256  # AV decode budget; template asks for 2-3 snippets in <explanation> tags
_CONCURRENCY = 8  # default parallel AV requests (overridable via --concurrency); set by main()


def alpha_tag(a: float) -> str:
    return ("%g" % a).replace(".", "p").replace("-", "m")


# ----------------------------- client adapters --------------------------------
# Reconciled against the live kitft/nla-inference (nla_inference.py, 2026-03) on-pod:
#   * AV = NLAClient(dir, sglang_url=...); verbalize is `generate(activation, **sampling)`
#     (NOT `verbalize`); returns the <explanation> text. Sampling kwargs (temperature,
#     max_new_tokens) pass through to sglang sampling_params. No per-request `seed`
#     (run-level determinism is set via sglang --random-seed at launch).
#   * AR = NLACritic(dir, device=...): PURE TORCH, in-process on the GPU — it is NOT an
#     sglang server (so `ar_url` is unused). `score(text, vec)` returns a (mse, cos)
#     TUPLE (both L2-normalized to mse_scale=sqrt(d); mse = 2(1-cos)).

def make_clients(actor: str, critic: str, av_url: str, ar_url: str = ""):
    from nla_inference import NLAClient, NLACritic
    import torch
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    return NLAClient(actor, sglang_url=av_url), NLACritic(critic, device=dev)


def av_verbalize(client, vec: np.ndarray, temperature: float, seed: int | None):
    vec = np.ascontiguousarray(vec, dtype=np.float32)
    return client.generate(vec, temperature=temperature, max_new_tokens=MAX_NEW_TOKENS)


def ar_score(critic, text: str, vec: np.ndarray) -> dict:
    mse, cos = critic.score(text, np.ascontiguousarray(vec, dtype=np.float32))
    return {"mse": float(mse), "cosine_similarity": float(cos)}


def ar_reconstruct(critic, text: str) -> np.ndarray:
    h = critic.reconstruct(text)
    try:
        import torch
        if isinstance(h, torch.Tensor):
            h = h.detach().float().cpu().numpy()
    except Exception:  # noqa: BLE001
        pass
    return np.asarray(h, dtype=np.float32).reshape(-1)


# ----------------------------- meta + provenance ------------------------------

def load_meta(actor: str) -> dict:
    import yaml
    p = Path(actor) / "nla_meta.yaml"
    if not p.exists():  # sglang may have been given an HF id; check HF cache fallback
        cands = list(Path(actor).rglob("nla_meta.yaml"))
        if cands:
            p = cands[0]
    if not p.exists():
        raise FileNotFoundError(f"nla_meta.yaml not found under {actor} — required (do not hardcode conventions)")
    meta = yaml.safe_load(p.read_text(encoding="utf-8"))
    print(f"[meta] loaded {p}: d_model={meta.get('d_model')} "
          f"extraction={meta.get('extraction')}", flush=True)
    return meta


def hf_revision(path_or_id: str) -> str | None:
    try:
        from huggingface_hub import HfApi
        return HfApi().model_info(path_or_id).sha
    except Exception:  # noqa: BLE001
        rp = Path(path_or_id) / "REVISION"
        return rp.read_text().strip() if rp.exists() else None


# ----------------------------- resumable ledger -------------------------------

def load_done(ledger: Path) -> set[tuple]:
    done = set()
    if ledger.exists():
        for line in ledger.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                r = json.loads(line)
                done.add((r["source"], int(r["row_index"]), float(r["alpha"])))
            except Exception:  # noqa: BLE001
                continue
    return done


def append_jsonl(ledger: Path, rec: dict) -> None:
    with open(ledger, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")


# ----------------------------- core AV/round-trip loop ------------------------

def run_rows(client, critic, *, source: str, vecs: np.ndarray, rows: list[int],
             example_ids: list[str], alpha: float, ledger: Path,
             temperature: float, seed: int | None, av_rev: str | None,
             do_score: bool, do_recon: bool, recon_out: Path | None,
             extra_cols=None, concurrency: int | None = None) -> None:
    """Verbalize each row's vector (vecs aligned to rows); optionally score (cos/mse)
    and reconstruct (ĥ). Streams to `ledger`; resumable. extra_cols: row_index->dict.

    AV calls run on a thread pool (concurrency workers): each `client.generate` is an
    httpx POST to sglang that releases the GIL during network I/O, so requests overlap and
    sglang's continuous batcher packs them — the single-stream input_embeds path is ~0.2/s,
    batched it is many×. AR (torch GPU) calls are serialized under a lock (a shared CUDA
    module isn't concurrency-safe); they're cheap (~one short forward) so this isn't the
    bottleneck. Ledger append + recon dict are guarded by a write lock."""
    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed
    if concurrency is None:
        concurrency = _CONCURRENCY
    done = load_done(ledger)
    recon = {} if do_recon else None
    pending = [(j, int(row)) for j, row in enumerate(rows)
               if (source, int(row), float(alpha)) not in done]
    n_total = len(pending)
    ar_lock = threading.Lock()
    write_lock = threading.Lock()
    state = {"n": 0}
    t0 = time.time()

    def work(j: int, row: int):
        # Retry transient sglang failures (RemoteProtocolError "server disconnected",
        # timeouts) so one blip can't crash the whole phase. Last resort: skip the row
        # (left out of the ledger -> a resume pass mops it up) rather than abort.
        text = None
        for attempt in range(6):
            try:
                text = av_verbalize(client, vecs[j], temperature, seed)  # parallel (network I/O)
                break
            except Exception as e:  # noqa: BLE001
                if attempt == 5:
                    with write_lock:
                        print(f"[{source} a={alpha}] row {row} SKIP after 6 retries: "
                              f"{type(e).__name__}: {str(e)[:80]}", flush=True)
                    return
                time.sleep(1.0 * (attempt + 1))
        rec = {
            "source": source, "row_index": int(row),
            "example_id": example_ids[j], "alpha": float(alpha),
            "nl_text": text, "n_tokens": len(text.split()),
            "av_seed": seed, "av_temperature": temperature, "av_rev": av_rev,
        }
        if extra_cols and int(row) in extra_cols:
            rec.update(extra_cols[int(row)])
        h = None
        if do_score or do_recon:
            with ar_lock:  # torch GPU forward: serialize (shared module not thread-safe)
                if do_score:
                    s = ar_score(critic, text, vecs[j])
                    rec["cos_roundtrip"] = float(s.get("cosine_similarity"))
                    rec["mse_roundtrip"] = float(s.get("mse"))
                if do_recon:
                    h = ar_reconstruct(critic, text)
        with write_lock:
            append_jsonl(ledger, rec)
            if h is not None:
                recon[int(row)] = h
            state["n"] += 1
            if state["n"] % 50 == 0:
                rate = state["n"] / max(time.time() - t0, 1e-6)
                print(f"[{source} a={alpha}] {state['n']}/{n_total} new {rate:.1f}/s", flush=True)

    if n_total:
        with ThreadPoolExecutor(max_workers=concurrency) as ex:
            futs = [ex.submit(work, j, row) for j, row in pending]
            for f in as_completed(futs):
                f.result()  # surface any worker exception
    if do_recon and recon_out is not None and recon:
        order = sorted(recon)
        arr = np.stack([recon[r] for r in order]).astype(np.float32)
        np.save(recon_out, arr)
        (recon_out.with_suffix(".rows.json")).write_text(json.dumps(order), encoding="utf-8")
        print(f"[{source} a={alpha}] saved recon {arr.shape} -> {recon_out}", flush=True)
    print(f"[{source} a={alpha}] done: +{state['n']} (ledger {ledger})", flush=True)


# ----------------------------- loaders ----------------------------------------

def load_subset(inputs: Path):
    s = json.loads((inputs / "subset_rows.json").read_text(encoding="utf-8"))
    return s["row_indices"], s["example_ids"]


def load_alpha_array(inputs: Path, alpha: float) -> np.ndarray:
    return np.load(inputs / f"h_layer20_steered_a{alpha_tag(alpha)}.npy", mmap_mode="r")


def fve_fixed(cos_by_alpha: dict[float, np.ndarray], orig_vecs: np.ndarray) -> dict:
    """FVE with fixed denominator Var0 = mean_i ||u_i0 - mean(u0)||^2 over unit-normed orig."""
    u0 = orig_vecs / np.linalg.norm(orig_vecs, axis=1, keepdims=True)
    var0 = float(np.mean(np.sum((u0 - u0.mean(0)) ** 2, axis=1)))  # = 1 - ||mean u0||^2
    out = {}
    for a, cos in cos_by_alpha.items():
        mse = 2.0 * (1.0 - cos)  # unit-normed MSE
        out[a] = {"mean_cos": float(cos.mean()), "mean_mse": float(mse.mean()),
                  "fve_fixed": float(1.0 - mse.mean() / var0), "n": int(cos.size)}
    out["_var0"] = var0
    return out


# ----------------------------- subcommands ------------------------------------

def cmd_calibrate(args):
    meta = load_meta(args.actor)
    client, critic = make_clients(args.actor, args.critic, args.av_url, args.ar_url)
    inputs = Path(args.inputs); out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    rows, ids = load_subset(inputs)
    a0 = load_alpha_array(inputs, 0.0)  # == orig
    vecs = np.asarray(a0[rows], dtype=np.float32)
    av_rev = hf_revision(args.actor)
    ledger = out / "calib_orig.jsonl"
    run_rows(client, critic, source="calib_orig", vecs=vecs, rows=rows, example_ids=ids,
             alpha=0.0, ledger=ledger, temperature=args.temperature, seed=args.seed,
             av_rev=av_rev, do_score=True, do_recon=False, recon_out=None)
    # gather cos
    cos = np.array([json.loads(l)["cos_roundtrip"] for l in ledger.read_text().splitlines() if l.strip()])
    res = fve_fixed({0.0: cos}, vecs)
    fve = res[0.0]["fve_fixed"]; mean_cos = res[0.0]["mean_cos"]
    passed = (CALIB_FVE_MIN <= mean_cos <= CALIB_FVE_MAX) or (CALIB_FVE_MIN <= fve <= CALIB_FVE_MAX)
    report = {"schema_version": "nla_calib.v1", "config_hash": CONFIG_HASH,
              "n": int(cos.size), "mean_cos": mean_cos, "fve_fixed": fve,
              "var0": res["_var0"], "gate_min": CALIB_FVE_MIN, "gate_max": CALIB_FVE_MAX,
              "passed": bool(passed), "meta_extraction": meta.get("extraction"),
              "av_rev": av_rev, "temperature": args.temperature, "seed": args.seed}
    (out / "calibration.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\n[C] FVE(orig) mean_cos={mean_cos:.4f} fve_fixed={fve:.4f} "
          f"PASS={passed} (gate {CALIB_FVE_MIN}-{CALIB_FVE_MAX})", flush=True)
    if not passed:
        print("[C] STOP: calibration gate FAILED — fix normalization/scale/layer before sweeping.",
              file=sys.stderr, flush=True)
        raise SystemExit(3)


def cmd_sweep(args):
    import pyarrow as pa, pyarrow.parquet as pq
    load_meta(args.actor)
    client, critic = make_clients(args.actor, args.critic, args.av_url, args.ar_url)
    inputs = Path(args.inputs); out = Path(args.out)
    (out / "recon").mkdir(parents=True, exist_ok=True)
    rows, ids = load_subset(inputs)
    av_rev = hf_revision(args.actor)
    # join ratio/cos_h_hp from norms.parquet (per row,alpha) for the steered schema
    norms = pq.read_table(inputs / "norms.parquet").to_pydict()
    nidx = {(int(norms["row_index"][i]), float(norms["alpha"][i])):
            (float(norms["ratio"][i]), float(norms["cos_h_hp"][i])) for i in range(len(norms["alpha"]))}
    alphas = [float(a) for a in (args.alphas.split(",") if args.alphas else ALPHAS)]
    ledger = out / "sweep.jsonl"
    for a in alphas:
        arr = load_alpha_array(inputs, a)
        vecs = np.asarray(arr[rows], dtype=np.float32)
        extra = {int(r): {"ratio": nidx.get((int(r), a), (None, None))[0],
                          "cos_h_hp": nidx.get((int(r), a), (None, None))[1]} for r in rows}
        run_rows(client, critic, source="steered", vecs=vecs, rows=rows, example_ids=ids,
                 alpha=a, ledger=ledger, temperature=args.temperature, seed=args.seed,
                 av_rev=av_rev, do_score=True, do_recon=True,
                 recon_out=out / "recon" / f"h_recon_a{alpha_tag(a)}.npy", extra_cols=extra)
    compact_sweep(out, inputs)


def compact_sweep(out: Path, inputs: Path):
    """Ledger -> fve/per_row.parquet, fve/fve_by_alpha.json, nl/steered_a{α}.parquet."""
    import pyarrow as pa, pyarrow.parquet as pq
    recs = [json.loads(l) for l in (out / "sweep.jsonl").read_text().splitlines() if l.strip()]
    by_alpha: dict[float, list] = {}
    for r in recs:
        by_alpha.setdefault(float(r["alpha"]), []).append(r)
    (out / "fve").mkdir(parents=True, exist_ok=True)
    (out / "nl").mkdir(parents=True, exist_ok=True)
    # orig vecs for Var0
    a0 = load_alpha_array(inputs, 0.0); rows, _ = load_subset(inputs)
    orig = np.asarray(a0[rows], dtype=np.float32)
    cos_by_alpha = {a: np.array([x["cos_roundtrip"] for x in rs]) for a, rs in by_alpha.items()}
    fve = fve_fixed(cos_by_alpha, orig)
    (out / "fve" / "fve_by_alpha.json").write_text(json.dumps(fve, indent=2), encoding="utf-8")
    # per_row.parquet
    pr = {k: [] for k in ("alpha", "row_index", "example_id", "cos", "mse")}
    for r in recs:
        pr["alpha"].append(float(r["alpha"])); pr["row_index"].append(int(r["row_index"]))
        pr["example_id"].append(r["example_id"]); pr["cos"].append(r.get("cos_roundtrip"))
        pr["mse"].append(r.get("mse_roundtrip"))
    pq.write_table(pa.table(pr), out / "fve" / "per_row.parquet")
    # steered NL parquet per alpha
    for a, rs in by_alpha.items():
        cols = {k: [] for k in ("example_id", "row_index", "split", "source", "alpha",
                                 "nl_text", "n_tokens", "av_seed", "av_rev", "ratio",
                                 "cos_h_hp", "cos_roundtrip")}
        for r in rs:
            cols["example_id"].append(r["example_id"]); cols["row_index"].append(int(r["row_index"]))
            cols["split"].append("test"); cols["source"].append("steered"); cols["alpha"].append(float(a))
            cols["nl_text"].append(r["nl_text"]); cols["n_tokens"].append(int(r["n_tokens"]))
            cols["av_seed"].append(r.get("av_seed")); cols["av_rev"].append(r.get("av_rev"))
            cols["ratio"].append(r.get("ratio")); cols["cos_h_hp"].append(r.get("cos_h_hp"))
            cols["cos_roundtrip"].append(r.get("cos_roundtrip"))
        pq.write_table(pa.table(cols), out / "nl" / f"steered_a{alpha_tag(a)}.parquet")
    print(f"[D] compacted: {len(recs)} rows -> fve/ + nl/ ; FVE by alpha written", flush=True)


def cmd_verbalize_orig(args):
    import pyarrow as pa, pyarrow.parquet as pq
    load_meta(args.actor)
    client, critic = make_clients(args.actor, args.critic, args.av_url, args.ar_url)
    inputs = Path(args.inputs); out = Path(args.out); (out / "nl").mkdir(parents=True, exist_ok=True)
    orig = np.load(inputs / "h_layer20_steered_a0.npy", mmap_mode="r")  # == orig
    full_ids = json.loads(Path(args.exp04_ids).read_text())["example_ids"] if args.exp04_ids else None
    n = orig.shape[0]
    rows = list(range(n))
    example_ids = full_ids if full_ids else [str(i) for i in rows]
    norms = pq.read_table(inputs / "norms.parquet").to_pydict()
    hnorm = {int(norms["row_index"][i]): float(norms["h_norm"][i])
             for i in range(len(norms["alpha"])) if float(norms["alpha"][i]) == 0.0}
    extra = {r: {"h_norm": hnorm.get(r)} for r in rows}
    av_rev = hf_revision(args.actor)
    ledger = out / "orig.jsonl"
    run_rows(client, critic, source="orig", vecs=np.asarray(orig, dtype=np.float32), rows=rows,
             example_ids=example_ids, alpha=0.0, ledger=ledger, temperature=args.temperature,
             seed=args.seed, av_rev=av_rev, do_score=bool(args.score), do_recon=False,
             recon_out=None, extra_cols=extra)
    recs = [json.loads(l) for l in ledger.read_text().splitlines() if l.strip()]
    cols = {k: [] for k in ("example_id", "row_index", "split", "source", "nl_text",
                            "n_tokens", "av_seed", "av_rev", "h_norm")}
    for r in recs:
        cols["example_id"].append(r["example_id"]); cols["row_index"].append(int(r["row_index"]))
        cols["split"].append("all"); cols["source"].append("orig"); cols["nl_text"].append(r["nl_text"])
        cols["n_tokens"].append(int(r["n_tokens"])); cols["av_seed"].append(r.get("av_seed"))
        cols["av_rev"].append(r.get("av_rev")); cols["h_norm"].append(r.get("h_norm"))
    pq.write_table(pa.table(cols), out / "nl" / "orig.parquet")
    print(f"[E] orig.parquet: {len(recs)} rows", flush=True)


def cmd_verbalize_headline(args):
    import pyarrow as pa, pyarrow.parquet as pq
    load_meta(args.actor)
    client, critic = make_clients(args.actor, args.critic, args.av_url, args.ar_url)
    inputs = Path(args.inputs); out = Path(args.out); (out / "nl").mkdir(parents=True, exist_ok=True)
    # full test rows from norms.parquet (split==test, dedup), full example_ids
    norms = pq.read_table(inputs / "norms.parquet").to_pydict()
    test_rows = sorted({int(norms["row_index"][i]) for i in range(len(norms["alpha"]))
                        if norms["split"][i] == "test"})
    id_by_row = {int(norms["row_index"][i]): norms["example_id"][i] for i in range(len(norms["alpha"]))}
    av_rev = hf_revision(args.actor)
    ledger = out / "headline.jsonl"
    for a in HEADLINE_ALPHAS:
        arr = load_alpha_array(inputs, a)
        vecs = np.asarray(arr[test_rows], dtype=np.float32)
        ids = [id_by_row[r] for r in test_rows]
        run_rows(client, critic, source="headline", vecs=vecs, rows=test_rows, example_ids=ids,
                 alpha=a, ledger=ledger, temperature=args.temperature, seed=args.seed,
                 av_rev=av_rev, do_score=bool(args.score), do_recon=False, recon_out=None)
    recs = [json.loads(l) for l in ledger.read_text().splitlines() if l.strip()]
    by_alpha: dict[float, list] = {}
    for r in recs:
        by_alpha.setdefault(float(r["alpha"]), []).append(r)
    for a, rs in by_alpha.items():
        cols = {k: [] for k in ("example_id", "row_index", "split", "source", "alpha",
                                "nl_text", "n_tokens", "av_seed", "av_rev")}
        for r in rs:
            cols["example_id"].append(r["example_id"]); cols["row_index"].append(int(r["row_index"]))
            cols["split"].append("test"); cols["source"].append("headline"); cols["alpha"].append(float(a))
            cols["nl_text"].append(r["nl_text"]); cols["n_tokens"].append(int(r["n_tokens"]))
            cols["av_seed"].append(r.get("av_seed")); cols["av_rev"].append(r.get("av_rev"))
        pq.write_table(pa.table(cols), out / "nl" / f"headline_a{alpha_tag(a)}.parquet")
    print(f"[E] headline parquet: {len(recs)} rows across {len(by_alpha)} alpha", flush=True)


def cmd_rescale(args):
    """H3 control: round-trip the subset at one alpha for c in {0.5,1,2} (c=1 reused from
    sweep). The AV client normalizes by injection_scale/||v||, so the injected vector is
    identical across c -> FVE must be invariant (within T>0 sampling noise). Proves any
    FVE drop under steering is DIRECTIONAL, not magnitude."""
    load_meta(args.actor)
    client, critic = make_clients(args.actor, args.critic, args.av_url, args.ar_url)
    inputs = Path(args.inputs); out = Path(args.out); (out / "fve").mkdir(parents=True, exist_ok=True)
    rows, ids = load_subset(inputs)
    arr = load_alpha_array(inputs, args.alpha)
    base = np.asarray(arr[rows], dtype=np.float32)
    result = {"alpha": args.alpha, "by_c": {}}
    ledger = out / "rescale.jsonl"
    for c in (0.5, 2.0):
        run_rows(client, critic, source=f"rescale_c{c}", vecs=base * np.float32(c), rows=rows,
                 example_ids=ids, alpha=args.alpha, ledger=ledger, temperature=args.temperature,
                 seed=args.seed, av_rev=hf_revision(args.actor), do_score=True, do_recon=False,
                 recon_out=None)
    for line in ledger.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line); src = r["source"]
        result["by_c"].setdefault(src, []).append(r["cos_roundtrip"])
    result["by_c"] = {k: {"mean_cos": float(np.mean(v)), "n": len(v)} for k, v in result["by_c"].items()}
    (out / "fve" / "rescale_control.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"[H3] rescale control @alpha={args.alpha}: {result['by_c']}", flush=True)


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="On-pod NLA round-trip + verbalization driver.")
    ap.add_argument("--actor", default="./actor_hf"); ap.add_argument("--critic", default="./critic_hf")
    ap.add_argument("--av-url", default="http://localhost:30000")
    ap.add_argument("--ar-url", default="http://localhost:30001")
    ap.add_argument("--inputs", default="./inputs"); ap.add_argument("--out", default="./out")
    ap.add_argument("--temperature", type=float, default=1.0); ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--concurrency", type=int, default=8,
                    help="parallel AV requests to sglang (the input_embeds path is ~0.2/s single-stream)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("calibrate")
    sp = sub.add_parser("sweep"); sp.add_argument("--alphas", default="")
    so = sub.add_parser("verbalize-orig"); so.add_argument("--exp04-ids", default="")
    so.add_argument("--score", action="store_true")
    sh = sub.add_parser("verbalize-headline"); sh.add_argument("--score", action="store_true")
    sr = sub.add_parser("rescale"); sr.add_argument("--alpha", type=float, default=10.0)
    args = ap.parse_args(argv)
    global _CONCURRENCY
    _CONCURRENCY = args.concurrency
    {"calibrate": cmd_calibrate, "sweep": cmd_sweep, "rescale": cmd_rescale,
     "verbalize-orig": cmd_verbalize_orig, "verbalize-headline": cmd_verbalize_headline}[args.cmd](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
