"""Bundle of per-visit identity fields used by the eval orchestrator."""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from nl_ae.schema.models import ItemIdStr, LetterStr, Sha256Hex


class PromptIdentity(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    item_id: ItemIdStr
    dataset_name: Annotated[str, StringConstraints(min_length=1, max_length=64)]
    dataset_split: Annotated[str, StringConstraints(min_length=1, max_length=32)]
    subject: str | None
    template_id: Annotated[str, StringConstraints(min_length=1, max_length=64)]
    template_content_hash: Sha256Hex
    permutation_id: Annotated[int, Field(ge=0)]
    n_options: Annotated[int, Field(ge=2, le=26)]
    letters: tuple[LetterStr, ...]
    gold_letter: LetterStr | None
    prompt_hash: Sha256Hex
    chat_template_hash: Sha256Hex | None


__all__ = ["PromptIdentity"]
