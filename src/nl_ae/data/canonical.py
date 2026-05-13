"""Canonical, dataset-agnostic item shape + per-permutation projection."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from nl_ae.schema.models import ItemIdStr, LetterStr

DatasetName = Literal["mmlu", "opinionqa"]


class CanonicalItem(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    item_id: ItemIdStr
    dataset_name: DatasetName
    dataset_split: Annotated[str, StringConstraints(min_length=1, max_length=32)]
    subject: str | None = None
    question: Annotated[str, StringConstraints(min_length=1)]
    choices: tuple[Annotated[str, StringConstraints(min_length=1)], ...] = Field(
        min_length=2, max_length=26
    )
    gold_index: Annotated[int, Field(ge=0)] | None
    metadata: dict[str, str | int | float | bool | None] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check_gold(self) -> "CanonicalItem":
        if self.gold_index is not None and self.gold_index >= len(self.choices):
            raise ValueError(
                f"gold_index={self.gold_index} out of range for n_options={len(self.choices)}"
            )
        return self

    @property
    def n_options(self) -> int:
        return len(self.choices)

    @property
    def letter_set(self) -> tuple[LetterStr, ...]:
        from .permute import letters_for  # noqa: PLC0415

        return letters_for(self.n_options)

    @property
    def gold_letter(self) -> LetterStr | None:
        if self.gold_index is None:
            return None
        return self.letter_set[self.gold_index]


class PermutedItem(BaseModel):
    """One ``(item, permutation_id)`` view: which choices map to which letters."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    base: CanonicalItem
    permutation_id: Annotated[int, Field(ge=0)]
    perm: tuple[int, ...]  # perm[new_pos] = old_index
    letters: tuple[LetterStr, ...]

    @model_validator(mode="after")
    def _check_shapes(self) -> "PermutedItem":
        if len(self.perm) != self.base.n_options:
            raise ValueError(
                f"permutation length {len(self.perm)} != n_options {self.base.n_options}"
            )
        if sorted(self.perm) != list(range(self.base.n_options)):
            raise ValueError("permutation is not a valid bijection")
        if len(self.letters) != self.base.n_options:
            raise ValueError("letters length must equal n_options")
        return self

    @property
    def n_options(self) -> int:
        return self.base.n_options

    @property
    def choices_in_order(self) -> tuple[str, ...]:
        return tuple(self.base.choices[i] for i in self.perm)

    @property
    def gold_letter(self) -> LetterStr | None:
        if self.base.gold_index is None:
            return None
        new_pos = self.perm.index(self.base.gold_index)
        return self.letters[new_pos]


__all__ = ["CanonicalItem", "DatasetName", "PermutedItem"]
