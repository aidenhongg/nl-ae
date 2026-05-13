"""``RunPlan``: enumerates (item, permutation_id, template_id) visits."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from nl_ae.data.canonical import CanonicalItem
from nl_ae.schema.models import ItemIdStr

PermutationMode = Literal["identity", "seeded", "enumerated"]
TemplateIdStr = Annotated[
    str, StringConstraints(min_length=1, max_length=64, pattern=r"^[a-z0-9_\-\.]+$")
]


@dataclass(frozen=True)
class EvalVisit:
    item: CanonicalItem
    permutation_id: int
    template_id: str


class RunPlan(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    template_ids: tuple[TemplateIdStr, ...] = Field(min_length=1)
    permutation_mode: PermutationMode = "seeded"
    permutations_per_item: Annotated[int, Field(ge=1, le=120)] = 1
    enumerated_cap: Annotated[int, Field(ge=1, le=24)] = 24
    item_filter: tuple[ItemIdStr, ...] | None = None

    @property
    def permutation_ids(self) -> tuple[int, ...]:
        return tuple(range(self.permutations_per_item))

    def iter_visits(self, items: Iterable[CanonicalItem]) -> Iterator[EvalVisit]:
        allow = set(self.item_filter) if self.item_filter is not None else None
        for item in items:
            if allow is not None and item.item_id not in allow:
                continue
            for permutation_id in self.permutation_ids:
                for template_id in self.template_ids:
                    yield EvalVisit(
                        item=item, permutation_id=permutation_id, template_id=template_id
                    )


__all__ = ["EvalVisit", "PermutationMode", "RunPlan", "TemplateIdStr"]
