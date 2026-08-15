"""Shared pipeline result type and the multi-face rollup rule."""
from __future__ import annotations

from dataclasses import dataclass, field

from df.aggregation import AggregationResult, ScoredItem
from df.bands import Routing
from df.inference.base import ModelVersion


@dataclass
class PipelineResult:
    items: list[ScoredItem]
    aggregation: AggregationResult
    routing: Routing
    model_version: ModelVersion
    # Faces found. 0 => undetermined, and the pipeline must not have produced a
    # real/fake verdict. NULL for audio, which has no faces.
    face_count: int | None
    notes: list[str] = field(default_factory=list)

    @property
    def score(self) -> float | None:
        return self.aggregation.score
