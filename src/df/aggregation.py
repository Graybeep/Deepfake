"""Aggregate per-frame / per-chunk scores into one job-level score.

CLAUDE.md: weighted/trimmed mean, weighted down by detection/alignment
confidence. Never a plain mean -- a plain mean lets a handful of garbage frames
(a blurred face, a bad alignment, a single frame of a passer-by) drag a whole
verdict around, and it gives low-confidence detections the same vote as clean
ones.

The method name and the exact params used are written onto the job row so a
score can be reproduced later. Changing a default here without bumping
`AGGREGATION_METHOD` silently invalidates comparisons against older rows.
"""
from __future__ import annotations

from dataclasses import dataclass, field

AGGREGATION_METHOD = "weighted_trimmed_mean.v1"


@dataclass(frozen=True)
class ScoredItem:
    """One frame, chunk, or face crop that the model scored."""

    index: int
    score: float          # 0-100, calibrated model output
    confidence: float     # 0-1, detection/alignment confidence -> aggregation weight
    kind: str = "frame"   # frame | chunk | image
    face_index: int | None = None


@dataclass(frozen=True)
class AggregationParams:
    # Fraction trimmed off EACH tail of the score distribution before averaging.
    trim_frac: float = 0.10
    # Items below this confidence contribute nothing. They are still recorded in
    # job_items -- dropped, not deleted, so the audit trail shows what was ignored.
    min_confidence: float = 0.30
    # Trimming needs enough items to be meaningful; below this we only weight.
    min_items_for_trim: int = 5
    # Fewer usable items than this => refuse to score (undetermined) rather than
    # publish a verdict off one or two frames.
    min_items_for_score: int = 3

    def as_dict(self) -> dict:
        return {
            "trim_frac": self.trim_frac,
            "min_confidence": self.min_confidence,
            "min_items_for_trim": self.min_items_for_trim,
            "min_items_for_score": self.min_items_for_score,
        }


@dataclass(frozen=True)
class AggregationResult:
    # None => undetermined. Callers must not coerce this to 0.0.
    score: float | None
    method: str
    params: dict
    items_total: int
    items_used: int
    items_dropped_low_confidence: int
    items_trimmed: int
    notes: list[str] = field(default_factory=list)
    # The items that actually produced `score`, after dropping and trimming.
    # Retention uses this to identify the crops that drove a flagged verdict --
    # a dispute over a >80 result needs the evidence that made it >80, not a
    # sample of everything that was ever extracted.
    used_items: list[ScoredItem] = field(default_factory=list)


def aggregate(
    items: list[ScoredItem],
    params: AggregationParams | None = None,
) -> AggregationResult:
    """Confidence-weighted, symmetrically trimmed mean over item scores."""
    params = params or AggregationParams()
    notes: list[str] = []
    total = len(items)

    if total == 0:
        return AggregationResult(
            None, AGGREGATION_METHOD, params.as_dict(), 0, 0, 0, 0,
            ["no items to aggregate"],
        )

    kept = [i for i in items if i.confidence >= params.min_confidence]
    dropped = total - len(kept)
    if dropped:
        notes.append(f"dropped {dropped} item(s) below confidence {params.min_confidence}")

    if len(kept) < params.min_items_for_score:
        notes.append(
            f"only {len(kept)} usable item(s), need {params.min_items_for_score} -> undetermined"
        )
        return AggregationResult(
            None, AGGREGATION_METHOD, params.as_dict(), total, 0, dropped, 0, notes
        )

    kept.sort(key=lambda i: i.score)

    trimmed_count = 0
    if len(kept) >= params.min_items_for_trim and params.trim_frac > 0:
        k = int(len(kept) * params.trim_frac)
        # Never trim so hard that we fall under the scoring floor.
        while k > 0 and len(kept) - 2 * k < params.min_items_for_score:
            k -= 1
        if k > 0:
            kept = kept[k:-k]
            trimmed_count = 2 * k
            notes.append(f"trimmed {k} item(s) from each tail")

    weight_sum = sum(i.confidence for i in kept)
    if weight_sum <= 0:
        notes.append("all surviving weights are zero -> undetermined")
        return AggregationResult(
            None, AGGREGATION_METHOD, params.as_dict(), total, 0, dropped, trimmed_count, notes
        )

    score = sum(i.score * i.confidence for i in kept) / weight_sum

    return AggregationResult(
        score=round(score, 4),
        method=AGGREGATION_METHOD,
        params=params.as_dict(),
        items_total=total,
        items_used=len(kept),
        items_dropped_low_confidence=dropped,
        items_trimmed=trimmed_count,
        notes=notes,
        used_items=list(kept),
    )


def aggregate_identity(item: ScoredItem, params: AggregationParams | None = None) -> AggregationResult:
    """Image pipeline: aggregation is the identity.

    Still goes through this module so the job row records a method + params and
    the image path is auditable the same way video and audio are.
    """
    params = params or AggregationParams()
    if item.confidence < params.min_confidence:
        return AggregationResult(
            None, "identity.v1", params.as_dict(), 1, 0, 1, 0,
            [f"single item below confidence {params.min_confidence} -> undetermined"],
        )
    return AggregationResult(
        round(item.score, 4), "identity.v1", params.as_dict(), 1, 1, 0, 0, [], [item]
    )
