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

import os
from dataclasses import dataclass, field


def _min_item_confidence() -> float:
    """Absolute confidence floor for aggregation. 0.0 = none, which is the
    default; see AggregationParams.min_confidence for why."""
    try:
        return max(0.0, float(os.environ.get("DF_MIN_ITEM_CONFIDENCE", "0.0")))
    except ValueError:
        return 0.0

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
    #
    # DEFAULT 0.0: NO ABSOLUTE FLOOR. Changed 2026-09-01, and the reasoning is
    # already in this repository -- it is the argument that replaced an absolute
    # 0.3 DETECTION floor with a relative ratio: `_haar_confidence` is an
    # unbounded cascade reject level divided by 10, so the quantity has no
    # semantics and "no absolute floor is more justified than another".
    #
    # This floor was documented as never having fired (378 item rows, lowest
    # confidence 0.6, all from the STUB extractor which produces 0.6/0.7/0.8 by
    # construction). Real Haar on real photographs produces 0.044 to 1.000 with
    # no gap anywhere near 0.3, so as soon as detection improved it started
    # firing -- and what it destroyed was correct results.
    #
    # `measured: yes` 2026-09-01. A detection fallback for glasses and bad
    # lighting took hard-case detection from 20/30 to 24/30, and this floor then
    # discarded six of them, turning a scored verdict into `undetermined`:
    #   glasses_gandhi 0.171 | lowlight_douglass 0.044 | lowlight_twain 0.098
    #   shadow_twain 0.252 | shadow_douglass 0.300 | tesla.jpg 0.290
    # `tesla.jpg` is the one that settles it: an ordinary portrait, no hard
    # lighting, refused a verdict because two detections landed at 0.290.
    #
    # The RELATIVE gate in the CPU worker still runs and still drops detections
    # below `DF_DETECTION_CONFIDENCE_RATIO` of the best in frame -- that is the
    # mechanism for "is this a face at all", and it cannot empty a non-empty
    # set. This floor was doing something different and unjustifiable: rejecting
    # weakly-detected REAL faces against an arbitrary constant.
    #
    # Still configurable, so an absolute floor can be restored on a running
    # deployment if a labelled set ever justifies one.
    min_confidence: float = field(default_factory=lambda: _min_item_confidence())
    # Trimming needs enough items to be meaningful; below this we only weight.
    min_items_for_trim: int = 5
    # Fewer usable items than this => refuse to score (undetermined) rather than
    # publish a verdict off one or two frames.
    #
    # 3 IS AN UNVALIDATED PLACEHOLDER for video and audio. It was picked as the
    # smallest number that is more than a couple, not derived from anything.
    # Deriving it means measuring score variance at k=1,2,3,5,10 on validation
    # clips and taking the point where it flattens; there are no validation
    # clips, and the only scores this system has ever produced come from a hash
    # of the input bytes, so no such measurement exists or can exist yet.
    # Recorded here rather than in a doc because a constant that has quietly
    # acquired the authority of being written down is harder to revisit than one
    # that says what it is.
    min_items_for_score: int = 3

    def as_dict(self) -> dict:
        return {
            "trim_frac": self.trim_frac,
            "min_confidence": self.min_confidence,
            "min_items_for_trim": self.min_items_for_trim,
            "min_items_for_score": self.min_items_for_score,
        }

    @classmethod
    def for_media(cls, media_type: str, **overrides) -> "AggregationParams":
        """Per-modality floor. An image is not a sample from anything.

        A rule about sampling variance should not apply to a thing that was not
        sampled: a video frame is one draw from a distribution over frames, an
        image is a complete observation of its subject. Requiring 3 items of a
        single image asks for evidence that cannot exist in principle.

        The image path already behaved this way, by accident rather than
        decision -- `aggregate_identity` never consulted the floor at all. The
        harm was in the audit trail: every image job recorded
        `min_items_for_score: 3` on its row, a parameter that had not governed
        the result. The row asserted a rule the code did not apply.

        Video and audio keep 3, still unvalidated -- see the field comment.
        """
        floors = {"image": 1, "video": 3, "audio": 3}
        if media_type not in floors:
            raise ValueError(f"unknown media_type {media_type!r}")
        return cls(min_items_for_score=floors[media_type], **overrides)


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

    @property
    def coverage(self) -> float | None:
        """Fraction of extracted items that actually produced the score.

        Reported on EVERY verdict, at every k. The objection to scoring off one
        frame is not that the score is wrong, it is that a one-frame verdict and
        a fifty-frame verdict are indistinguishable in the response -- so the
        fix is to mark the difference, not to withhold the verdict.

        Once this is in the response the minimum-items gate stops being the only
        protection a reader has, and a consumer can set its own bar instead of
        inheriting one this codebase cannot yet defend.

        NULL/None when nothing was extracted: 0/0 is undefined, not 0.0.
        """
        if not self.items_total:
            return None
        return round(self.items_used / self.items_total, 4)


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

    Defaults to the image params, not the generic ones. This path never
    consulted `min_items_for_score` -- correctly, an image has exactly one item
    by construction -- but it used to record the generic `3` on the job row
    regardless, so every image result carried a parameter that had not been
    applied to it. The row now records the floor that actually governed it.
    """
    params = params or AggregationParams.for_media("image")
    if item.confidence < params.min_confidence:
        return AggregationResult(
            None, "identity.v1", params.as_dict(), 1, 0, 1, 0,
            [f"single item below confidence {params.min_confidence} -> undetermined"],
        )
    return AggregationResult(
        round(item.score, 4), "identity.v1", params.as_dict(), 1, 1, 0, 0, [], [item]
    )
