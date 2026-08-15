"""Score-band routing.

Bands apply to the AGGREGATED score (df/aggregation.py), never to a raw
per-frame / per-chunk / per-face score. See CLAUDE.md.

Bands are a TOTAL partition of the range: routing has no undefined input.
CLAUDE.md names three thresholds -- <20, 40-60, >80 -- and specifies how the two
gaps behave:

  20-40  auto-clears like <20. Being wrong here just means a probably-real item
         gets deleted on schedule.
  60-80  is the riskier gap: it sits next to the highest-severity threshold,
         whose calibration is not trusted enough to display as a raw percentage.
         It gets the same DB-flag-plus-alert treatment as 40-60, at LOW urgency
         -- never a silent pass-through into normal deletion. It still deletes
         on the normal schedule; the flag is the record that it happened.

Calibration note: the face model and the audio model each get their own
calibration pass, so a "70" from the face model and a "70" from the audio model
are only comparable because each was calibrated onto this shared band scale
first. Do not reuse one temperature across both.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ResultClass(str, Enum):
    AUTHENTIC = "authentic"
    MANIPULATED = "manipulated"
    UNCERTAIN = "uncertain"
    # No usable signal: 0 faces detected, no decodable frames, no audio chunks.
    # Never collapse this into authentic/manipulated.
    UNDETERMINED = "undetermined"


class Band(str, Enum):
    LIKELY_AUTHENTIC = "likely_authentic"        # score < 20        (named in CLAUDE.md)
    LEANING_AUTHENTIC = "leaning_authentic"      # 20 <= score < 40  (auto-clears)
    UNCERTAIN = "uncertain"                      # 40 <= score <= 60 (named in CLAUDE.md)
    LEANING_MANIPULATED = "leaning_manipulated"  # 60 < score <= 80  (low-urgency flag)
    LIKELY_MANIPULATED = "likely_manipulated"    # score > 80        (named in CLAUDE.md)
    UNDETERMINED = "undetermined"                # no score at all


class ReviewUrgency(str, Enum):
    NONE = "none"
    # Passive: recorded and alerted so it is never a silent pass-through, but it
    # is not asking anyone to drop what they are doing.
    LOW = "low"
    NORMAL = "normal"


@dataclass(frozen=True)
class Routing:
    band: Band
    result_class: ResultClass
    # Tier 2: flag into cold storage under a fixed 30-day timer.
    # This is an "extended retention window", NOT a legal hold.
    extended_retention: bool
    # Tier 3 substitute for the human review dashboard: DB flag + Slack/email.
    flag_for_review: bool
    review_reason: str | None = None
    review_urgency: ReviewUrgency = ReviewUrgency.NONE


def route(score: float | None) -> Routing:
    """Map an aggregated score (0-100) to a band and downstream actions.

    `score is None` means the pipeline produced no usable items -- undetermined.
    """
    if score is None:
        return Routing(
            band=Band.UNDETERMINED,
            result_class=ResultClass.UNDETERMINED,
            extended_retention=False,
            flag_for_review=True,
            review_reason="no usable signal (0 faces / no decodable items)",
            review_urgency=ReviewUrgency.NORMAL,
        )

    if not 0.0 <= score <= 100.0:
        raise ValueError(f"aggregated score out of range: {score!r}")

    if score < 20:
        return Routing(Band.LIKELY_AUTHENTIC, ResultClass.AUTHENTIC, False, False)

    if score < 40:
        # Auto-clears like <20. Sanctioned by CLAUDE.md: being wrong here means a
        # probably-real item gets deleted on schedule.
        return Routing(Band.LEANING_AUTHENTIC, ResultClass.AUTHENTIC, False, False)

    if score <= 60:
        return Routing(
            Band.UNCERTAIN,
            ResultClass.UNCERTAIN,
            extended_retention=False,
            flag_for_review=True,
            review_reason="score in uncertain band (40-60)",
            review_urgency=ReviewUrgency.NORMAL,
        )

    if score <= 80:
        # The riskier gap: adjacent to the >80 threshold whose calibration is not
        # trusted enough to show as a raw percentage. Flagged and alerted at low
        # urgency so it is never a silent pass-through -- but it still deletes on
        # the normal schedule and does NOT open the extended retention window.
        return Routing(
            Band.LEANING_MANIPULATED,
            ResultClass.MANIPULATED,
            extended_retention=False,
            flag_for_review=True,
            review_reason="score in 60-80 band, adjacent to the high threshold",
            review_urgency=ReviewUrgency.LOW,
        )

    return Routing(
        Band.LIKELY_MANIPULATED,
        ResultClass.MANIPULATED,
        extended_retention=True,
        flag_for_review=True,
        review_reason="score in high band (>80)",
        review_urgency=ReviewUrgency.NORMAL,
    )


def worst_case(scores: list[float]) -> float | None:
    """Roll several per-face scores up to one item-level score.

    CLAUDE.md: >1 face -> score each face, roll up to worst-case severity.
    Worst case == highest manipulation score. This is a DEFAULT, not fixed;
    confirm before anything downstream assumes it (DECISIONS.md).
    """
    usable = [s for s in scores if s is not None]
    return max(usable) if usable else None
