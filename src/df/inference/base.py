"""Detector interface and model identity.

model_version_id is written onto every job row and is the only thing tying a
stored score back to the thing that produced it. It must never be aspirational:
if the stub scorer ran, the id says `stub`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

# How much a score from these weights can be relied on. Distinct from
# is_real_detector, which only says whether a model ran at all: a research
# checkpoint IS a real detector and is NOT validated for use here, and
# collapsing those two into one boolean is what let a plausible score reach a
# caller with no caveat.
VALIDATION_PLACEHOLDER = "placeholder"
VALIDATION_RESEARCH = "research-checkpoint"
VALIDATION_PRODUCTION = "production-validated"

VALIDATION_LEVELS = frozenset(
    {VALIDATION_PLACEHOLDER, VALIDATION_RESEARCH, VALIDATION_PRODUCTION}
)


@dataclass(frozen=True)
class ModelVersion:
    """Identity of the weights that produced a score.

    `is_real_detector=False` marks a backend that does not actually detect
    anything. Anything surfacing a verdict must check this before presenting a
    score as a detection result.

    `validation` is the separate question of whether a real detector's output
    means anything here. It travels onto the job row so a reader is never left
    inferring trustworthiness from the shape of a model id -- the previous
    advisory matched the substring "stub", which meant loading any real
    checkpoint silently removed every caveat from every result.
    """

    model_version_id: str
    architecture: str
    modality: str           # "face" (video + image) or "audio"
    weights_sha256: str | None
    calibration: str        # e.g. "temperature.v1:launch-snapshot" or "none"
    is_real_detector: bool
    validation: str         # one of VALIDATION_LEVELS

    def __post_init__(self) -> None:
        if self.validation not in VALIDATION_LEVELS:
            raise ValueError(
                f"unknown validation level {self.validation!r}; "
                f"expected one of {sorted(VALIDATION_LEVELS)}"
            )


@dataclass(frozen=True)
class Prediction:
    score: float        # 0-100, calibrated. Higher = more likely manipulated.
    confidence: float   # 0-1. For faces this is detection/alignment confidence.
    # Raw pre-sigmoid model output, before any temperature was applied.
    #
    # Carried because a temperature can ONLY be fitted from logits: `score` has
    # already had a sigmoid and a temperature applied, and neither can be undone
    # without knowing the temperature you are trying to find. Without this field
    # the calibration set would have to be scored by a second code path, and a
    # temperature fitted on preprocessing that differs from production is fitted
    # for a distribution production never sees.
    #
    # In-memory only. Deliberately NOT persisted: it is an input to a one-off
    # fitting run, not part of the audit trail, and a column would invite
    # re-deriving stored verdicts from it. None for backends with no meaningful
    # logit -- the stub scores by hashing and has none.
    logit: float | None = None


class Detector(Protocol):
    @property
    def version(self) -> ModelVersion: ...

    def predict_batch(self, inputs: list) -> list[Prediction]: ...
