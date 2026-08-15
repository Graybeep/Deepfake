"""Detector interface and model identity.

model_version_id is written onto every job row and is the only thing tying a
stored score back to the thing that produced it. It must never be aspirational:
if the stub scorer ran, the id says `stub`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class ModelVersion:
    """Identity of the weights that produced a score.

    `is_real_detector=False` marks a backend that does not actually detect
    anything. Anything surfacing a verdict must check this before presenting a
    score as a detection result.
    """

    model_version_id: str
    architecture: str
    modality: str           # "face" (video + image) or "audio"
    weights_sha256: str | None
    calibration: str        # e.g. "temperature.v1:launch-snapshot" or "none"
    is_real_detector: bool


@dataclass(frozen=True)
class Prediction:
    score: float        # 0-100, calibrated. Higher = more likely manipulated.
    confidence: float   # 0-1. For faces this is detection/alignment confidence.


class Detector(Protocol):
    @property
    def version(self) -> ModelVersion: ...

    def predict_batch(self, inputs: list) -> list[Prediction]: ...
