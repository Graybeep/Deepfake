"""Temperature scaling (Tier 2).

Fit ONCE at launch, per model. The face model and the audio model each get
their own temperature -- sharing one curve across two different modalities
means one of them is miscalibrated, and the score bands in df/bands.py assume
both models have been mapped onto the same scale.

This is a LAUNCH SNAPSHOT. It is not wired to a recalibration pipeline and does
not track drift, so it must never be described as production-validated
calibration.

Per-retrain recalibration and isotonic regression stay deferred, and NOT for
scheduling reasons -- both need real trained weights and a held-out set, and
neither exists yet. More time does not unblock them; weights do.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

CALIBRATION_SCHEME = "temperature.v1:launch-snapshot"


@dataclass(frozen=True)
class Temperature:
    """T > 1 softens overconfident logits; T < 1 sharpens."""

    value: float
    fitted_on: str  # description of the held-out set, for the audit trail

    def apply(self, logit: float) -> float:
        """Logit -> calibrated probability -> 0-100 score."""
        if self.value <= 0:
            raise ValueError("temperature must be positive")
        p = 1.0 / (1.0 + math.exp(-logit / self.value))
        return round(p * 100.0, 4)


# Placeholders until the launch calibration pass runs against a held-out set.
# T=1.0 means "no calibration applied yet", which is the honest state today.
FACE_TEMPERATURE = Temperature(value=1.0, fitted_on="NOT YET FITTED (placeholder)")
AUDIO_TEMPERATURE = Temperature(value=1.0, fitted_on="NOT YET FITTED (placeholder)")
