"""Temperature scaling (Tier 2).

Fit ONCE at launch, per model. The face model and the audio model each get
their own temperature -- sharing one curve across two different modalities
means one of them is miscalibrated, and the score bands in df/bands.py assume
both models have been mapped onto the same scale.

This is a LAUNCH SNAPSHOT. It is not wired to a recalibration pipeline and does
not track drift, so it must never be described as production-validated
calibration.

WHAT FITTING ACTUALLY REQUIRES, and why this is still unfitted
--------------------------------------------------------------
Temperature scaling is a one-parameter logistic regression: it picks the T that
minimises negative log-likelihood of `sigmoid(logit / T)` against **ground-truth
labels** on a held-out set. Labels are not an input that can be approximated.
Without them there is no loss surface, nothing to minimise, and no such thing as
a fitted T.

Real weights landed 2026-08-29, which removed one of the two blockers CLAUDE.md
named. The other one stands: this repository has no labelled held-out set, and
the public evaluation sets already assessed for licensing (FF++, DeepfakeBench,
DFDC itself) are gated, non-commercial, or both.

So `fit_temperature` below is real and tested, and it has never been run on real
data. Both temperatures remain 1.0. **Do not invent a T.** A fabricated
temperature is strictly worse than 1.0: 1.0 is visibly the identity and reads as
"nothing applied", while a plausible-looking 1.7 reads as "someone measured
this" and nothing in the system could contradict it.

Per-retrain recalibration and isotonic regression stay deferred, and NOT for
scheduling reasons -- they need the same held-out set, and more of it.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

CALIBRATION_METHOD = "temperature.v1"

# The scheme string is DERIVED from whether a fit actually happened, never
# hardcoded. It used to be the constant "temperature.v1:launch-snapshot",
# attached to every result the torch backend produced while T was 1.0 and
# `fitted_on` said "NOT YET FITTED" -- the audit trail asserting a calibration
# snapshot that had never been taken, in the single field a reader would consult
# to find out whether one had.
SCHEME_UNFITTED = f"{CALIBRATION_METHOD}:unfitted"
SCHEME_LAUNCH_SNAPSHOT = f"{CALIBRATION_METHOD}:launch-snapshot"

# Search bounds on the inverse-temperature axis. A T outside roughly
# [0.05, 20] is not a calibration, it is a broken input set, and quietly
# returning a boundary value would present that as a fit.
_MIN_INV_T = 1.0 / 20.0
_MAX_INV_T = 1.0 / 0.05


@dataclass(frozen=True)
class Temperature:
    """T > 1 softens overconfident logits; T < 1 sharpens."""

    value: float
    fitted_on: str  # description of the held-out set, for the audit trail
    # Whether `value` came from minimising NLL against labels. Explicit rather
    # than inferred from `value != 1.0`: a genuine fit can legitimately land on
    # 1.0, and "fitted, and the answer was 1.0" is a different fact from "never
    # fitted". Inferring would collapse them and report the honest case as the
    # dishonest one.
    fitted: bool = False

    @property
    def scheme(self) -> str:
        return SCHEME_LAUNCH_SNAPSHOT if self.fitted else SCHEME_UNFITTED

    def apply(self, logit: float) -> float:
        """Logit -> calibrated probability -> 0-100 score."""
        if self.value <= 0:
            raise ValueError("temperature must be positive")
        p = 1.0 / (1.0 + math.exp(-logit / self.value))
        return round(p * 100.0, 4)


def _nll(logits: list[float], labels: list[int], inv_t: float) -> float:
    """Mean binary cross-entropy of sigmoid(logit * inv_t) against labels.

    Parameterised by the INVERSE temperature on purpose. With w = 1/T this is
    exactly a one-weight logistic regression with no intercept, whose NLL is
    convex in w -- so a bounded unimodal search finds the global minimum. In T
    itself the objective is not convex, and a search there can settle in the
    wrong place with nothing to indicate it did.
    """
    total = 0.0
    for z, y in zip(logits, labels):
        s = z * inv_t
        # log(1 + exp(s)), evaluated stably at both tails.
        softplus = s + math.log1p(math.exp(-s)) if s > 0 else math.log1p(math.exp(s))
        total += softplus - (y * s)
    return total / len(logits)


def fit_temperature(
    logits: list[float],
    labels: list[int],
    *,
    fitted_on: str,
    tol: float = 1e-6,
) -> Temperature:
    """Fit T by minimising NLL against ground-truth labels.

    `labels` are 1 for manipulated, 0 for authentic. `logits` are the model's
    raw pre-sigmoid outputs -- NOT the 0-100 scores this service reports, which
    have already had a sigmoid applied and cannot be un-applied without knowing
    the temperature that produced them.

    Golden-section search on the inverse temperature; see `_nll` for the axis.
    Deterministic and dependency-free on purpose: a calibration has to be
    reproducible from the audit trail years later, and pinning a numerical
    optimiser's version is a heavier commitment than forty lines of search.
    """
    if len(logits) != len(labels):
        raise ValueError(f"got {len(logits)} logits and {len(labels)} labels")
    if not logits:
        raise ValueError("cannot fit a temperature on an empty set")
    if set(labels) - {0, 1}:
        raise ValueError("labels must be 0 (authentic) or 1 (manipulated)")
    # One-class data carries no calibration signal: NLL is then minimised by
    # driving the temperature to a bound, which would be returned looking like
    # a confident fit.
    if len(set(labels)) < 2:
        raise ValueError(
            "held-out set contains only one class; temperature is unidentifiable "
            "and any value returned would be an artefact of the search bound"
        )

    lo, hi = _MIN_INV_T, _MAX_INV_T
    phi = (math.sqrt(5.0) - 1.0) / 2.0
    a, b = hi - phi * (hi - lo), lo + phi * (hi - lo)
    fa, fb = _nll(logits, labels, a), _nll(logits, labels, b)
    while hi - lo > tol:
        if fa < fb:
            hi, b, fb = b, a, fa
            a = hi - phi * (hi - lo)
            fa = _nll(logits, labels, a)
        else:
            lo, a, fa = a, b, fb
            b = lo + phi * (hi - lo)
            fb = _nll(logits, labels, b)

    return Temperature(value=1.0 / ((lo + hi) / 2.0), fitted_on=fitted_on, fitted=True)


def expected_calibration_error(
    probabilities: list[float], labels: list[int], bins: int = 10
) -> float:
    """ECE: mean gap between confidence and accuracy, weighted by bin size.

    Reported either side of a fit as the diagnostic that says whether it helped.
    It is deliberately NOT the fitting objective: ECE is insensitive to the sign
    of miscalibration within a bin and can be driven down by a temperature that
    makes the model uniformly less useful. NLL does the fitting, ECE does the
    reporting.
    """
    if not probabilities:
        raise ValueError("no probabilities")
    total = 0.0
    for i in range(bins):
        lo, hi = i / bins, (i + 1) / bins
        # The last bin is closed, so p == 1.0 is counted rather than dropped.
        members = [
            (p, y)
            for p, y in zip(probabilities, labels)
            if (lo <= p < hi) or (i == bins - 1 and p == hi)
        ]
        if not members:
            continue
        conf = sum(p for p, _ in members) / len(members)
        acc = sum(y for _, y in members) / len(members)
        total += (len(members) / len(probabilities)) * abs(conf - acc)
    return total


# Unfitted. T=1.0 is the identity -- sigmoid(logit/1.0) is sigmoid(logit) -- so
# scores pass through uncalibrated and `scheme` reports "unfitted" rather than
# claiming a snapshot nobody took. See the module docstring for why this cannot
# be fitted here yet, and scripts/fit_calibration.py for the command that does
# it once a labelled held-out set exists.
FACE_TEMPERATURE = Temperature(
    value=1.0, fitted_on="NOT YET FITTED (no labelled held-out set)", fitted=False
)
AUDIO_TEMPERATURE = Temperature(
    value=1.0, fitted_on="NOT YET FITTED (no labelled held-out set)", fitted=False
)
