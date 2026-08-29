"""The temperature fitter, checked against data whose true T is known.

There is no labelled held-out set in this project, so the fitter cannot be
validated against a real model. It CAN be validated against synthetic data built
by distorting a calibrated set by a known factor: if the model's logits have been
scaled by k, the temperature that undoes it is exactly k, and a correct fitter
recovers it without being told.

That is a genuine ground truth, and it tests exactly one thing -- the optimiser.
It says nothing whatsoever about whether any real detector is calibrated, and a
green run here must never be read as "calibration is done".

# In-process only, and no live probe is possible for this: a live probe would
# need the labelled held-out set whose absence is the reason the real fit has
# not happened.
"""
from __future__ import annotations

import math
import random

import pytest

from df.gateway.app import _calibration_advisories
from df.inference.calibration import (
    SCHEME_LAUNCH_SNAPSHOT,
    SCHEME_UNFITTED,
    FACE_TEMPERATURE,
    Temperature,
    expected_calibration_error,
    fit_temperature,
)


def _sigmoid(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-z))


def _calibrated_set(n: int = 20000, seed: int = 42) -> tuple[list[float], list[int]]:
    """Logits whose sigmoid IS the true probability, so the correct T is 1.0."""
    rng = random.Random(seed)
    logits = [rng.gauss(0.0, 2.5) for _ in range(n)]
    labels = [1 if rng.random() < _sigmoid(z) else 0 for z in logits]
    return logits, labels


# --- the fitter recovers a temperature it was never told --------------------


@pytest.mark.parametrize("true_t", [0.5, 1.0, 1.8, 3.0])
def test_fit_recovers_a_known_temperature(true_t):
    """Distort a calibrated set by a known factor and fit it back.

    A model that is overconfident by k produces logits k times too large; the
    temperature that corrects it is k by construction. Covers both directions:
    T > 1 softens an overconfident model, T < 1 sharpens an underconfident one.
    A fitter that only ever returned values above 1 would pass a one-sided test.
    """
    logits, labels = _calibrated_set()
    distorted = [z / true_t for z in logits]  # model now miscalibrated by 1/true_t

    fitted = fit_temperature(distorted, labels, fitted_on="synthetic")

    assert fitted.value == pytest.approx(1.0 / true_t, rel=0.08)


def test_fitting_reduces_calibration_error():
    """The point of the exercise. ECE is the diagnostic, not the objective, so
    this is a separate claim from "NLL went down"."""
    logits, labels = _calibrated_set()
    overconfident = [z * 3.0 for z in logits]

    fitted = fit_temperature(overconfident, labels, fitted_on="synthetic")

    before = expected_calibration_error([_sigmoid(z) for z in overconfident], labels)
    after = expected_calibration_error(
        [_sigmoid(z / fitted.value) for z in overconfident], labels
    )
    assert after < before
    # And the improvement is not marginal noise -- an overconfident model is
    # badly calibrated to begin with.
    assert before > 0.05
    assert after < 0.02


def test_an_already_calibrated_set_fits_near_one__the_null_case():
    """The control. A fitter that always returned something dramatic would pass
    the recovery tests above by luck of the parametrisation."""
    logits, labels = _calibrated_set()

    assert fit_temperature(logits, labels, fitted_on="synthetic").value == pytest.approx(
        1.0, rel=0.08
    )


# --- refusing to produce a number that would not mean anything --------------


def test_a_one_class_set_is_refused_rather_than_fitted():
    """With one class, NLL is minimised at a search bound. Returning that would
    hand back a boundary artefact wearing six decimal places."""
    with pytest.raises(ValueError, match="only one class"):
        fit_temperature([1.0, -2.0, 0.5], [1, 1, 1], fitted_on="bad")


def test_an_empty_set_is_refused():
    with pytest.raises(ValueError, match="empty"):
        fit_temperature([], [], fitted_on="bad")


def test_mismatched_lengths_are_refused():
    with pytest.raises(ValueError, match="logits and"):
        fit_temperature([1.0, 2.0], [1], fitted_on="bad")


def test_non_binary_labels_are_refused():
    """Probabilities or -1/1 labels would fit silently and wrongly."""
    with pytest.raises(ValueError, match="must be 0"):
        fit_temperature([1.0, 2.0], [0, 2], fitted_on="bad")


# --- the scheme string must follow reality ----------------------------------


def test_scheme_reports_unfitted_until_a_fit_happens():
    """The bug this replaced: the scheme was the constant
    "temperature.v1:launch-snapshot", stamped onto every result while T was 1.0
    and nothing had been fitted -- the audit trail asserting a snapshot nobody
    took, in the one field a reader would check."""
    assert Temperature(1.0, fitted_on="x", fitted=False).scheme == SCHEME_UNFITTED
    assert Temperature(1.7, fitted_on="x", fitted=True).scheme == SCHEME_LAUNCH_SNAPSHOT
    # The discriminating case, and the only one that separates `fitted` from
    # `value != 1.0`. Added after the mutation harness reported GREEN: the two
    # assertions above both pass under a version that infers fitted-ness from
    # the value, and the fitted-set test below misses it too because a real fit
    # lands NEAR 1.0 rather than exactly on it.
    assert Temperature(1.0, fitted_on="x", fitted=True).scheme == SCHEME_LAUNCH_SNAPSHOT


def test_a_genuine_fit_landing_on_one_is_not_reported_as_unfitted():
    """Why `fitted` is a field and not `value != 1.0`. A real fit can land on
    1.0; that is "measured, and the answer was no correction needed", which is a
    different fact from "never measured". Inferring collapses them and reports
    the honest case as the dishonest one."""
    logits, labels = _calibrated_set()

    fitted = fit_temperature(logits, labels, fitted_on="synthetic")

    assert fitted.value == pytest.approx(1.0, rel=0.08)
    assert fitted.scheme == SCHEME_LAUNCH_SNAPSHOT


def test_the_shipped_temperatures_are_still_honestly_unfitted():
    """Guards the thing most likely to go wrong later: someone pastes a fitted
    value in without flipping `fitted`, or flips `fitted` without fitting. If
    this test starts failing because a real calibration landed, update it in the
    same commit as the fit and say what the held-out set was."""
    assert FACE_TEMPERATURE.value == 1.0
    assert FACE_TEMPERATURE.fitted is False
    assert FACE_TEMPERATURE.scheme == SCHEME_UNFITTED
    assert "NOT YET FITTED" in FACE_TEMPERATURE.fitted_on


def test_temperature_of_one_is_the_identity():
    """T=1.0 must not quietly alter scores: sigmoid(z/1) is sigmoid(z)."""
    t = Temperature(1.0, fitted_on="identity")

    for z in (-4.0, -0.5, 0.0, 0.5, 4.0):
        assert t.apply(z) == pytest.approx(_sigmoid(z) * 100.0, abs=1e-3)


def test_a_non_positive_temperature_is_rejected():
    with pytest.raises(ValueError, match="must be positive"):
        Temperature(0.0, fitted_on="bad").apply(1.0)


# --- the advisory that tells a reader the number is not a probability -------


def _advice(job: dict) -> str:
    out = _calibration_advisories(job)
    return out[0] if out else ""


def test_an_uncalibrated_real_model_says_so():
    """The gap this closes. A research-checkpoint caveat says nothing about
    scale: it warns that the weights are unvalidated, while the reader is still
    looking at a raw sigmoid being presented as a 0-100 figure."""
    advice = _advice({
        "model_validation": "research-checkpoint",
        "calibration": SCHEME_UNFITTED,
    })

    assert "UNCALIBRATED" in advice
    assert "not a calibrated probability" in advice


def test_a_fitted_snapshot_is_still_caveated_but_differently():
    advice = _advice({
        "model_validation": "research-checkpoint",
        "calibration": SCHEME_LAUNCH_SNAPSHOT,
    })

    assert "LAUNCH-SNAPSHOT" in advice
    assert "drift" in advice


def test_unrecorded_calibration_warns_rather_than_staying_silent():
    """Fails closed, like the validation advisory beside it. NULL is a pre-007
    row and an unknown string is a scheme someone added without updating this
    function; neither is a reason to reassure the reader."""
    assert "UNRECORDED" in _advice({"model_validation": "research-checkpoint"})
    assert "UNRECORDED" in _advice({
        "model_validation": "research-checkpoint", "calibration": "isotonic.v9",
    })


def test_the_stub_gets_no_calibration_advisory():
    """Its PLACEHOLDER caveat already says the number carries no meaning.
    Warning that a meaningless number is also uncalibrated is noise, and noise
    is what teaches readers to skip advisories."""
    assert _calibration_advisories({
        "model_validation": "placeholder", "calibration": SCHEME_UNFITTED,
    }) == []
