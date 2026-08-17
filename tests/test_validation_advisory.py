"""Every result says how far its score can be trusted. This FAILS CLOSED.

Live probe: covered end-to-end by scripts/smoke_compose.py, which asserts the
advisory on a real completed job.

The advisory this replaces matched the substring "stub" against
model_version_id. That failed OPEN: loading a real checkpoint changes the id to
face-efficientnet_b4-<hash>, the substring disappears, and every caveat
disappears with it -- silently, at exactly the moment scores start looking
plausible enough to be believed.

So the tests that matter here are the negative ones: an unrecognised level, a
NULL, and a realistic real-model id must all still produce a caveat.
"""
from __future__ import annotations

from df.gateway.app import _validation_advisories
from df.inference.base import (
    VALIDATION_PLACEHOLDER,
    VALIDATION_PRODUCTION,
    VALIDATION_RESEARCH,
)


def job(level, model_version_id="face-efficientnet_b4-9f2a1c4d7e01"):
    return {"model_validation": level, "model_version_id": model_version_id}


def test_placeholder_says_it_is_not_a_detector():
    (advisory,) = _validation_advisories(job(VALIDATION_PLACEHOLDER))
    assert "PLACEHOLDER MODEL" in advisory
    assert "no detection meaning" in advisory


def test_research_checkpoint_is_marked_as_indicative_not_a_finding():
    (advisory,) = _validation_advisories(job(VALIDATION_RESEARCH))
    assert "RESEARCH CHECKPOINT" in advisory
    assert "not from weights validated for this system" in advisory
    assert "indicative, not as a finding" in advisory


def test_a_real_looking_model_id_does_not_suppress_the_caveat():
    """The regression. A convincing id must not buy silence.

    Under the old substring check this exact input produced no advisory at all.
    """
    advisories = _validation_advisories(job(VALIDATION_RESEARCH))
    assert advisories, "a real-looking model id silenced the caveat"


def test_null_validation_produces_the_strongest_caveat_not_none():
    advisories = _validation_advisories(job(None))
    assert advisories, "an unrecorded validation level produced no caveat"
    assert "UNVERIFIED MODEL" in advisories[0]


def test_an_unrecognised_level_fails_closed():
    """Someone adds a level and forgets this function. The default must warn."""
    advisories = _validation_advisories(job("some-future-level"))
    assert advisories, "an unknown level produced no caveat"
    assert "UNVERIFIED MODEL" in advisories[0]


def test_only_production_validated_is_silent():
    """The one level that may omit a caveat, and it cannot be reached without
    an explicit sign-off (see efficientnet._validation_level)."""
    assert _validation_advisories(job(VALIDATION_PRODUCTION)) == []


def test_every_level_except_production_warns():
    """Guards the property rather than the three current cases."""
    from df.inference.base import VALIDATION_LEVELS

    for level in VALIDATION_LEVELS - {VALIDATION_PRODUCTION}:
        assert _validation_advisories(job(level)), f"{level} produced no caveat"
