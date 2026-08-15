from __future__ import annotations

import pytest

from df.bands import Band, ResultClass, ReviewUrgency, route, worst_case


@pytest.mark.parametrize(
    "score,band",
    [
        (0.0, Band.LIKELY_AUTHENTIC),
        (19.9, Band.LIKELY_AUTHENTIC),
        (20.0, Band.LEANING_AUTHENTIC),
        (39.9, Band.LEANING_AUTHENTIC),
        (40.0, Band.UNCERTAIN),
        (60.0, Band.UNCERTAIN),
        (60.1, Band.LEANING_MANIPULATED),
        (80.0, Band.LEANING_MANIPULATED),
        (80.1, Band.LIKELY_MANIPULATED),
        (100.0, Band.LIKELY_MANIPULATED),
    ],
)
def test_band_boundaries(score, band):
    assert route(score).band is band


def test_routing_is_total_over_the_whole_range():
    """The three bands CLAUDE.md names leave 20-40 and 60-80 undefined; routing
    still has to be total. Gap-fill behaviour is an assumption (DECISIONS.md)."""
    for i in range(0, 1001):
        assert route(i / 10).band is not None


def test_no_score_is_undetermined_and_flagged():
    routing = route(None)

    assert routing.result_class is ResultClass.UNDETERMINED
    assert routing.band is Band.UNDETERMINED
    assert routing.flag_for_review is True
    assert routing.extended_retention is False


def test_only_the_high_band_opens_the_extended_retention_window():
    assert route(95.0).extended_retention is True
    assert route(75.0).extended_retention is False
    assert route(50.0).extended_retention is False
    assert route(5.0).extended_retention is False


def test_uncertain_band_is_flagged_for_review():
    routing = route(50.0)

    assert routing.result_class is ResultClass.UNCERTAIN
    assert routing.flag_for_review is True
    assert routing.review_urgency is ReviewUrgency.NORMAL


def test_60_to_80_is_never_a_silent_pass_through():
    """CLAUDE.md: the riskier gap. It sits next to the >80 threshold whose
    calibration is not trusted enough to show as a raw percentage, so it gets
    the same DB-flag-plus-alert treatment as 40-60, at low urgency."""
    routing = route(70.0)

    assert routing.flag_for_review is True
    assert routing.review_urgency is ReviewUrgency.LOW
    assert routing.review_reason is not None
    # It still deletes on the normal schedule; the flag is the record.
    assert routing.extended_retention is False


def test_20_to_40_auto_clears_like_the_low_band():
    """Explicitly sanctioned: being wrong here means a probably-real item gets
    deleted on schedule."""
    routing = route(30.0)

    assert routing.result_class is ResultClass.AUTHENTIC
    assert routing.flag_for_review is False
    assert routing.review_urgency is ReviewUrgency.NONE


def test_low_band_is_not_flagged():
    assert route(5.0).flag_for_review is False


def test_every_manipulated_leaning_score_is_recorded_somewhere():
    """No score above the uncertain band may pass without a flag."""
    for score in [60.5, 70.0, 80.0, 80.5, 95.0, 100.0]:
        assert route(score).flag_for_review is True, f"{score} passed silently"


def test_out_of_range_score_raises():
    with pytest.raises(ValueError):
        route(101.0)
    with pytest.raises(ValueError):
        route(-1.0)


def test_worst_case_takes_the_highest_severity():
    assert worst_case([5.0, 92.0, 11.0]) == 92.0


def test_worst_case_of_nothing_is_none():
    assert worst_case([]) is None
