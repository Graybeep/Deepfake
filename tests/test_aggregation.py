from __future__ import annotations

from df.aggregation import AggregationParams, ScoredItem, aggregate, aggregate_identity


def items(scores, confidence=1.0):
    return [ScoredItem(index=i, score=s, confidence=confidence) for i, s in enumerate(scores)]


def test_never_a_plain_mean_outliers_are_trimmed():
    """CLAUDE.md: never a plain mean. A couple of extreme frames must not carry
    the verdict."""
    scores = [50.0] * 10 + [0.0, 100.0]
    plain_mean = sum(scores) / len(scores)

    result = aggregate(items(scores))

    assert result.items_trimmed > 0
    assert result.score == 50.0
    assert result.score != plain_mean or plain_mean == 50.0


def test_low_confidence_items_are_dropped_not_averaged_in():
    good = [ScoredItem(index=i, score=10.0, confidence=0.9) for i in range(5)]
    garbage = [ScoredItem(index=i + 5, score=95.0, confidence=0.05) for i in range(5)]

    result = aggregate(good + garbage)

    assert result.items_dropped_low_confidence == 5
    assert result.score < 20


def test_confidence_weights_the_mean():
    weighted = aggregate(
        [
            ScoredItem(index=0, score=0.0, confidence=0.9),
            ScoredItem(index=1, score=100.0, confidence=0.3),
            ScoredItem(index=2, score=0.0, confidence=0.9),
        ],
        AggregationParams(trim_frac=0.0),
    )
    # Plain mean would be 33.3; the low-confidence 100 is weighted down.
    assert weighted.score < 20


def test_too_few_usable_items_is_undetermined_not_a_score():
    result = aggregate(items([50.0, 50.0]))

    assert result.score is None
    assert result.items_used == 0


def test_empty_input_is_undetermined():
    result = aggregate([])

    assert result.score is None
    assert "no items to aggregate" in result.notes


def test_all_zero_weights_is_undetermined():
    result = aggregate(
        [ScoredItem(index=i, score=50.0, confidence=0.0) for i in range(10)],
        AggregationParams(min_confidence=0.0),
    )

    assert result.score is None


def test_trimming_never_drops_below_the_scoring_floor():
    result = aggregate(
        items([10.0, 20.0, 30.0, 40.0, 50.0]),
        AggregationParams(trim_frac=0.45, min_items_for_score=3, min_items_for_trim=5),
    )

    assert result.score is not None
    assert result.items_used >= 3


def test_method_and_params_are_reported_for_the_audit_trail():
    result = aggregate(items([10.0] * 10))

    assert result.method == "weighted_trimmed_mean.v1"
    assert result.params["trim_frac"] == 0.10
    assert result.params["min_confidence"] == 0.30


def test_image_identity_aggregation_still_records_a_method():
    result = aggregate_identity(ScoredItem(index=0, score=72.5, confidence=0.8))

    assert result.score == 72.5
    assert result.method == "identity.v1"
    assert result.params  # params recorded even though nothing was aggregated


def test_image_identity_below_confidence_is_undetermined():
    result = aggregate_identity(ScoredItem(index=0, score=95.0, confidence=0.1))

    assert result.score is None
