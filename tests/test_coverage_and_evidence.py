"""Coverage on every verdict, and per-face evidence instead of a bare label.

Both exist for the same reason: a decision was being reduced to a scalar and the
inputs to it discarded, so no consumer could tell a thin result from a
well-covered one, or a crowd scene from a close-up. Neither change alters a
label. They stop the label being the only artifact.

Neither is a substitute for the thresholds that should eventually govern this.
A per-face bar fitted per size bucket, and a video floor derived from score
variance at k, both need labelled validation data and a scorer whose output
means something; there is none, and the only scores this system has produced
come from a hash of the input bytes. These tests pin the reporting contract, not
a tuned policy.

# FakeDb / in-process only: no live probe covers face geometry end to end yet.
# The columns land via migration 006; scripts/verify_attribution.py is the probe
# that would prove they survive a real INSERT, and it does not check them.
"""
from __future__ import annotations

import pytest

from df.aggregation import (
    AggregationParams,
    ScoredItem,
    aggregate,
    aggregate_identity,
)
from df.rollup import MAX_REPORTED_FACES, face_evidence


def _items(n: int, *, score: float = 50.0, confidence: float = 0.9) -> list[ScoredItem]:
    return [ScoredItem(index=i, score=score, confidence=confidence) for i in range(n)]


# --- the modality split -----------------------------------------------------


def test_the_floor_is_per_modality():
    """An image is a complete observation; a video frame is one sample."""
    assert AggregationParams.for_media("image").min_items_for_score == 1
    assert AggregationParams.for_media("video").min_items_for_score == 3
    assert AggregationParams.for_media("audio").min_items_for_score == 3


def test_an_unknown_media_type_raises_rather_than_defaulting():
    """Silently handing back the video floor for an unrecognised modality would
    apply a sampling rule to something whose sampling behaviour is unknown."""
    with pytest.raises(ValueError, match="unknown media_type"):
        AggregationParams.for_media("hologram")


def test_a_single_image_scores_rather_than_returning_undetermined():
    result = aggregate_identity(_items(1, score=72.0)[0])
    assert result.score == 72.0


def test_an_image_result_records_the_floor_that_actually_governed_it():
    """The audit-trail half of the bug, and the part that was actually wrong.

    aggregate_identity never consulted min_items_for_score -- correctly, an
    image has one item by construction -- but it recorded the generic 3 on the
    job row anyway. Every image result asserted a parameter that had not been
    applied to it, which is the job row claiming a rule the code did not run.
    """
    assert aggregate_identity(_items(1)[0]).params["min_items_for_score"] == 1


def test_video_still_refuses_to_score_below_its_floor__with_a_positive_control():
    """The negative check, and its sibling in the same setup.

    A block asserting "this returns undetermined" proves nothing on its own: a
    setup broken so that nothing ever scores would satisfy it. The k=3 case is
    the control that shows the floor, not the harness, is what refused.
    """
    params = AggregationParams.for_media("video")

    assert aggregate(_items(2), params).score is None      # below the floor
    assert aggregate(_items(3), params).score is not None   # at it


# --- coverage ---------------------------------------------------------------


def test_coverage_is_reported_at_every_k_including_one():
    """The objection to a k=1 verdict is that it looks identical to a k=50 one.
    The answer is to mark the difference, not to withhold the verdict."""
    assert aggregate_identity(_items(1)[0]).coverage == 1.0

    # 4 usable of 10: above the video floor, so this DOES score -- and the
    # score alone would look identical to one computed off all ten.
    thin = aggregate(
        _items(4, confidence=0.9) + _items(6, confidence=0.1),
        AggregationParams.for_media("video"),
    )
    assert thin.score is not None
    assert thin.items_total == 10
    assert thin.coverage == 0.4


def test_coverage_counts_trimming_too_not_only_confidence_drops():
    """Both robustness mechanisms remove items from the score, so both belong in
    the denominator's answer. Ten clean items still report 0.8, because the
    trimmed mean drops one from each tail -- and a reader comparing coverage
    across jobs needs that to be the same measurement every time."""
    result = aggregate(_items(10), AggregationParams.for_media("video"))

    assert result.items_total == 10
    assert result.items_trimmed == 2
    assert result.coverage == 0.8


def test_coverage_distinguishes_a_thin_verdict_from_a_complete_one():
    # 4 items: above the floor, below min_items_for_trim, so nothing is removed.
    complete = aggregate(_items(4), AggregationParams.for_media("video"))
    thin = aggregate(
        _items(4, confidence=0.9) + _items(6, confidence=0.1),
        AggregationParams.for_media("video"),
    )

    assert complete.coverage == 1.0
    assert thin.coverage == 0.4
    # Same score, different coverage: the whole point of reporting it.
    assert complete.score == thin.score


def test_coverage_is_none_not_zero_when_nothing_was_extracted():
    """0/0 is undefined. A caller must not read "nothing extracted" as "0% of
    what was extracted survived" -- those are different facts, and the
    undetermined path already carries the first one."""
    assert aggregate([]).coverage is None


# --- per-face evidence ------------------------------------------------------


def _face(item_index: int, face_index: int, score: float, **kw) -> dict:
    row = {
        "item_index": item_index,
        "face_index": face_index,
        "score": score,
        "confidence": kw.get("confidence", 0.8),
        "item_kind": "frame",
    }
    row.update({k: v for k, v in kw.items() if k != "confidence"})
    return row


def test_evidence_ranks_by_score_so_the_face_that_set_the_label_is_first():
    """Worst-case rollup means the top-scoring face is the one that decided the
    result. If the cap ever dropped it, the report would omit the only face the
    label can be explained by."""
    rows = [_face(3, 0, 12.0), _face(412, 2, 88.1), _face(9, 1, 40.0)]

    top = face_evidence(rows)["top_faces"]

    assert [f["score"] for f in top] == [88.1, 40.0, 12.0]
    assert top[0]["item_index"] == 412


def test_evidence_reports_the_true_total_even_when_the_list_is_capped():
    """"3 of 47" is the triageable part. Reporting only the capped list would
    make a crowd scene indistinguishable from a close-up again."""
    rows = [_face(i, 0, float(i)) for i in range(MAX_REPORTED_FACES + 25)]

    ev = face_evidence(rows)

    assert ev["faces_total"] == MAX_REPORTED_FACES + 25
    assert ev["faces_reported"] == MAX_REPORTED_FACES
    assert len(ev["top_faces"]) == MAX_REPORTED_FACES


def test_evidence_carries_geometry_when_it_was_recorded():
    rows = [_face(412, 2, 88.1, face_w=38, face_h=41)]

    ev = face_evidence(rows)

    assert ev["geometry_available"] is True
    assert ev["faces_with_geometry"] == 1
    assert (ev["top_faces"][0]["face_w"], ev["top_faces"][0]["face_h"]) == (38, 41)


def test_missing_geometry_is_flagged_rather_than_read_as_a_small_face():
    """Pre-006 rows and the stub extractor record no bbox. A consumer bucketing
    by size must be able to tell "not recorded" from "small", or the first
    fitted threshold will be fitted against a fabricated feature."""
    ev = face_evidence([_face(1, 0, 50.0)])

    assert ev["geometry_available"] is False
    assert ev["faces_with_geometry"] == 0
    assert ev["top_faces"][0]["face_w"] is None


def test_audio_chunks_are_not_reported_as_faces():
    """face_index is NULL for audio. Counting chunks as faces would report a
    face count for a modality that has none."""
    rows = [
        {"item_index": 0, "face_index": None, "score": 70.0, "confidence": 1.0,
         "item_kind": "chunk"},
        _face(1, 0, 30.0),
    ]

    assert face_evidence(rows)["faces_total"] == 1
