"""Worst-case rollup as the router applies it to stored per-face rows."""
from __future__ import annotations

from df.rollup import rollup_items


def row(item_index, score, confidence=0.9, face_index=0, kind="frame"):
    return {
        "item_index": item_index,
        "item_kind": kind,
        "face_index": face_index,
        "score": score,
        "confidence": confidence,
    }


def test_single_face_per_frame_passes_through():
    items, max_faces = rollup_items([row(0, 10.0), row(1, 20.0)])

    assert [i.score for i in items] == [10.0, 20.0]
    assert max_faces == 1


def test_multiple_faces_in_one_frame_roll_up_to_the_worst():
    items, max_faces = rollup_items(
        [row(0, 5.0, face_index=0), row(0, 88.0, face_index=1), row(0, 12.0, face_index=2)]
    )

    assert len(items) == 1
    assert items[0].score == 88.0
    assert items[0].face_index == 1
    assert max_faces == 3


def test_rolled_up_item_carries_the_driving_face_confidence():
    """A worst case set by a marginal detection must not count at full strength."""
    items, _ = rollup_items(
        [row(0, 5.0, confidence=0.95, face_index=0), row(0, 99.0, confidence=0.31, face_index=1)]
    )

    assert items[0].score == 99.0
    assert items[0].confidence == 0.31


def test_items_come_back_in_index_order():
    items, _ = rollup_items([row(5, 1.0), row(1, 2.0), row(3, 3.0)])

    assert [i.index for i in items] == [1, 3, 5]


def test_audio_chunks_have_no_faces():
    items, max_faces = rollup_items(
        [row(0, 40.0, face_index=None, kind="chunk"), row(1, 60.0, face_index=None, kind="chunk")]
    )

    assert max_faces == 0
    assert len(items) == 2


def test_no_rows_yields_nothing():
    items, max_faces = rollup_items([])

    assert items == []
    assert max_faces == 0
