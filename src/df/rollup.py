"""Per-item multi-face rollup.

Lives apart from the router worker so it can be tested without dragging in
Postgres, and so the rule has one home rather than being restated at each call
site.
"""
from __future__ import annotations

from typing import Any

from df.aggregation import ScoredItem
from df.bands import worst_case


def rollup_items(rows: list[dict[str, Any]]) -> tuple[list[ScoredItem], int]:
    """Group per-face score rows by frame/item and take worst-case severity.

    CLAUDE.md: >1 face -> score each face, roll up to worst-case severity for the
    video/image-level class. The surviving item keeps the confidence of the face
    that SET the score, so a worst case driven by a marginal detection is
    weighted down during aggregation instead of counting at full strength.

    Returns (items, max_faces_on_any_item).
    """
    by_index: dict[int, list[dict]] = {}
    for r in rows:
        by_index.setdefault(r["item_index"], []).append(r)

    items: list[ScoredItem] = []
    max_faces = 0
    for index in sorted(by_index):
        group = by_index[index]
        max_faces = max(max_faces, sum(1 for g in group if g.get("face_index") is not None))
        top = max(group, key=lambda g: g["score"])
        items.append(
            ScoredItem(
                index=index,
                score=worst_case([g["score"] for g in group]),
                confidence=top["confidence"],
                kind=group[0]["item_kind"],
                face_index=top.get("face_index"),
            )
        )
    return items, max_faces


# How many per-face rows a result reports. A long video can carry thousands of
# faces and the response is not the place to stream them; the rows remain in
# job_items either way, which is the audit trail. Ranked by score, so the faces
# that could have set the label are the ones that survive the cap.
MAX_REPORTED_FACES = 10


def face_evidence(
    rows: list[dict[str, Any]],
    limit: int = MAX_REPORTED_FACES,
    discarded: dict[str, Any] | None = None,
) -> dict:
    """Per-face detail behind a rolled-up label, as an array rather than a scalar.

    The worst-case rollup is a lossy reduction: N faces become one number and
    one class, and every downstream consumer inherits that reduction with no way
    to see what it discarded. "manipulated" is not triageable. "3 faces over 80,
    the highest is 38x41px in frame 412 of 47 faces total" is triageable in a
    second, and it is the same information the rollup already had and threw
    away.

    Reporting this does NOT decide anything. There is deliberately no per-face
    threshold here and no `flagged` count, because a per-face bar is exactly
    what this codebase cannot yet justify: it would need a false-positive rate
    measured per size bucket, which needs labelled validation data and a scorer
    whose output means something. Inventing a bar to make the field look
    complete would bake a second unvalidated constant into the contract while
    claiming to fix the first.

    So this reports observations and lets the consumer set its own bar. When the
    validation data exists, the threshold policy can change without a schema
    change, because the schema stopped being a scalar.
    """
    faces = [r for r in rows if r.get("face_index") is not None]
    ranked = sorted(faces, key=lambda r: r["score"], reverse=True)
    sized = [r for r in faces if r.get("face_w") is not None]

    disc = discarded or {}
    return {
        # Detections the confidence floor rejected before they reached the
        # model. Reported rather than hidden: a reader seeing "discarded, 32%
        # confidence" can tell a gated artefact from a face that scored low,
        # and those are very different facts. NULL-ish (absent) when nothing
        # recorded it, which is not the same as "none were discarded".
        "detections_discarded": disc.get("detections_discarded"),
        "min_detection_confidence": disc.get("min_detection_confidence"),
        "discarded_faces": disc.get("discarded") or [],
        "faces_total": len(faces),
        "faces_reported": min(len(ranked), limit),
        # How many of those faces carry a recorded size. Every row written
        # before migration 006 has none, and so does anything the stub
        # extractor produced, so a caller must not read a missing face_w as
        # "small". geometry_available says whether the field means anything at
        # all for this job; faces_with_geometry says how much of it is covered.
        "faces_with_geometry": len(sized),
        "geometry_available": bool(sized),
        "top_faces": [
            {
                "item_index": r["item_index"],
                "face_index": r["face_index"],
                "score": r["score"],
                "confidence": r["confidence"],
                "face_w": r.get("face_w"),
                "face_h": r.get("face_h"),
            }
            for r in ranked[:limit]
        ],
    }
