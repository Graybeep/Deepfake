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
