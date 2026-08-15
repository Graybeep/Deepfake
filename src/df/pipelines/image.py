"""Image pipeline.

Ingest -> Face Extract -> Face Align -> face model -> Router
Aggregation is the identity, but still goes through df.aggregation so the job
row records a method and params like every other pipeline.

Same face model and same model_version_id as the video pipeline.
"""
from __future__ import annotations

from df.aggregation import AggregationParams, ScoredItem, aggregate, aggregate_identity
from df.bands import route, worst_case
from df.inference.registry import get_face_model
from df.pipelines.common import PipelineResult
from df.pipelines.extract import FaceExtractor


def run(
    image_bytes: bytes,
    *,
    extractor: FaceExtractor,
    detector=None,
    params: AggregationParams | None = None,
) -> PipelineResult:
    detector = detector or get_face_model()
    notes: list[str] = []

    crops = extractor.extract(image_bytes, frame_index=0)
    if not crops:
        notes.append("0 faces detected -> undetermined")
        agg = aggregate([], params)
        return PipelineResult([], agg, route(None), detector.version, face_count=0, notes=notes)

    preds = detector.predict_batch([c.data for c in crops])
    face_scores = [p.score for p in preds]

    items = [
        ScoredItem(
            index=i, score=preds[i].score, confidence=crops[i].confidence,
            kind="image", face_index=crops[i].face_index,
        )
        for i in range(len(crops))
    ]

    if len(crops) > 1:
        notes.append(f"{len(crops)} faces, rolled up to worst-case severity")

    top = max(range(len(face_scores)), key=lambda i: face_scores[i])
    rolled = ScoredItem(
        index=0,
        score=worst_case(face_scores),
        confidence=crops[top].confidence,
        kind="image",
        face_index=crops[top].face_index,
    )

    agg = aggregate_identity(rolled, params)
    notes.extend(agg.notes)
    return PipelineResult(
        items=items,
        aggregation=agg,
        routing=route(agg.score),
        model_version=detector.version,
        face_count=len(crops),
        notes=notes,
    )
