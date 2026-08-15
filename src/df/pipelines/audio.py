"""Audio pipeline.

Ingest -> Chunk -> Spectrogram -> audio model -> Aggregate -> Router

Separate model from the face pipeline, with its own model_version_id and its own
calibration temperature. face_count is None here -- audio has no faces, and
reporting 0 would collide with the video/image "0 faces => undetermined" rule.
"""
from __future__ import annotations

from df.aggregation import AggregationParams, ScoredItem, aggregate
from df.bands import route
from df.inference.registry import get_audio_model
from df.pipelines.common import PipelineResult
from df.pipelines.extract import AudioChunker


def run(
    audio_bytes: bytes,
    *,
    chunker: AudioChunker,
    detector=None,
    params: AggregationParams | None = None,
) -> PipelineResult:
    detector = detector or get_audio_model()
    notes: list[str] = []

    chunks = chunker.chunk(audio_bytes)
    if not chunks:
        notes.append("no decodable audio chunks -> undetermined")
        agg = aggregate([], params)
        return PipelineResult([], agg, route(None), detector.version, face_count=None, notes=notes)

    preds = detector.predict_batch([c.spectrogram for c in chunks])
    items = [
        ScoredItem(
            index=chunks[i].index,
            score=preds[i].score,
            confidence=chunks[i].confidence,
            kind="chunk",
        )
        for i in range(len(chunks))
    ]

    agg = aggregate(items, params)
    notes.extend(agg.notes)
    return PipelineResult(
        items=items,
        aggregation=agg,
        routing=route(agg.score),
        model_version=detector.version,
        face_count=None,
        notes=notes,
    )
