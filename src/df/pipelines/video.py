"""Video pipeline.

Ingest -> Frame Sample -> Face Extract -> Face Align -> face model -> Aggregate -> Router

Two rules from CLAUDE.md are enforced here and must not be relaxed downstream:

  * 0 faces across the whole video  -> `undetermined`. Never fall through into
    authentic/manipulated. A video we could not find a face in is a video we
    have no opinion about.
  * >1 face in a frame -> score every face, then roll the frame up to WORST-CASE
    severity (highest manipulation score). One manipulated face in a two-face
    frame makes the frame manipulated. This is a default, not a fixed rule --
    confirm before anything downstream assumes it.
"""
from __future__ import annotations

from df.aggregation import AggregationParams, ScoredItem, aggregate
from df.bands import route, worst_case
from df.inference.registry import get_face_model
from df.pipelines.common import PipelineResult
from df.pipelines.extract import FaceExtractor, FrameSampler


def run(
    video_bytes: bytes,
    *,
    sampler: FrameSampler,
    extractor: FaceExtractor,
    detector=None,
    params: AggregationParams | None = None,
) -> PipelineResult:
    detector = detector or get_face_model()
    notes: list[str] = []

    frames = sampler.sample(video_bytes)
    if not frames:
        notes.append("no decodable frames")
        agg = aggregate([], params)
        return PipelineResult([], agg, route(None), detector.version, face_count=0, notes=notes)

    items: list[ScoredItem] = []
    max_faces_in_frame = 0
    total_faces = 0

    for frame in frames:
        crops = extractor.extract(frame.data, frame_index=frame.index)
        max_faces_in_frame = max(max_faces_in_frame, len(crops))
        total_faces += len(crops)
        if not crops:
            # Frame contributes nothing. It is not a zero-score vote -- a frame
            # with no detected face is missing evidence, not evidence of nothing.
            continue

        preds = detector.predict_batch([c.data for c in crops])
        face_scores = [p.score for p in preds]

        if len(crops) > 1:
            notes.append(f"frame {frame.index}: {len(crops)} faces, rolled up worst-case")

        frame_score = worst_case(face_scores)
        if frame_score is None:
            continue

        # The frame's weight is the confidence of the face that set the score,
        # so a worst-case driven by a barely-detected face is weighted down.
        driving = max(range(len(face_scores)), key=lambda i: face_scores[i])
        items.append(
            ScoredItem(
                index=frame.index,
                score=frame_score,
                confidence=crops[driving].confidence,
                kind="frame",
                face_index=crops[driving].face_index,
            )
        )

    if total_faces == 0:
        notes.append(f"0 faces across {len(frames)} sampled frame(s) -> undetermined")
        agg = aggregate([], params)
        return PipelineResult([], agg, route(None), detector.version, face_count=0, notes=notes)

    agg = aggregate(items, params)
    notes.extend(agg.notes)
    return PipelineResult(
        items=items,
        aggregation=agg,
        routing=route(agg.score),
        model_version=detector.version,
        face_count=max_faces_in_frame,
        notes=notes,
    )
