"""Run one local file through a pipeline end to end -- no Postgres, Redis, or S3.

This is the fastest way to see the ingest -> extract -> score -> aggregate ->
route path behave, including the undetermined and multi-face cases, before any
infrastructure is up.

    python scripts/run_local.py video sample.mp4
    python scripts/run_local.py image sample.jpg --faces 0     # undetermined path
    python scripts/run_local.py image sample.jpg --faces 3     # worst-case rollup
    python scripts/run_local.py audio sample.wav

With DF_INFERENCE_BACKEND=stub (the default) the scores are placeholders and
mean nothing about the file. The pipeline behaviour around them is real.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from df.config import settings  # noqa: E402
from df.pipelines import audio as audio_pipeline  # noqa: E402
from df.pipelines import image as image_pipeline  # noqa: E402
from df.pipelines import video as video_pipeline  # noqa: E402
from df.pipelines.extract import (  # noqa: E402
    StubFaceExtractor,
    build_audio_chunker,
    build_face_extractor,
    build_frame_sampler,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("media_type", choices=["video", "image", "audio"])
    ap.add_argument("path", type=pathlib.Path)
    ap.add_argument(
        "--faces", type=int, default=None,
        help="force N faces per frame (stub extractor only); 0 exercises the undetermined path",
    )
    args = ap.parse_args()

    if not args.path.exists():
        print(f"no such file: {args.path}", file=sys.stderr)
        return 1

    data = args.path.read_bytes()
    extractor = (
        StubFaceExtractor(force_faces=args.faces)
        if args.faces is not None
        else build_face_extractor()
    )

    if args.media_type == "video":
        result = video_pipeline.run(data, sampler=build_frame_sampler(), extractor=extractor)
    elif args.media_type == "image":
        result = image_pipeline.run(data, extractor=extractor)
    else:
        result = audio_pipeline.run(data, chunker=build_audio_chunker())

    out = {
        "media_type": args.media_type,
        "file": str(args.path),
        "result_class": result.routing.result_class.value,
        "band": result.routing.band.value,
        "aggregate_score": result.score,
        "face_count": result.face_count,
        "items_total": result.aggregation.items_total,
        "items_used": result.aggregation.items_used,
        "items_dropped_low_confidence": result.aggregation.items_dropped_low_confidence,
        "items_trimmed": result.aggregation.items_trimmed,
        "aggregation_method": result.aggregation.method,
        "aggregation_params": result.aggregation.params,
        "model_version_id": result.model_version.model_version_id,
        "is_real_detector": result.model_version.is_real_detector,
        "extended_retention": result.routing.extended_retention,
        "flag_for_review": result.routing.flag_for_review,
        "notes": result.notes,
    }
    print(json.dumps(out, indent=2))

    if not result.model_version.is_real_detector:
        print(
            f"\n!! backend={settings.inference_backend}: score is a PLACEHOLDER, "
            f"not a detection result.",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
