"""Environment-backed settings.

Plain dataclass + os.environ rather than pydantic-settings: keeps the test run
dependency-light and makes it obvious that nothing here reaches out to a network
at import time.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field


def _int(key: str, default: int) -> int:
    return int(os.environ.get(key, default))


def _float(key: str, default: float) -> float:
    return float(os.environ.get(key, default))


def _bool(key: str, default: bool) -> bool:
    raw = os.environ.get(key)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    pg_dsn: str = field(
        default_factory=lambda: os.environ.get(
            "DF_PG_DSN", "postgresql://deepfake:deepfake@localhost:5432/deepfake"
        )
    )
    redis_url: str = field(
        default_factory=lambda: os.environ.get("DF_REDIS_URL", "redis://localhost:6379/0")
    )

    # object storage
    s3_endpoint: str = field(
        default_factory=lambda: os.environ.get("DF_S3_ENDPOINT", "http://localhost:9000")
    )
    # What the browser should talk to. Differs from s3_endpoint inside compose,
    # where the gateway resolves `minio` but the user's browser does not.
    s3_public_endpoint: str = field(
        default_factory=lambda: os.environ.get(
            "DF_S3_PUBLIC_ENDPOINT", os.environ.get("DF_S3_ENDPOINT", "http://localhost:9000")
        )
    )
    s3_bucket: str = field(
        default_factory=lambda: os.environ.get("DF_S3_BUCKET", "deepfake-ingest")
    )
    s3_region: str = field(
        default_factory=lambda: os.environ.get("DF_S3_REGION", "us-east-1")
    )
    s3_force_path_style: bool = field(
        default_factory=lambda: _bool("DF_S3_FORCE_PATH_STYLE", True)
    )
    presign_ttl_seconds: int = field(
        default_factory=lambda: _int("DF_PRESIGN_TTL_SECONDS", 900)
    )
    max_upload_bytes: int = field(
        default_factory=lambda: _int("DF_MAX_UPLOAD_BYTES", 2 * 1024**3)
    )

    # ingress rate limiting (Tier 1)
    ratelimit_capacity: int = field(
        default_factory=lambda: _int("DF_RATELIMIT_CAPACITY", 30)
    )
    ratelimit_refill_per_sec: float = field(
        default_factory=lambda: _float("DF_RATELIMIT_REFILL_PER_SEC", 0.5)
    )

    # inference
    inference_backend: str = field(
        default_factory=lambda: os.environ.get("DF_INFERENCE_BACKEND", "stub")
    )
    face_weights: str = field(default_factory=lambda: os.environ.get("DF_FACE_WEIGHTS", ""))
    audio_weights: str = field(default_factory=lambda: os.environ.get("DF_AUDIO_WEIGHTS", ""))

    # retention
    raw_ttl_seconds: int = field(default_factory=lambda: _int("DF_RAW_TTL_SECONDS", 0))
    extended_retention_days: int = field(
        default_factory=lambda: _int("DF_EXTENDED_RETENTION_DAYS", 30)
    )
    # Age at which a job still in awaiting_upload is treated as abandoned and
    # its media cleared. Must comfortably exceed the presign TTL above.
    abandoned_upload_hours: int = field(
        default_factory=lambda: _int("DF_ABANDONED_UPLOAD_HOURS", 24)
    )
    # Age at which a job still in an in-flight state is treated as stalled --
    # its queue message is gone and nothing will advance it. Must exceed the
    # slowest realistic job by a wide margin: failing a merely-slow job is
    # recoverable by resubmitting, but never sweeping retains media forever.
    stalled_job_hours: int = field(
        default_factory=lambda: _int("DF_STALLED_JOB_HOURS", 6)
    )

    # queue / DLQ
    max_attempts: int = field(default_factory=lambda: _int("DF_MAX_ATTEMPTS", 3))
    # "streams" (consumer groups) or "lists" (the original BRPOPLPUSH handoff,
    # kept as a rollback path).
    queue_backend: str = field(
        default_factory=lambda: os.environ.get("DF_QUEUE_BACKEND", "streams")
    )
    # How long a taken-but-unacked message must sit idle before another
    # consumer may claim it. Too low and a slow-but-healthy worker has its work
    # stolen and done twice; too high and a dead worker's jobs stall. Must
    # exceed the slowest expected handler by a comfortable margin.
    queue_reclaim_ms: int = field(
        default_factory=lambda: _int("DF_QUEUE_RECLAIM_MS", 120_000)
    )

    # sampling
    video_fps_sample: float = field(default_factory=lambda: _float("DF_VIDEO_FPS_SAMPLE", 2.0))
    video_max_frames: int = field(default_factory=lambda: _int("DF_VIDEO_MAX_FRAMES", 300))
    # Detection-confidence floor. Boxes below this never reach the model.
    #
    # UNTUNED, and chosen under deadline pressure from n=1 example: a public-
    # domain portrait where Haar returned one real face at 0.97 and two
    # artefacts at 0.32 and 0.08, and the 0.32 artefact set the whole verdict.
    # 0.3 sits between those observations and nothing more principled than that.
    #
    # It gates rather than reweights on purpose. A non-face region entering the
    # model produces an arbitrary number, and averaging an arbitrary number with
    # a real one is not more principled than taking the max of them -- just less
    # alarming. Worst-case rollup over survivors is preserved, because one
    # manipulated face is what makes an image manipulated, and any averaging
    # across faces trades a visible false positive for invisible false negatives
    # on exactly the case this tool exists for.
    #
    # Discarded detections are RECORDED (preprocess.complete event, and surfaced
    # per-face in the API), never silently dropped.
    min_detection_confidence: float = field(
        default_factory=lambda: _float("DF_MIN_DETECTION_CONFIDENCE", 0.3)
    )
    audio_chunk_seconds: float = field(
        default_factory=lambda: _float("DF_AUDIO_CHUNK_SECONDS", 3.0)
    )


settings = Settings()
