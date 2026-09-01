"""Environment-backed settings.

Plain dataclass + os.environ rather than pydantic-settings: keeps the test run
dependency-light and makes it obvious that nothing here reaches out to a network
at import time.
"""
from __future__ import annotations

import os
import pathlib
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


def memory_limit_bytes() -> int | None:
    """Memory this container may use, from the cgroup. None if unlimited.

    Exists for the same reason `cpu_quota` does: the number that matters is the
    cgroup's, and nothing else reports it. Railway's CLI does not expose it
    (`limitOverride` is None, meaning "plan default", and the default is not
    stated), so the only way to know is to read it from inside.

    Worth knowing precisely, because it is the deciding input on video. Video
    fails about 40% of the time on this host and `measured: yes` 2026-09-01 the
    cost is NOT in this repo's code: `cv2.VideoCapture.read()` alone, retaining
    nothing, peaks at 123.4 MB on a 1080p file, against 21.7 MB for twenty
    extractions. Streaming the frames and removing a PNG round trip made it 32%
    faster and barely moved peak RSS. So the question is whether the container
    has ~150 MB of headroom, and that is a number, not an opinion.

    cgroup v2 (`memory.max`) then v1 (`memory.limit_in_bytes`). v1 reports a
    huge sentinel when unlimited, so anything at or above 2^62 is treated as no
    limit rather than reported as a real figure.
    """
    for path in ("/sys/fs/cgroup/memory.max",
                 "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        try:
            raw = pathlib.Path(path).read_text().strip()
        except Exception:
            continue
        if raw == "max":
            return None
        try:
            value = int(raw)
        except ValueError:
            continue
        if value >= 1 << 62:        # v1's "unlimited" sentinel
            return None
        return value
    return None


def cpu_quota() -> float | None:
    """CPU cores this process may actually use, from the cgroup. None if unlimited.

    Exists because `os.cpu_count()` reports the HOST's cores inside a container,
    not the quota. torch then sizes its thread pool to a number of cores it does
    not have, and the threads fight over a fraction of one.

    That is not a small effect. `measured: yes` 2026-09-01 on Railway: model load
    41.8s against 6-16s locally, and a first inference of **225 seconds** against
    0.40s locally -- roughly 500x, on a machine that is not 500x slower. A single
    image took 222s end to end, which for a demo is indistinguishable from broken.

    Reads cgroup v2 first (`cpu.max`, "quota period" or "max period"), then v1
    (`cpu.cfs_quota_us` / `cpu.cfs_period_us`). Any failure returns None and the
    caller falls back to os.cpu_count(), which is the right behaviour on a real
    host where there is no quota to find.
    """
    try:
        raw = pathlib.Path("/sys/fs/cgroup/cpu.max").read_text().split()
        if len(raw) == 2 and raw[0] != "max":
            return int(raw[0]) / int(raw[1])
    except Exception:
        pass
    try:
        q = int(pathlib.Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read_text())
        period = int(pathlib.Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read_text())
        if q > 0 and period > 0:
            return q / period
    except Exception:
        pass
    return None


def usable_threads() -> int:
    """Thread count to give torch/OpenMP. At least 1, never more than the quota."""
    quota = cpu_quota()
    if quota:
        return max(1, int(quota))
    return max(1, os.cpu_count() or 1)


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
    # How many proxies sit between a client and this process, and are therefore
    # trusted to have appended a truthful entry to X-Forwarded-For.
    #
    # 0 (default) means trust nothing: key the limiter on the socket peer. That
    # is correct on a bare host and safe everywhere -- worst case the limiting is
    # coarse, never wrong in the client's favour.
    #
    # Behind a platform proxy the socket peer is the proxy, and `measured: yes`
    # 2026-09-01 on Railway that is not even ONE stable address: the gateway saw
    # 20+ distinct source IPs (100.64.0.2-.22) rotating across requests. So 45
    # rapid requests spread over many near-fresh buckets and rate limiting did
    # nothing at all. Not coarse -- absent.
    #
    # Why a hop COUNT rather than "just read X-Forwarded-For": the header is
    # client-supplied and only its rightmost entries are trustworthy. Each proxy
    # APPENDS the peer it received from, so the rightmost entry was written by
    # the proxy nearest this process, the next one in by the proxy before it, and
    # anything further left may have been invented by the client. Taking the
    # leftmost -- the common shortcut -- lets any caller choose its own bucket by
    # sending a header, which is strictly worse than no limiting because it looks
    # like protection. With N trusted hops the real client is the Nth entry from
    # the right.
    trusted_proxy_hops: int = field(
        default_factory=lambda: max(0, _int("DF_TRUSTED_PROXY_HOPS", 0))
    )

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
    # Detection gate, expressed as a RATIO of the best detection in the same
    # frame. A detection is dropped when its confidence is below
    # `ratio * max(confidence in that frame)`.
    #
    # Relative rather than absolute, because the quantity has no semantics to
    # threshold against. These confidences are OpenCV `levelWeights` from
    # `detectMultiScale3(outputRejectLevels=True)` -- unbounded stage-rejection
    # scores from inside the cascade, squashed into 0-1 by dividing by 10. That
    # is a monotone transform of an internal score, not a probability. So no
    # absolute floor is more justified than any other: 0.3 and 0.5 are equally
    # arbitrary, and "which threshold is correct" is a question the underlying
    # number cannot answer.
    #
    # A ratio sidesteps that. It is invariant to the scale of the confidence,
    # which is the part we do not trust, and it compares detections only against
    # each other within the same image -- which is the comparison Haar's weights
    # can actually support.
    #
    # And it CANNOT empty a non-empty detection set: the best detection always
    # has ratio 1.0. A lone marginal face is therefore kept and reported with its
    # low confidence rather than gated into `undetermined`, so a degraded answer
    # is available instead of no answer. That property is structural, not a
    # fallback branch that might be wrong.
    #
    # 0.4 is still a chosen number, and it is chosen on failure asymmetry rather
    # than evidence: on the one image measured, the artefact/real ratio was
    # 0.316/0.968 = 0.33 (gated), and a lone marginal detection is ratio 1.0
    # (kept). The real repair remains a detector that returns a genuine detection
    # probability -- RetinaFace/SCRFD -- not a better constant here.
    # Where LocalDiskStorage keeps media when DF_S3_ENDPOINT is a file:// URL.
    # Ephemeral on a container platform: a redeploy wipes it while Postgres rows
    # survive, so rows may reference media that is gone. Survivable only because
    # the evidence display reads scores from Postgres and never re-reads the
    # image -- Tier 1 deletes the media on completion anyway.
    local_storage_root: str = field(
        default_factory=lambda: os.environ.get("DF_LOCAL_STORAGE_ROOT", "/data/media")
    )
    # Absolute base URL this service is reachable at, used to build the upload
    # grant. On a platform deploy the container cannot know its own public
    # hostname, and a relative URL would break a browser on a different origin.
    public_base_url: str = field(
        default_factory=lambda: os.environ.get("DF_PUBLIC_BASE_URL", "").rstrip("/")
    )
    # Browser origins allowed to call this API. Comma-separated. Empty means
    # same-origin only, which is the safe default and also what breaks a Vercel
    # frontend, so it must be set explicitly on deploy.
    cors_origins: str = field(
        default_factory=lambda: os.environ.get("DF_CORS_ORIGINS", "")
    )

    detection_confidence_ratio: float = field(
        default_factory=lambda: _float("DF_DETECTION_CONFIDENCE_RATIO", 0.4)
    )
    # Longest side, in pixels, of the image Haar actually runs on. Crops are
    # still taken from the FULL-RESOLUTION original -- this bounds detection
    # only, never the crop.
    #
    # `measured: yes` 2026-09-01 on the deployed service: a 12.2 MP upload
    # (4032x3024, 1.7 MB on disk, 36.6 MB decoded) sat in `preprocessing` for
    # 85+ seconds and then took the container down with SIGKILL, and the job
    # came back `undetermined`. That is the ordinary case, not an edge one: it
    # is what a current phone camera produces, and "upload a photo from your
    # phone" is the demo.
    #
    # Detection accuracy is the other half, and it argues the same way.
    # `minSize=(48, 48)` is only meaningful relative to the image: at 1.2 MP it
    # found one 523x523 face, at 12.2 MP it found three boxes of which the first
    # was 52x52 -- noise, because a real face there is ~1500 px and a 48 px
    # window is looking at skin texture. Detecting at a bounded size makes the
    # minimum mean the same thing whatever the camera did.
    #
    # Env-tunable so this can be adjusted on a running deployment without a
    # rebuild. 0 or negative disables the cap and restores full-resolution
    # detection.
    detect_max_side: int = field(
        default_factory=lambda: _int("DF_DETECT_MAX_SIDE", 1600)
    )
    audio_chunk_seconds: float = field(
        default_factory=lambda: _float("DF_AUDIO_CHUNK_SECONDS", 3.0)
    )


settings = Settings()
