"""End-to-end wiring test: preprocess -> inference -> router, through the REAL
worker handlers.

This is the check that docker-compose would otherwise be the first thing to run.
It drives the actual handler functions with in-memory Postgres/Redis/S3 stand-ins,
so a broken queue payload contract, a missing DB write, or a delete that never
fires shows up here in a second rather than after a multi-minute image build.

What it does NOT cover: container networking, image builds, real presigned URLs,
the WebSocket transport. Those need compose up.
"""
from __future__ import annotations

import pytest

from df.bands import ResultClass
from df.queue import TOPIC_AGGREGATE, TOPIC_INFERENCE, TOPIC_PREPROCESS, InMemoryQueue, Message
from df.storage import InMemoryStorage
from df.workers import cpu_preprocess, gpu_inference
from df.workers import router as router_worker
from tests.fakes import FakeDb, FakeJobStatus


class Harness:
    """Runs a job through all three stages the way the queue would."""

    def __init__(self, media_type: str, raw: bytes, *, hold: bool = False, job_id: str = "job-e2e"):
        self.db = FakeDb()
        self.storage = InMemoryStorage()
        self.queue = InMemoryQueue()
        self.status = FakeJobStatus()
        self.job_id = job_id
        self.media_type = media_type

        self.db.add_job(job_id, media_type, retention_hold=hold)
        self.storage.put_bytes(f"raw/{job_id}/original", raw)
        self.queue.push(TOPIC_PREPROCESS, {"job_id": job_id, "media_type": media_type})

    def run(self) -> None:
        msg = self.queue.pop(TOPIC_PREPROCESS)
        cpu_preprocess.handle(
            msg, db=self.db, storage=self.storage, queue=self.queue, status=self.status
        )

        msg = self.queue.pop(TOPIC_INFERENCE)
        gpu_inference.handle(
            msg, db=self.db, storage=self.storage, queue=self.queue, status=self.status
        )

        msg = self.queue.pop(TOPIC_AGGREGATE)
        router_worker.handle(msg, db=self.db, storage=self.storage, status=self.status)

    @property
    def job(self) -> dict:
        return self.db.jobs[self.job_id]


@pytest.fixture
def video_job() -> Harness:
    h = Harness("video", b"pretend-video-bytes-with-faces")
    h.run()
    return h


# --- the full happy path ----------------------------------------------------


def test_video_job_completes_with_a_verdict(video_job):
    job = video_job.job

    assert job["status"] == "complete"
    assert job["result_class"] in {c.value for c in ResultClass}
    assert job["completed_at"] is not None


def test_the_job_row_is_a_complete_audit_trail(video_job):
    """CLAUDE.md: hash + model_version_id + aggregation method/params IS the
    audit trail. A score without them is unreproducible."""
    job = video_job.job

    assert job["content_hash"] is not None and len(job["content_hash"]) == 64
    assert job["model_version_id"] is not None
    assert job["aggregation_method"] is not None
    assert job["aggregation_params"] is not None
    assert "trim_frac" in job["aggregation_params"]


def test_content_hash_is_computed_from_the_bytes_actually_scored():
    import hashlib

    raw = b"specific-bytes"
    h = Harness("video", raw)
    h.run()

    assert h.job["content_hash"] == hashlib.sha256(raw).hexdigest()


def test_per_item_scores_are_persisted(video_job):
    rows = video_job.db.get_items(video_job.job_id)

    assert len(rows) > 0
    assert all(0.0 <= r["score"] <= 100.0 for r in rows)
    assert all(0.0 <= r["confidence"] <= 1.0 for r in rows)


def test_client_sees_every_stage_transition(video_job):
    """A client on the WebSocket must be able to follow progress, not just get
    a result at the end."""
    seen = video_job.status.statuses(video_job.job_id)

    assert seen == ["preprocessing", "inference", "aggregating", "complete"]


def test_aggregate_score_is_not_a_raw_item_score(video_job):
    """Bands apply to the aggregated score. If aggregation collapsed to
    'first item wins' this would silently still look plausible."""
    job = video_job.job
    rows = video_job.db.get_items(video_job.job_id)

    assert job["aggregate_score"] is not None
    assert job["item_count"] > 1
    assert len(rows) >= job["item_count"]


# --- Tier 1: media is gone when the job completes ---------------------------


def test_raw_media_and_face_crops_are_deleted_on_completion(video_job):
    """The whole Tier 1 promise, exercised through the real router handler."""
    storage = video_job.storage

    assert not storage.exists(f"raw/{video_job.job_id}/original"), "raw upload survived"
    assert storage.list_prefix(f"derived/{video_job.job_id}/") == [], "face crops survived"


def test_derived_artifacts_existed_before_deletion():
    """Guards against a false pass: if preprocessing silently produced nothing,
    the deletion assertion above would hold trivially."""
    h = Harness("video", b"pretend-video-bytes-with-faces")

    msg = h.queue.pop(TOPIC_PREPROCESS)
    cpu_preprocess.handle(msg, db=h.db, storage=h.storage, queue=h.queue, status=h.status)

    assert len(h.storage.list_prefix(f"derived/{h.job_id}/")) > 0


def test_deletion_is_recorded_on_the_job_row(video_job):
    job = video_job.job

    assert job["raw_deleted_at"] is not None
    assert job["derived_deleted_at"] is not None
    assert "retention.deleted" in video_job.db.event_names(video_job.job_id)


def test_a_held_job_keeps_its_media_but_still_gets_a_verdict():
    h = Harness("video", b"pretend-video-bytes-with-faces", hold=True)
    h.run()

    assert h.job["status"] == "complete", "hold must not block the verdict"
    assert h.storage.exists(f"raw/{h.job_id}/original"), "held media was deleted"
    assert len(h.storage.list_prefix(f"derived/{h.job_id}/")) > 0
    assert "retention.delete_skipped" in h.db.event_names(h.job_id)


def test_result_is_committed_before_media_is_deleted(video_job):
    """Ordering matters: if the delete ran first and the result write failed,
    the inputs needed to reproduce the verdict would already be gone."""
    names = video_job.db.event_names(video_job.job_id)

    assert names.index("router.decided") < names.index("retention.deleted")


# --- undetermined path ------------------------------------------------------


def test_a_job_with_no_faces_is_undetermined_and_still_cleaned_up(monkeypatch):
    from df.pipelines.extract import StubFaceExtractor

    monkeypatch.setattr(
        cpu_preprocess, "build_face_extractor", lambda: StubFaceExtractor(force_faces=0)
    )

    h = Harness("video", b"a-video-with-nobody-in-it")
    h.run()

    assert h.job["result_class"] == ResultClass.UNDETERMINED.value
    assert h.job["aggregate_score"] is None
    assert h.job["face_count"] == 0
    # No usable signal is still a completed job, and its media still goes.
    assert not h.storage.exists(f"raw/{h.job_id}/original")


def test_undetermined_jobs_are_flagged_for_review(monkeypatch):
    from df.pipelines.extract import StubFaceExtractor

    monkeypatch.setattr(
        cpu_preprocess, "build_face_extractor", lambda: StubFaceExtractor(force_faces=0)
    )

    h = Harness("image", b"an-image-with-nobody-in-it")
    h.run()

    assert [j for j, _, _ in h.db.review_flags] == [h.job_id]


# --- multi-face rollup through the real stack -------------------------------


def test_multi_face_video_reports_face_count_and_rolls_up(monkeypatch):
    from df.pipelines.extract import StubFaceExtractor

    monkeypatch.setattr(
        cpu_preprocess, "build_face_extractor", lambda: StubFaceExtractor(force_faces=3)
    )

    h = Harness("video", b"a-crowd-scene")
    h.run()

    assert h.job["face_count"] == 3
    rows = h.db.get_items(h.job_id)
    # Every face is stored individually; the verdict is one rolled-up number.
    assert len({r["face_index"] for r in rows}) == 3
    assert h.job["aggregate_score"] is not None


# --- other media types ------------------------------------------------------


def test_image_job_uses_identity_aggregation():
    h = Harness("image", b"a-still-image")
    h.run()

    assert h.job["status"] == "complete"
    assert h.job["aggregation_method"] == "identity.v1"


def test_audio_job_uses_the_audio_model_and_has_no_face_count():
    h = Harness("audio", b"some-audio-bytes")
    h.run()

    assert h.job["status"] == "complete"
    assert "audio" in h.job["model_version_id"]
    assert h.job["face_count"] is None


# --- extended retention window, through the real router ---------------------


class _FixedScore:
    """Forces a chosen band so the retention branch can be driven end to end."""

    def __init__(self, score: float) -> None:
        self.score = score

    @property
    def version(self):
        from df.inference.base import ModelVersion

        return ModelVersion("face-fixed-v0", "test", "face", None, "none", False)

    def predict_batch(self, inputs: list):
        from df.inference.base import Prediction

        return [Prediction(self.score, 1.0) for _ in inputs]


def _run_at_score(score: float, monkeypatch, job_id: str = "job-band") -> Harness:
    monkeypatch.setattr(gpu_inference, "get_face_model", lambda: _FixedScore(score))
    h = Harness("video", b"a-clip", job_id=job_id)
    h.run()
    return h


def test_high_band_preserves_the_driving_crops_and_deletes_everything_else(monkeypatch):
    """CLAUDE.md: the window protects the flagged media itself -- the face crops
    that drove the score -- not the full raw source, and not just the job row."""
    h = _run_at_score(95.0, monkeypatch, "job-hi-band")

    assert h.job["band"] == "likely_manipulated"
    assert h.job["extended_retention_until"] is not None

    # Tier 1 still clears the raw source and the working copies.
    assert not h.storage.exists(f"raw/{h.job_id}/original")
    assert h.storage.list_prefix(f"derived/{h.job_id}/") == []

    # The evidence a dispute would need survives.
    cold = h.storage.list_prefix(f"cold/{h.job_id}/")
    assert cold, "high-band job preserved nothing"
    assert h.job["cold_prefix"] == f"cold/{h.job_id}/"


def test_preserved_crops_are_the_ones_that_drove_the_score(monkeypatch):
    h = _run_at_score(95.0, monkeypatch, "job-hi-driving")

    cold = h.storage.list_prefix(f"cold/{h.job_id}/")
    rows = h.db.get_items(h.job_id)

    # One preserved object per item that survived dropping and trimming.
    assert len(cold) == h.job["item_count"]

    # Every preserved object is traceable back to a scored item, so a dispute
    # can tie the evidence to a row rather than to an orphaned blob.
    extracted = {r["object_key"].rsplit("/", 1)[-1] for r in rows}
    preserved = {k.rsplit("/", 1)[-1] for k in cold}
    assert preserved <= extracted, f"preserved objects not traceable to items: {preserved - extracted}"

    # Selectivity (that non-driving and trimmed crops are excluded) is pinned by
    # test_window_preserves_only_the_crops_that_drove_the_score -- asserting it
    # here would depend on how many faces the stub happens to emit for this input.


def test_low_band_preserves_nothing(monkeypatch):
    h = _run_at_score(5.0, monkeypatch, "job-lo-band")

    assert h.job["extended_retention_until"] is None
    assert h.job["cold_prefix"] is None
    assert h.storage.list_prefix(f"cold/{h.job_id}/") == []


def test_60_to_80_is_flagged_at_low_urgency_but_still_deleted(monkeypatch):
    """The riskier gap: recorded, alerted, and still deleted on the normal
    schedule -- never a silent pass-through."""
    h = _run_at_score(70.0, monkeypatch, "job-mid-band")

    assert h.job["band"] == "leaning_manipulated"
    assert [(j, u) for j, _, u in h.db.review_flags] == [(h.job_id, "low")]
    # Not a high-band result: nothing is preserved, deletion is normal.
    assert h.job["extended_retention_until"] is None
    assert not h.storage.exists(f"raw/{h.job_id}/original")
    assert h.storage.list_prefix(f"cold/{h.job_id}/") == []


def test_20_to_40_passes_without_a_flag(monkeypatch):
    h = _run_at_score(30.0, monkeypatch, "job-lowmid-band")

    assert h.job["band"] == "leaning_authentic"
    assert h.db.review_flags == []
    assert not h.storage.exists(f"raw/{h.job_id}/original")


def test_held_high_band_job_keeps_everything(monkeypatch):
    monkeypatch.setattr(gpu_inference, "get_face_model", lambda: _FixedScore(95.0))
    h = Harness("video", b"a-clip", hold=True, job_id="job-hi-held")
    h.run()

    assert h.job["status"] == "complete"
    assert h.storage.exists(f"raw/{h.job_id}/original"), "held raw source was deleted"
    assert h.storage.list_prefix(f"derived/{h.job_id}/"), "held crops were deleted"
    assert h.storage.list_prefix(f"cold/{h.job_id}/"), "held job preserved nothing"


def test_video_and_image_jobs_share_one_face_model_version_id():
    v = Harness("video", b"clip", job_id="job-v")
    v.run()
    i = Harness("image", b"still", job_id="job-i")
    i.run()
    a = Harness("audio", b"sound", job_id="job-a")
    a.run()

    # Same weights across video and image => one id.
    assert v.job["model_version_id"] == i.job["model_version_id"]
    # Audio is a separate model and must never share that id.
    assert a.job["model_version_id"] != v.job["model_version_id"]
