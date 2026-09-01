"""GPU inference worker.

Scores the items the CPU worker produced and writes one job_items row per item.
This is the cost driver -- it is the pod HPA scales on in the target K8s state.

Model selection follows CLAUDE.md: video and image both use the shared face
model (one model_version_id); audio uses its own model.
"""
from __future__ import annotations

import logging
import os
import time

from df import storage as storage_mod
from df.config import settings
from df.db import Db
from df.inference.registry import get_audio_model, get_face_model
from df.jobstatus import JobStatus
from df.queue import TOPIC_AGGREGATE, TOPIC_INFERENCE, Message, Queue, build_queue
from df.workers.loop import configure_logging, run_worker

log = logging.getLogger("df.worker.gpu")


def _int(name: str, default: int) -> int:
    try:
        return max(1, int(os.environ.get(name, default)))
    except ValueError:
        return default

# How many crops go through the model at once.
#
# This is a MEMORY bound, not a throughput knob, and 32 was a throughput number.
# Peak RSS scales with batch size, so a large batch makes memory scale with the
# number of faces in the photo -- which is attacker-and-judge-controlled input.
#
# `measured: yes` 2026-09-01, B7 at 380px on CPU:
#     loaded 836MB · 2 crops 867MB · 4 crops 1101MB · 6 crops 1359MB · 8 crops 1626MB
#
# A six-face group photo went through as one batch of six and OOM-killed a 2GB
# container on Railway: the supervisor took the container down, the platform
# restarted it, and the in-flight job stranded at status=inference. The user saw
# a timeout. Nothing in the logs said "out of memory" -- just a clean boot,
# because SIGKILL leaves no traceback.
#
# At 2 the peak is ~870MB no matter how many faces are in the frame: a 20-face
# photo runs 10 sequential batches instead of one enormous one. On 2 cores the
# throughput cost is near zero, because a larger batch was not buying
# parallelism there anyway.
BATCH_SIZE = _int("DF_INFERENCE_BATCH_SIZE", 1)


def handle(msg: Message, *, db: Db, storage: storage_mod.Storage, queue: Queue, status: JobStatus) -> None:
    job_id = msg.payload["job_id"]
    media_type = msg.payload["media_type"]
    manifest = msg.payload["manifest"]

    db.set_status(job_id, "inference")
    status.publish(job_id, "inference", items=len(manifest))

    detector = get_audio_model() if media_type == "audio" else get_face_model()

    rows: list[dict] = []
    for start in range(0, len(manifest), BATCH_SIZE):
        batch = manifest[start : start + BATCH_SIZE]
        blobs = [storage.get_bytes(item["key"]) for item in batch]
        preds = detector.predict_batch(blobs)
        for item, pred in zip(batch, preds):
            rows.append({
                "item_index": item["item_index"],
                "item_kind": item["item_kind"],
                "face_index": item["face_index"],
                "score": pred.score,
                # Detection/alignment confidence from preprocessing is what
                # weights aggregation -- the model's own confidence says nothing
                # about whether we got a clean face to look at.
                "confidence": item["confidence"],
                # Recorded so the router can preserve exactly the objects that
                # drove a flagged score, rather than rebuilding key names by
                # convention and silently missing when the layout changes.
                "object_key": item["key"],
                # Face geometry from preprocessing. Carried through unchanged so
                # a future per-face bar can be fitted against size buckets
                # instead of a single global constant -- the feature has to be
                # recorded now for that to be possible later at all.
                "face_w": item.get("face_w"),
                "face_h": item.get("face_h"),
                # Which weights produced THIS number. The job row's
                # model_version_id is derived from the rows that survive, not
                # from this worker's message -- during a rolling deploy the two
                # can disagree, and the surviving rows are the ones that were
                # actually used to compute the score.
                "model_version_id": detector.version.model_version_id,
                # Trust level of the weights that produced THIS number,
                # carried with it so the caveat cannot drift from the score.
                "model_validation": detector.version.validation,
                # How the raw logit became this number. Carried per row
                # for the same reason the model id is: refitting the
                # temperature changes the score without changing the
                # weights hash, so the id alone cannot distinguish them.
                "calibration": detector.version.calibration,
            })

    db.insert_items(job_id, rows)
    db.record_event(job_id, "inference.complete", {
        "scored": len(rows),
        "model_version_id": detector.version.model_version_id,
        "is_real_detector": detector.version.is_real_detector,
    })

    queue.push(TOPIC_AGGREGATE, {
        "job_id": job_id,
        "media_type": media_type,
        "model_version_id": detector.version.model_version_id,
    })
    log.info("scored job=%s items=%d model=%s", job_id, len(rows), detector.version.model_version_id)


def _warm_models() -> None:
    """Load the weights and run one throwaway inference BEFORE consuming work.

    The model is behind an lru_cache and was previously built on the first
    message, so the first real job paid the whole cost: `measured: yes`
    2026-08-31, 12.0s to load the 254MB B7 and initialise timm, on top of
    0.32s/face steady-state.

    Running always-on does not help if the load is lazy -- the first request
    after a deploy still pays it, which on a demo is the one request that
    matters. Warming here moves that cost into container startup, where nobody
    is waiting on it, and the dummy forward pass matters as much as the load:
    the first inference allocates workspace and specialises kernels, so loading
    without running leaves some of the stall in place.

    Failing here is deliberate. A worker that cannot load its weights should die
    at boot with a clear error, not accept a job and dead-letter it.
    """
    if settings.inference_backend != "torch":
        log.info("stub backend: nothing to warm")
        return

    import numpy as np

    t0 = time.perf_counter()
    detector = get_face_model()
    loaded = time.perf_counter() - t0

    # A 1x1 PNG is enough to force the first forward pass; the detector resizes
    # whatever it is given.
    import cv2

    blank = cv2.imencode(".png", np.zeros((8, 8, 3), np.uint8))[1].tobytes()
    t0 = time.perf_counter()
    detector.predict_batch([blank])
    warmed = time.perf_counter() - t0

    log.info(
        "warm: %s loaded in %.2fs, first inference %.2fs, ready",
        detector.version.model_version_id, loaded, warmed,
    )


def main() -> None:
    db, storage = Db(), storage_mod.build_storage()
    queue, status = build_queue(), JobStatus()
    # Before warming, so the warm-up is visible in the logs at all.
    configure_logging()
    _warm_models()
    run_worker(
        TOPIC_INFERENCE,
        lambda m: handle(m, db=db, storage=storage, queue=queue, status=status),
        queue=queue, db=db, status=status,
    )


if __name__ == "__main__":
    main()
