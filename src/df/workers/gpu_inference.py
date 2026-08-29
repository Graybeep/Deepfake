"""GPU inference worker.

Scores the items the CPU worker produced and writes one job_items row per item.
This is the cost driver -- it is the pod HPA scales on in the target K8s state.

Model selection follows CLAUDE.md: video and image both use the shared face
model (one model_version_id); audio uses its own model.
"""
from __future__ import annotations

import logging

from df import storage as storage_mod
from df.db import Db
from df.inference.registry import get_audio_model, get_face_model
from df.jobstatus import JobStatus
from df.queue import TOPIC_AGGREGATE, TOPIC_INFERENCE, Message, Queue, build_queue
from df.workers.loop import run_worker

log = logging.getLogger("df.worker.gpu")

BATCH_SIZE = 32


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


def main() -> None:
    db, storage = Db(), storage_mod.build_storage()
    queue, status = build_queue(), JobStatus()
    run_worker(
        TOPIC_INFERENCE,
        lambda m: handle(m, db=db, storage=storage, queue=queue, status=status),
        queue=queue, db=db, status=status,
    )


if __name__ == "__main__":
    main()
