"""CPU preprocessing worker.

Downloads the raw upload, hashes it for the audit trail, and turns it into
scoreable items (aligned face crops, or spectrograms), which it writes under
derived/<job_id>/ for the GPU worker to pick up.

SECURITY: this is the process that parses untrusted, attacker-supplied media.
There is no AV scanning in front of it (Tier 3). Its substitute -- a
network-isolated, locked-down container -- is not optional and ships with this
worker, not later. See docker-compose.yml (`internal` network, read_only,
cap_drop, no-new-privileges) and DECISIONS.md.
"""
from __future__ import annotations

import hashlib
import logging

from df import storage as storage_mod
from df.config import settings
from df.db import Db
from df.jobstatus import JobStatus
from df.pipelines.extract import build_audio_chunker, build_face_extractor, build_frame_sampler
from df.queue import TOPIC_INFERENCE, TOPIC_PREPROCESS, Message, Queue, build_queue
from df.workers.loop import run_worker

log = logging.getLogger("df.worker.cpu")



def _face_size(crop) -> dict:
    """Width/height from the detector's bbox, or NULLs when it reported none.

    A detector that returns no bbox (the deterministic stub) records NULL rather
    than a guess: an invented size would be indistinguishable from a measured
    one in the very column that exists to make the rollup analysable.
    """
    if crop.bbox is None:
        return {"face_w": None, "face_h": None}
    _x, _y, w, h = crop.bbox
    return {"face_w": int(w), "face_h": int(h)}



def _gate_detections(crops: list, discarded: list, *, frame_index: int) -> list:
    """Drop detections the detector itself is not confident in, and RECORD them.

    Gating here -- before the crop is stored and before it reaches the model --
    rather than reweighting downstream. The reason is that a non-face region
    entering the model produces an arbitrary number, and there is no principled
    way to combine an arbitrary number with a real one. Averaging it is not
    better than taking the max of it, only less alarming.

    Measured 2026-08-31 on a public-domain portrait: Haar returned the real face
    at confidence 0.97 scoring 0.54 (authentic), plus two artefacts at 0.32 and
    0.08. The 0.32 artefact scored 55.79 and, through worst-case rollup, set the
    verdict for the whole image to `uncertain`. The real face was never the
    problem; a non-face crop was.

    Worst-case rollup over the survivors is deliberately unchanged. One
    manipulated face is what makes an image manipulated, so averaging across
    faces would trade this visible false positive for invisible false negatives
    on precisely the case the detector exists for.

    This restores, with a record, the gate that was removed on 2026-08-30 for
    leaving no trace. The detections are not silently dropped: they land in the
    preprocess.complete event and are surfaced per-face by the API, so a reader
    sees "discarded, 32% confidence" rather than nothing at all.
    """
    floor = settings.min_detection_confidence
    kept = []
    for crop in crops:
        if crop.confidence < floor:
            discarded.append({
                "frame_index": frame_index,
                "face_index": crop.face_index,
                "confidence": round(float(crop.confidence), 6),
                **_face_size(crop),
            })
            continue
        kept.append(crop)
    return kept


def handle(msg: Message, *, db: Db, storage: storage_mod.Storage, queue: Queue, status: JobStatus) -> None:
    job_id = msg.payload["job_id"]
    media_type = msg.payload["media_type"]

    db.set_status(job_id, "preprocessing")
    status.publish(job_id, "preprocessing")
    db.bump_attempts(job_id)

    raw = storage.get_bytes(storage_mod.raw_key(job_id))

    # Hash the bytes we actually scored, not what the client claimed to send.
    content_hash = hashlib.sha256(raw).hexdigest()
    db.set_content_hash(job_id, content_hash)

    prefix = storage_mod.derived_prefix(job_id)
    manifest: list[dict] = []
    # Detections the confidence floor rejected. Recorded, not dropped.
    discarded: list[dict] = []

    if media_type == "video":
        sampler = build_frame_sampler()
        extractor = build_face_extractor()
        for frame in sampler.sample(raw):
            for crop in _gate_detections(
                extractor.extract(frame.data, frame_index=frame.index),
                discarded, frame_index=frame.index,
            ):
                key = f"{prefix}items/f{frame.index:05d}_x{crop.face_index}.png"
                storage.put_bytes(key, crop.data, "image/png")
                manifest.append({
                    "key": key, "item_index": frame.index, "item_kind": "frame",
                    "face_index": crop.face_index, "confidence": crop.confidence,
                    # Face geometry, carried so the rollup is analysable later.
                    # This was extracted and then discarded here, which is why
                    # no job in this system's history records how big any face
                    # was. See migration 006.
                    **_face_size(crop),
                })

    elif media_type == "image":
        extractor = build_face_extractor()
        for crop in _gate_detections(
            extractor.extract(raw, frame_index=0), discarded, frame_index=0
        ):
            key = f"{prefix}items/i00000_x{crop.face_index}.png"
            storage.put_bytes(key, crop.data, "image/png")
            manifest.append({
                "key": key, "item_index": 0, "item_kind": "image",
                "face_index": crop.face_index, "confidence": crop.confidence,
                **_face_size(crop),
            })

    elif media_type == "audio":
        chunker = build_audio_chunker()
        for chunk in chunker.chunk(raw):
            key = f"{prefix}items/c{chunk.index:05d}.png"
            storage.put_bytes(key, chunk.spectrogram, "image/png")
            manifest.append({
                "key": key, "item_index": chunk.index, "item_kind": "chunk",
                "face_index": None, "confidence": chunk.confidence,
                # No face to measure. NULL, not 0 -- see migration 006.
                "face_w": None, "face_h": None,
            })
    else:
        raise ValueError(f"unknown media_type {media_type!r}")

    db.record_event(job_id, "preprocess.complete", {
        "content_hash": content_hash,
        "items": len(manifest),
        "media_type": media_type,
        # The audit record for what the confidence floor rejected. job_items
        # cannot hold these -- score is NOT NULL and these were never scored, so
        # a row would have to invent one. This event is the permanent trace.
        "detections_discarded": len(discarded),
        "min_detection_confidence": settings.min_detection_confidence,
        "discarded": discarded[:50],
    })

    # An empty manifest is a legitimate outcome (0 faces / no decodable audio).
    # It is forwarded so the router can record `undetermined` -- not dropped and
    # not defaulted into a verdict.
    queue.push(TOPIC_INFERENCE, {
        "job_id": job_id, "media_type": media_type, "manifest": manifest,
    })
    log.info("preprocessed job=%s items=%d", job_id, len(manifest))


def main() -> None:
    db, storage = Db(), storage_mod.build_storage()
    queue, status = build_queue(), JobStatus()
    run_worker(
        TOPIC_PREPROCESS,
        lambda m: handle(m, db=db, storage=storage, queue=queue, status=status),
        queue=queue, db=db, status=status,
    )


if __name__ == "__main__":
    main()
