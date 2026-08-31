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
from df.pipelines.extract import (
    build_audio_chunker,
    build_face_extractor,
    build_frame_sampler,
    decode_image,
    sniff_format,
)
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
    """Drop detections far weaker than the best one in the same frame, and
    RECORD what was dropped.

    Gating here -- before the crop is stored and before it reaches the model --
    rather than reweighting downstream. A non-face region entering the model
    produces an arbitrary number, and there is no principled way to combine an
    arbitrary number with a real one: averaging it is not better than taking the
    max of it, only less alarming.

    Worst-case rollup over the survivors is deliberately untouched. One
    manipulated face is what makes an image manipulated, so a confidence-weighted
    mean across faces would drag a swapped face's score toward the crowd in a
    group photo -- trading a visible false positive for invisible false negatives
    on precisely the case this detector exists for.

    RELATIVE, not an absolute floor. See `Settings.detection_confidence_ratio`:
    the confidence is a squashed cascade reject level with no probabilistic
    meaning, so comparing detections against each other is the only comparison it
    supports. Two consequences worth stating:

      * it is invariant to the scale of the untrusted quantity;
      * it cannot empty a non-empty set, because the best detection is always
        ratio 1.0. A lone marginal face is kept and reported with its low
        confidence rather than gated into `undetermined` -- a degraded answer
        instead of no answer, guaranteed structurally rather than by a fallback
        branch.

    Measured 2026-08-31, the entire evidence base: a public-domain portrait where
    Haar returned the real face at 0.968 (which B7 scored 0.54, authentic) plus
    artefacts at 0.316 and 0.075. The 0.316 artefact scored 55.79 and, through
    worst-case rollup, set the verdict for the whole image to `uncertain`. Ratio
    0.316/0.968 = 0.33, so it gates; the real face survives at 1.0.

    This restores, with a record, the gate removed on 2026-08-30 for leaving no
    trace. Dropped detections land in the preprocess.complete event and are
    surfaced per-face by the API.
    """
    if not crops:
        return []

    best = max(c.confidence for c in crops)
    if best <= 0:
        # Nothing to compare against. Keeping all is the honest degradation:
        # a ratio is undefined here, and inventing an ordering would be worse.
        return list(crops)

    ratio = settings.detection_confidence_ratio
    kept = []
    for crop in crops:
        rel = crop.confidence / best
        if rel < ratio:
            discarded.append({
                "frame_index": frame_index,
                "face_index": crop.face_index,
                "confidence": round(float(crop.confidence), 6),
                "relative_to_best": round(float(rel), 6),
                "best_in_frame": round(float(best), 6),
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

    media_format = sniff_format(raw)
    sampled_any = True   # images decide decodability by the decoder, not sampling
    prefix = storage_mod.derived_prefix(job_id)
    manifest: list[dict] = []
    # Detections the confidence floor rejected. Recorded, not dropped.
    discarded: list[dict] = []

    if media_type == "video":
        sampler = build_frame_sampler()
        extractor = build_face_extractor()
        frames = sampler.sample(raw)
        sampled_any = bool(frames)
        for frame in frames:
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
        chunks = chunker.chunk(raw)
        sampled_any = bool(chunks)
        for chunk in chunks:
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

    # Did anything read the bytes at all? Distinct from "read them and found no
    # face", which is a legitimate `undetermined`. For images the decoder is the
    # authority; for video and audio, producing zero frames/chunks from a
    # non-empty upload is the same signal.
    if media_type == "image":
        decodable = decode_image(raw) is not None
    else:
        decodable = bool(sampled_any)
    if not decodable:
        log.warning(
            "job=%s could not decode %s (format=%s); reporting undetermined with a reason",
            job_id, media_type, media_format,
        )

    db.record_event(job_id, "preprocess.complete", {
        "content_hash": content_hash,
        "items": len(manifest),
        "media_type": media_type,
        # The audit record for what the confidence floor rejected. job_items
        # cannot hold these -- score is NOT NULL and these were never scored, so
        # a row would have to invent one. This event is the permanent trace.
        # What the bytes actually were, and whether anything could read them.
        # Without this, an undecodable upload produces zero items and the job
        # reports `undetermined` -- indistinguishable from a photo with nobody
        # in it. A judge uploading a HEIC off an iPhone would be told no face
        # was found in a picture of their own face.
        "media_format": media_format,
        "decodable": decodable,
        "detections_discarded": len(discarded),
        "detection_confidence_ratio": settings.detection_confidence_ratio,
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
