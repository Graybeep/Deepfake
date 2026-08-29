"""Aggregation + routing worker.

Reads the per-item scores, rolls multi-face items up to worst case, aggregates,
routes on the score band, writes the verdict, and then triggers the Tier 1 TTL
delete of raw media and face crops.

Ordering is deliberate: the result is committed BEFORE the media is deleted. If
the delete fails, the sweeper retries it; if the result write failed we would
have deleted the only inputs that could reproduce it.
"""
from __future__ import annotations

import logging

from df import storage as storage_mod
from df.aggregation import AggregationParams, aggregate, aggregate_identity
from df.bands import ResultClass, ReviewUrgency, route
from df.db import Db
from df.jobstatus import JobStatus
from df.notify import notify_review_flag
from df.queue import TOPIC_AGGREGATE, Message, Queue, build_queue
from df.retention import delete_media_for_job, open_extended_retention_window
from df.rollup import rollup_items
from df.workers.loop import run_worker

log = logging.getLogger("df.worker.router")


def _driving_keys(used_items, rows: list[dict]) -> list[str]:
    """Storage keys of the crops/chunks that actually produced the score.

    `used_items` are the rolled-up items that survived confidence dropping and
    trimming; each carries the face_index that SET its worst-case score. Those
    are the objects a dispute over a >80 verdict would need to look at -- not
    every crop ever extracted, and not the raw source.
    """
    by_key = {(r["item_index"], r.get("face_index")): r.get("object_key") for r in rows}
    keys: list[str] = []
    for item in used_items:
        key = by_key.get((item.index, item.face_index))
        if key:
            keys.append(key)
    return keys


def handle(msg: Message, *, db: Db, storage: storage_mod.Storage, status: JobStatus) -> None:
    job_id = msg.payload["job_id"]
    media_type = msg.payload["media_type"]

    db.set_status(job_id, "aggregating")
    status.publish(job_id, "aggregating")

    rows = db.get_items(job_id)
    items, max_faces = rollup_items(rows)

    # Attribute the score to the model that produced the rows it was computed
    # from -- not to whichever message happened to arrive. Duplicate delivery
    # puts two aggregate messages in flight, write_result is an UPDATE, and the
    # payload's model version can belong to a consumer whose rows lost the
    # ON CONFLICT race. The rows are the evidence; the message is hearsay.
    observed = db.item_model_versions(job_id)
    if len(observed) > 1:
        # Two models each won some items. The score is a blend of both, so no
        # single model_version_id makes it reproducible, and CLAUDE.md does not
        # permit storing a result whose provenance is a guess. Fail loudly:
        # this dead-letters with the reason attached rather than emitting an
        # unauditable verdict.
        raise ValueError(
            f"job {job_id} has items scored by multiple model versions {observed}; "
            "refusing to attribute one score to one of them"
        )
    model_version_id = observed[0] if observed else msg.payload["model_version_id"]

    # Rows written before migration 003 carry no producer. item_model_versions
    # drops NULLs, so a job straddling that deploy -- some rows recorded, some
    # not -- reads as a single model and is attributed to it rather than
    # refused. That is the right default (refusing legitimate jobs mid-migration
    # would be worse than the ambiguity), but it means partial provenance is
    # being treated as full provenance, so say so in the audit trail instead of
    # letting the row imply more certainty than was observed.
    # Trust level, derived from the same rows as the model id. Refusing a job
    # scored by two models already guarantees one model, so more than one
    # validation level here means the rows disagree about weights we think are
    # identical -- unreproducible, and not something to average over.
    validations = db.item_model_validations(job_id)
    if len(validations) > 1:
        raise ValueError(
            f"job {job_id} has items recorded with multiple validation levels "
            f"{validations}; refusing to state one trust level for the result"
        )
    # None when nothing recorded a level. Downstream must read that as
    # untrusted, never as "fine" -- see _public_job.
    model_validation = validations[0] if validations else None

    unattributed = sum(1 for r in rows if r.get("model_version_id") is None)
    if unattributed and observed:
        db.record_event(job_id, "router.partial_provenance", {
            "rows_without_model_version": unattributed,
            "rows_total": len(rows),
            "attributed_to": model_version_id,
        })
        log.warning(
            "job=%s attributed to %s but %d/%d item rows have no recorded producer",
            job_id, model_version_id, unattributed, len(rows),
        )

    # The minimum-items floor is per modality: an image is a complete
    # observation of its subject, a video frame is one sample from a
    # distribution over frames, and a rule about sampling variance does not
    # apply to something that was not sampled. The video/audio floor of 3 is an
    # unvalidated placeholder -- see AggregationParams.min_items_for_score.
    params = AggregationParams.for_media(media_type)

    if media_type == "audio":
        face_count = None
        agg = aggregate(items, params)
    else:
        face_count = max_faces
        # 0 faces => undetermined, never a real/fake default.
        if not items:
            agg = aggregate([], params)
        elif media_type == "image":
            agg = aggregate_identity(items[0], params)
        else:
            agg = aggregate(items, params)

    routing = route(agg.score)

    db.write_result(
        job_id,
        result_class=routing.result_class.value,
        band=routing.band.value,
        aggregate_score=agg.score,
        model_version_id=model_version_id,
        aggregation_method=agg.method,
        aggregation_params=agg.params,
        item_count=agg.items_used,
        # What there was to begin with, so a consumer can tell a verdict off 1
        # usable frame of 50 from one off 50 of 50. Without it the row cannot
        # express "scored, but barely", which is what forced the floor to be
        # the only protection a reader had.
        items_total=agg.items_total,
        face_count=face_count,
        # On the audit row, not only in review_flags. The flag is operational
        # and gets read now; this row is what a dispute reads later, and it
        # must not claim full attribution when part of the evidence had no
        # recorded producer. 0 means measured and complete, which is itself a
        # statement worth having.
        items_unattributed=unattributed,
        model_validation=model_validation,
    )
    db.record_event(job_id, "router.decided", {
        "band": routing.band.value,
        "result_class": routing.result_class.value,
        "score": agg.score,
        "items_total": agg.items_total,
        "items_used": agg.items_used,
        "items_dropped_low_confidence": agg.items_dropped_low_confidence,
        "items_trimmed": agg.items_trimmed,
        "notes": agg.notes,
    })

    if routing.result_class is ResultClass.UNDETERMINED:
        log.info("job=%s undetermined (%s)", job_id, routing.review_reason)

    # Preserve the driving media BEFORE the Tier 1 delete removes derived/.
    # Ordering is load-bearing: reversed, the window would open over crops that
    # had already been deleted a moment earlier.
    if routing.extended_retention:
        driving_keys = _driving_keys(agg.used_items, rows)
        until, preserved = open_extended_retention_window(
            job_id, db, storage, driving_keys=driving_keys
        )
        log.info(
            "job=%s extended retention window until %s (fixed timer), %d object(s) preserved",
            job_id, until, len(preserved),
        )

    if routing.flag_for_review:
        reason = routing.review_reason or "flagged"
        urgency = routing.review_urgency.value
        db.flag_for_review(job_id, reason, urgency)
        notify_review_flag(job_id, reason, agg.score, routing.band.value, urgency=urgency)

    # Recorded is not the same as visible. An event in job_events is only found
    # by someone who already suspects something and goes looking, which is the
    # same failure mode as leaving the 60-80 band to pass silently into
    # deletion. Partial provenance means the stored model_version_id is true of
    # most of this job's rows but not all of them, and the job row IS the audit
    # trail -- so it goes through the same DB-flag-plus-alert path the bands
    # use. Independent of routing.flag_for_review on purpose: a likely_authentic
    # job with partial provenance raises no band flag and would otherwise be
    # completely silent.
    #
    # Low urgency: this is an attribution caveat on a result that is otherwise
    # sound, not a detection severity, and it is confined to jobs straddling the
    # migration-003 deploy. Flagged so it is never a silent pass-through; not
    # paging anyone.
    if unattributed and observed:
        reason = (
            f"partial provenance: {unattributed}/{len(rows)} item rows carry no "
            f"recorded model version; result attributed to {model_version_id}"
        )
        db.flag_for_review(job_id, reason, ReviewUrgency.LOW.value)
        notify_review_flag(
            job_id, reason, agg.score, routing.band.value,
            urgency=ReviewUrgency.LOW.value,
        )

    # Tier 1: TTL delete on inference completion. Checks the hold flag first.
    report = delete_media_for_job(job_id, db, storage)
    log.info("job=%s retention outcome=%s", job_id, report.outcome.value)

    status.publish(
        job_id, "complete",
        result_class=routing.result_class.value,
        band=routing.band.value,
        score=agg.score,
        media_deleted=report.outcome.value,
    )


def main() -> None:
    db, storage = Db(), storage_mod.build_storage()
    queue, status = build_queue(), JobStatus()
    run_worker(
        TOPIC_AGGREGATE,
        lambda m: handle(m, db=db, storage=storage, status=status),
        queue=queue, db=db, status=status,
    )


if __name__ == "__main__":
    main()
