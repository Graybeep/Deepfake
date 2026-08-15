"""Alerting for flagged results (Tier 3 substitute for the review dashboard).

CLAUDE.md: DB flag + Slack/email alert instead of a human-in-the-loop dashboard.
The DB flag (review_flags) is the durable part and is written first; this
notifier is best-effort on top of it. A dropped notification must never be the
reason a flagged job is lost, so failures here are logged and swallowed.
"""
from __future__ import annotations

import json
import logging
import os
import urllib.request

log = logging.getLogger("df.notify")

SLACK_WEBHOOK = os.environ.get("DF_SLACK_WEBHOOK", "")


def notify_review_flag(
    job_id: str,
    reason: str,
    score: float | None,
    band: str,
    *,
    urgency: str = "normal",
) -> bool:
    """Returns True if an alert was delivered. Never raises.

    `urgency='low'` is the 60-80 band. It uses the same DB-flag-plus-alert path
    as 40-60 -- the point is that nothing in that band passes silently into
    deletion -- but it is marked so an on-call filter can tell the two apart.
    """
    icon = ":eyes:" if urgency == "low" else ":mag:"
    text = (
        f"{icon} deepfake review flag ({urgency} urgency)\n"
        f"job `{job_id}`\nband `{band}` score `{score}`\nreason: {reason}"
    )
    if not SLACK_WEBHOOK:
        log.warning(
            "REVIEW FLAG urgency=%s job=%s band=%s score=%s reason=%s",
            urgency, job_id, band, score, reason,
        )
        return False

    try:
        req = urllib.request.Request(
            SLACK_WEBHOOK,
            data=json.dumps({"text": text}).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            return 200 <= resp.status < 300
    except Exception as exc:  # noqa: BLE001 - alerting must not fail the job
        log.error("review flag notify failed job=%s: %s", job_id, exc)
        return False
