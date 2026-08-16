# CLAUDE.md

Persistent context for this repo. Read before making architecture, safety, or scope
calls — especially anything touching deletion, retention, or claims about what's built.

## What this is
Deepfake detection service. Three ingest pipelines (video, image, audio) converging on
shared aggregation, score-band routing, and retention logic. Solo/small-team.

**The 5-day MVP sprint constraint was lifted on 2026-08-16 — there is more time.**
The tiers below still stand, but they now describe *order and dependency*, not what
fits in a week. Re-read any deferral that was justified by the clock: some had a
second, independent reason and still hold, some were purely budget and are now open.
Each is marked. "We have time now" is a reason to schedule something, never on its own
a reason to build it — a half-built adversarial pre-classifier is still worse than an
honestly absent one. Don't quietly build past a tier without updating this file.

## Pipelines
Video: Ingest → Frame Sample → Face Extract → Face Align → EfficientNet (face model) → Aggregate → Router
Image: Ingest → Face Extract → Face Align → EfficientNet (face model) → Router (aggregate = identity)
Audio: Ingest → Chunk → Spectrogram → EfficientNet (audio model) → Aggregate → Router

- Face model is shared across video and image — same weights, one model_version_id.
  Audio is a separate model with its own model_version_id.
- Aggregation default: weighted/trimmed mean over per-frame or per-chunk scores,
  weighted down by detection/alignment confidence. Never plain mean.
- Face Extraction returning 0 faces → `undetermined` class. Never silently default
  into real/fake.
- Face Extraction returning >1 face → score each face, roll up to worst-case severity
  for the video/image-level class. This is a default, not fixed — confirm before
  anything downstream assumes otherwise.
- Score-band thresholds (<20 / 40–60 / >80) apply to the aggregated score, not a raw
  per-item score. Face model and audio model each get their own calibration pass —
  don't share one curve across both.
- Bands must be a total partition of the range — routing has no undefined input.
  20–40 can auto-clear like <20; being wrong there just means a probably-real item
  gets deleted on schedule. 60–80 is the riskier gap: it sits next to the
  highest-severity threshold, whose calibration this file already says isn't trusted
  enough to display as a raw percentage. Give it at least a passive log or low-urgency
  flag — the same DB-flag-plus-alert pattern already used for 40–60 — not silent
  pass-through to normal deletion.

## Tier 1 — non-negotiable
- TTL-delete on raw video/face crops, triggered on inference completion. Every delete
  call checks the hold-flag column first — build the flag column and the check in the
  same commit as the delete path, before anything sets it true.
- Presigned S3 uploads, bypass API for large files. The signature is bound to the
  exact host used at signing time — swapping the hostname in a generated URL breaks
  it. Use two client configs: an internal one for server-side calls, a separate one
  pointed at whatever host/port the browser can actually reach, used only for
  presigned-post. It must be a POST policy, not a PUT URL: a presigned PUT signs the
  URL and not the body length, so it is an unbounded write to the bucket. These bytes
  never traverse the gateway — ingress rate limiting never sees them — so a
  `content-length-range` condition in the policy is the only place
  DF_MAX_UPLOAD_BYTES can be enforced at all.
- Rate limiting on ingress.
- Postgres job row: hash + model_version_id + aggregation method/params. This is the
  whole audit trail — treat it as such.
- Job status: Redis key + WebSocket push, polling fallback on reconnect. Confirm Redis
  persistence (AOF/RDB) is on — default in-memory config loses all in-flight job state
  on restart.

## Tier 2 — stub, don't skip silently
- Retention: score-band flag → cold storage, fixed 30-day timer. Call this "extended
  retention window" everywhere in code, comments, and copy — never "legal hold." A
  timer that auto-expires during an active dispute is a liability, not a safeguard.
- **The window protects the flagged media itself — specifically the face crop(s) that
  drove the score, not the full raw source — not just the job row.** The job row and
  per-item scores are already covered by Tier 1's audit trail for every item
  regardless of band; if the window only re-protects that, the >80 branch has nothing
  a dispute could actually use. The hold-flag check from Tier 1 applies to every
  delete path that can touch this media — the completion-triggered delete and any
  crash-recovery sweeper alike, not just one of them. This still needs real legal
  sign-off before it's relied on for an actual dispute — treat the above as the
  engineering default, not the final word.
- Calibration: temperature scaling once at launch. Flag as launch-snapshot only, not
  tied to a recalibration pipeline.
- DLQ: retry-limit + dead-letter + log line. No PagerDuty yet.

## Tier 3 — deferred, mark clearly, don't half-build
- Adversarial-input pre-classifier — not built. Scores are manipulable via adversarial
  perturbation; documented, not hidden.
- Human-in-the-loop dashboard — DB flag + Slack/email alert instead.
- AV scanning — network-isolated, locked-down CPU worker containers substitute. Build
  this in the same phase as the CPU preprocessing worker, not later — a worker parsing
  untrusted video/audio before isolation exists is an open compromise window.
- Redis Streams consumer groups, per-retrain recalibration, isotonic calibration —
  these were "week 2+" under the old budget. They are gated on a dependency, not on
  time: per-retrain recalibration and isotonic calibration both need real trained
  weights and a held-out set, neither of which exists yet. Redis Streams is the one
  item here that extra time genuinely unblocks.

## Deployment
MVP: Docker + docker-compose, single GPU node, one image per service (gateway,
CPU-preprocess, GPU-inference, aggregation/router). The gate was: do not start K8s
manifests until docker-compose runs end-to-end. **That gate is now satisfied —
verified 2026-08-16 — and the 5-day budget behind it is gone.** K8s is therefore
schedulable rather than forbidden.

It is still not next. Cluster provisioning, secrets, ingress and PVCs remain multi-day
work, and on a single GPU node they buy nothing the compose stack does not already do.
The trigger for K8s is a second node, a real availability requirement, or autoscaling
the GPU pod — not the availability of time.
Target state (post-MVP): K8s, one Deployment per service above, HPA on the
GPU-inference pod specifically — that's the cost driver — managed or PVC-backed
Postgres/Redis.
GPU passthrough is a separate prerequisite from installing Docker Desktop — the
WSL2-enabled NVIDIA driver installs on the Windows host, never inside the WSL distro.
`nvidia-container-toolkit` is the Linux-side piece — installs via `apt` inside the
WSL distro (Ubuntu), not on Windows; it needs a distro to live in, it doesn't replace
one. Verify with `nvidia-smi` inside WSL, then `docker run --gpus all ... nvidia-smi`,
before wiring the torch backend into compose — not on the day the weights land.
MinIO's official image dropped curl — a healthcheck written as `curl -f .../health/live`
fails immediately regardless of whether MinIO is actually up. Use `mc ready local`.

## Before any push to a public remote
Run a secrets scan (`gitleaks detect --source . -v` or trufflehog) against full git
history, not just the working tree — deleting a credential in a later commit doesn't
remove it from a history that's already public. Do this before publishing, not after.

## Platform
Web app, not native. Presigned S3 + WebSocket are already web-native. The timeline
half of this argument is gone, but the substantive half is not: there is still no
stated offline or push requirement, and two extra build targets plus store review is a
permanent ongoing cost, not a one-off spend that more time absorbs. Revisit only if
one becomes a real requirement.

## Cannot claim in code, comments, or user-facing copy
- "Adversarial robustness" — not built.
- "Legal hold" — it's a fixed-timer extended retention window.
- "Production-validated calibration" — true only for the launch snapshot.
- "BIPA/GDPR compliant" — partial deletion + partial audit trail is meaningfully
  better than nothing, but compliance is a legal determination this codebase doesn't
  get to assert.

## Testing status
TTL deletion is asserted against the storage backend, not a mock's call log — good,
keep it that way. The hold-flag gap is closed: `tests/test_retention_hold_gate.py`
covers all three delete triggers that can touch preserved media — completion
(`delete_media_for_job`), crash recovery (`sweep_undeleted`), and cold-storage expiry
(`expire_extended_retention`, both directly and via `sweep_expired_windows`) — each
tested held and unheld, asserting on the face crops rather than the job row. The
unheld half earns its place: a test that only checked the held case would also pass
against a delete path that was simply broken.

Two gaps remain, both about what an in-process suite structurally cannot see:
- Every test runs against `FakeDb`, which does not mirror psycopg3's
  Connection/Cursor split. That is precisely how `insert_items` shipped calling
  `executemany` on a Connection — dead-lettering every job against a real database
  while the suite stayed green. A passing pytest run is not evidence the DB layer
  works.
- The presigned upload size cap is enforced by object storage.
  `tests/test_presign_policy.py` proves the code still asks for the
  `content-length-range` condition; only `scripts/smoke_compose.py` proves storage
  honours it. Nothing in pytest can.

So: run `scripts/smoke_compose.py` against a live stack before treating "the tests
pass" as "the system works."
