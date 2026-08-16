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
  whole audit trail — treat it as such. That cuts both ways: anything qualifying the
  result belongs on this row, not only in a side table. `model_version_id` is derived
  from the item rows that actually produced the score (not from the queue message),
  and `items_unattributed` records how many of those rows had no recorded producer —
  NULL means never measured, 0 means measured and complete. A review flag or an alert
  is operational and gets read now; this row is what a dispute reads later, and it
  outlives both.
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
- Per-retrain recalibration and isotonic calibration — still deferred, and NOT because
  of time. Both need real trained weights and a held-out set, neither of which exists
  yet. More time does not unblock them; weights do.
- Redis Streams consumer groups — **built 2026-08-16, no longer deferred.** Streams is
  the default backend (`DF_QUEUE_BACKEND=streams`); the list backend is kept as a
  rollback path and both write the same `q:<topic>:dead` list. Any live consumer can
  reclaim a message whose worker stopped without acking it, once it has been idle past
  `DF_QUEUE_RECLAIM_MS` — so recovery no longer requires a worker restart, and a topic
  can have more than one consumer. Verified against a real Redis by
  `scripts/verify_queue.py`, which must run inside a container: Redis is on the
  internal network, and publishing a host port to test it would weaken the isolation
  being tested.

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

## Standing practice: docs/ regenerates in the same commit
`docs/solution-overview.html` is the source; the PDF beside it is generated from it
with headless Chrome. Any change to the schema, the test count, or what the system
claims to do regenerates both **in the same commit as the change**, not afterwards.

This has drifted twice already — first still claiming a 5-day sprint and 114 tests,
then 139 against an actual 147 — and both times it was caught by someone asking rather
than by anything in the process. A tracked document is worse than an untracked one
when it is stale, because being in the repo is itself a claim that it is current.
A written rule already lost to a human forgetting twice, so it is now mechanical.
`.githooks/pre-commit` runs `scripts/check_docs_current.py`, which fails the commit
when the document claims a test count pytest does not collect, and separately fails a
commit that touches `migrations/` without touching `docs/`. Enable it once per clone:

    git config core.hooksPath .githooks

The hook checks only what can be checked without judgement — the counted claims and
the schema case, which are exactly what drifted both times. Whether the prose still
describes the system is still a person's job. `--no-verify` exists for the case where
a migration genuinely changes nothing the document describes; reaching for it
routinely means the rule is wrong and should be changed rather than bypassed.

## Standing practice: prove the test fails first
Any test written to catch a specific bug must be **shown to go red before the fix is
in place**. Revert the fix, or disable the mechanism the test leans on, run the test,
confirm it fails, restore. A regression test that was never observed failing is not
evidence of anything — it may be asserting something that was already true.

This is not hypothetical. Three tests here passed without observing the thing they
named, each caught only by later inspection:
- `FakeDb` accepted `executemany` on a Connection, so the whole suite stayed green
  while the real DB write path was broken against psycopg3.
- The duplicate-delivery test asserted the aggregate *score*, which does not move
  under exact duplication — `rollup` takes `max`, which is idempotent. It passed
  identically with the natural key reverted.
- The group-recreate probe pushed its message *after* deleting the group, so it was
  satisfied whether the group came back at `0` or at `$`, and `$` abandons everything
  already in the stream.

Three instances is a pattern, not bad luck, and the tell is the same every time: the
test never observed the thing its name claims. The check is cheap and mechanical — do
it while writing the test.

**Mutate production source, and verify against a live probe. Where no live probe
exists for that path, the test must say so inline** — a trailing comment or a name,
e.g. `# FakeDb only: no live probe for the retention sweeps yet`. Not optional, and
not a nicety: "wherever one exists" on its own is a loophole that quietly makes
fake-only checking the default for every path that has no probe, and leaves the
weakness discoverable only by someone who later thinks to ask. Declaring it puts the
gap in front of whoever writes the next test in that file, which is the only moment it
is cheap to close. A test with no marker is a claim that a live probe covers it.
Mutating the fake and re-running pytest proves only that the test distinguishes the
fake's broken mode from the fake's fixed mode. That is a weaker claim than it sounds
in this codebase specifically, because the fake and the real query have diverged three
times — `FakeDb.executemany`, `get_items` not selecting `model_version_id`, and
`FakeDb.insert_items` accepting duplicates the unique index rejects — and every time
the fake was the more permissive of the two, so the fake-only check would have stayed
green. Revert the real code, rebuild the container, run `scripts/verify_attribution.py`
or `scripts/verify_queue.py` or `scripts/smoke_compose.py`, confirm red, restore, and
confirm green again.

Two mutations, not one, when a value is involved: dropping the write and writing a
wrong constant fail differently, and a check that only proves the column exists cannot
see the second. Mutating real source also forces you to read the file rather than your
memory of it — that is how the stray `insert_items` docstring was found.
