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
  for the video/image-level class. **Confirmed 2026-08-29, and deliberately not
  locked.** Worst-case stays as the aggregator, but it is no longer the only
  artifact: `face_evidence` publishes the per-face array (score, confidence,
  frame, pixel size) behind the label, so a crowd scene is distinguishable from a
  close-up without changing the label. Choosing between worst-case, largest-face
  and highest-confidence would have baked a lossy reduction into the contract for
  every downstream consumer — more expensive to undo than the choice is to make.
  There is deliberately NO per-face threshold: the right rule varies the bar by
  face size with a false-positive rate flat across buckets, and that needs
  labelled data that does not exist. Do not add an invented per-face constant.
  Face geometry (`face_w`/`face_h`, migration 006) was previously extracted and
  then discarded in the CPU worker, so no historical job records how big any face
  was — absolute pixels only, since frame dimensions are still unrecorded.
- Minimum items before a verdict is **per modality**: 1 for image, 3 for
  video/audio. An image is a complete observation; a video frame is one sample.
  3 is an unvalidated placeholder, labelled as one in the code. Every result
  carries `items_total` and `coverage` so the floor is not the only protection a
  reader has — `undetermined` should not be doing work that a coverage number
  does better.
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
  persistence is configured — `measured: yes` 2026-08-18, a default `redis:7-alpine`
  has `appendonly no` but RDB `save` points on, so an unconfigured restart loses up to
  the last snapshot window (an hour under low write volume), not everything. AOF
  `everysec` is set in compose and `assert_persistence_enabled()` refuses to boot
  without it — but note what that buys: `measured: yes`, a SIGKILL of the Redis
  process loses nothing even without fsync, because the writes are already in the
  kernel page cache. AOF is protection against host-level failure, not against a
  container restart.

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
NVIDIA is explicit: "This is the only driver you need to install. Do not install any
Linux display driver in WSL" — installing one can overwrite the host driver mapping,
since the Windows driver is stubbed into WSL2 as `libcuda.so`.
`measured: no (source)` [NVIDIA CUDA on WSL guide, read 2026-08-18:
<https://docs.nvidia.com/cuda/wsl-user-guide/index.html>]

`nvidia-container-toolkit` is the Linux-side piece and installs inside a WSL distro,
not on Windows — same source. But `measured: yes` here on 2026-08-16: it is
**not installed in this machine's Ubuntu**, and `docker run --gpus all` works anyway,
because Docker Desktop supplies GPU support from its own bundled distro rather than
from yours. Both statements are true of different distros. Install it only if you run
a native Docker engine inside Ubuntu; do not read a passing GPU check as evidence that
it is present.

Verify with `nvidia-smi` inside WSL, then `docker run --gpus all ... nvidia-smi`,
before wiring the torch backend into compose — not on the day the weights land.
[`measured: yes` 2026-08-16 on this box: driver 610.43.02, RTX 4060 Laptop, 8 GB.]

**CORRECTED 2026-08-18.** The old claim was `measured: no (reasoned)` while reading
exactly like `measured: yes` — the worked example for state 3 below. This file
previously said MinIO's image had dropped curl, and that a `curl -f .../health/live` healthcheck therefore fails regardless of
whether MinIO is up. **Both halves are false**, `measured: yes` by running the image:
`minio/minio:latest` ships curl at `/usr/bin/curl`, and `curl -sf
http://localhost:9000/minio/health/live` returns exit 0 against a healthy server.
(`docker run --rm --entrypoint sh minio/minio:latest -c "command -v curl"`, then the
same check inside a running server.) `wget` is genuinely absent, which may be where
the belief started.

Keep `mc ready local` anyway, for the reason that actually holds: it reports *cluster
readiness*, whereas `/health/live` reports liveness, and compose gates `minio-init` on
this healthcheck — a live-but-not-ready server would let bucket creation race.

## Provenance of the external-world claims in this file
Claims about this repo can be checked by reading it. Claims about the outside world —
what a vendor image contains, what a database or a protocol does — cannot, and those
are the ones that decay silently or were never right. Backfilled 2026-08-18; the rule
is not forward-only, and an uncited claim already in the file is the same problem as a
new one.

**Every statement of fact about running software carries an explicit `measured:` tag.**
Mandatory, and independent of how confident the sentence sounds. There are three
states, not two:

1. `measured: yes` — someone ran it. Name the command or the probe.
2. `measured: no (source)` — read in a vendor doc. Cite it.
3. `measured: no (reasoned)` — worked out from how the thing presumably behaves.

State 3 is the dangerous one, because it is invisible from the outside. The removed
curl claim was state 3: specific, actionable, load-bearing, and wrong — it named a
failure mode, prescribed a fix that happened to work, and so nothing ever contradicted
it. It read exactly like state 1. Confidence is self-report and this claim had plenty
of it, so the tag cannot be inferred from tone and has to be written down by whoever
makes the claim, at the moment they make it, when they still know which of the three
they did.

An untagged factual claim about external software is treated as state 3.

Also applies to code comments asserting external behaviour: cite the probe that
demonstrates it (`verify_queue.py`, `verify_retention.py`, `verify_attribution.py`,
`smoke_compose.py`) rather than stating it flat.

`measured: yes` — verified by running it on this machine, which is stronger evidence
than a citation:

- **Presigned PUT cannot carry a size cap; a POST policy can.** `measured: yes` —
  `scripts/smoke_compose.py` mints a grant with a 1 KiB bound, then shows storage
  accepting a body under it and rejecting one over it with 400, writing nothing.
- **`NULLS NOT DISTINCT` needs PG15+, and without it audio rows (NULL `face_index`)
  bypass the unique index.** `measured: yes` — both duplicate shapes confirmed rejected
  by name against live Postgres 16; see `migrations/002`.
- **Redis Streams: a taken-but-unacked message is claimable by another consumer once
  idle, and XGROUP CREATE at `0` replays entries already in the stream while `$` would
  abandon them.** `measured: yes` — `scripts/verify_queue.py`, including the
  plant-then-destroy case that distinguishes `0` from `$`.
- **psycopg3 `executemany` is a cursor method; `Connection` has only `execute()`.**
  `measured: yes`, the hard way — it dead-lettered every job against a real database
  while the whole suite stayed green.
- **This stack's Redis has AOF on.** `measured: yes` — the running server reports
  `appendonly yes`, and `jobstatus.assert_persistence_enabled()` refuses to start
  without it, exercised on every worker boot.
- **CORRECTED 2026-08-18: a default `redis:7-alpine` does NOT keep everything in
  memory.** `measured: yes` — running the image with no command override reports
  `appendonly no` but `save 3600 1 300 100 60 10000`, so RDB snapshotting is on by
  default. A restart loses up to the last save window, which under low write volume
  can be an hour — not "all in-flight state", which is what this file used to say.
  Turning AOF on is still right (it narrows that window to ~1s), but the reason was
  overstated. Found by splitting this from the measured bullet above and then actually
  running it: those were one bullet until this pass, and merging a measured fact with
  an assumed one is how the assumed half inherits the other's credibility.

`measured: no` — flagged rather than dressed up:

- **`min_confidence=0.3` and `trim_frac=0.1` have never affected a single result.**
  `measured: yes` that they are inert, `measured: no` that the values are right. Across
  378 item rows the lowest confidence ever produced is 0.6, so nothing has ever been
  dropped. Across 40 decisions and 228 items, zero were trimmed: 10 percent of a 6 item
  job floors to 0, and most jobs carry fewer than 10 items. So the aggregation that this
  project describes as a confidence weighted trimmed mean has in practice been a
  weighted mean with no trimming and no dropping. Confidence WEIGHTING does happen, the
  weights genuinely differ. The two robustness mechanisms are what have never fired.
  They are untuned defaults wearing the appearance of tuned ones, and they cannot be
  tuned until real weights produce a real score and confidence distribution.

- **CORRECTED 2026-08-18: AOF `everysec` does NOT lose writes when the Redis process
  dies.** `measured: yes` — 50,000 keys written and acked with
  `--appendonly yes --appendfsync everysec`, then `docker kill` (SIGKILL, no clean
  shutdown), then restart: **50,000 survived, zero lost.** The reason is that `write()`
  already handed the data to the kernel; page cache outlives the process. fsync
  protects against *machine* failure — power loss, kernel panic, VM hard-stop — not
  against a container dying, being OOM-killed, or being restarted.
  The ~1s window is real but applies only to host-level failure. `measured: no
  (source)` for that part, and not measurable here without hard-stopping the machine.
  This matters for how `sweep_stalled_jobs` is described: the motivating story was
  "Redis loses the push in its everysec window", and for the common failure — a
  container restart — that does not happen.
- **CORRECTED 2026-08-18: the stalled sweep fired on healthy jobs.** `measured: yes`
  -- a job queued 9h earlier whose message was still sitting in the stream was marked
  failed with the reason "stalled in-flight with no queue message", which was simply
  untrue of it, and the terminal sweep would then have deleted its upload. Age cannot
  tell "no message will ever come" from "a worker is behind", and after the correction
  below the second case is far MORE likely than the first: a worker down for over 6h is
  ordinary, the write/push race is microseconds wide. The sweep was destroying more good
  jobs than bad. `sweep_stalled_jobs` now asks the queue directly via
  `has_message_for()` across every topic and skips any job with work still waiting.
  Verified live: backlogged job untouched and still `queued`, genuinely stranded job
  still swept.
  Note what this means about threshold tuning: 6h was never the load-bearing part. The
  queue check is. Measured job duration is 0.65s at worst over 65 jobs, so any
  threshold in hours is far above real processing time; what mattered was the signal,
  not the number.

- **CORRECTED again, same day: the replacement signal was itself broken under the
  condition it exists for.** `measured: yes` -- `has_message_for` scanned with
  `xrange(count=1000)` and so returned False for anything past entry 1000. On a
  1500 deep stream, entry 0 was found and entries 1200 and 1499 were not. That is the
  destructive direction (a live job reported as having no work, then swept and its
  upload deleted) and it failed precisely under deep backlog, which is what a worker
  down for hours produces. It was justified by reasoning about Redis and verified only
  against worker-down, never against depth. Now paginates the whole stream, and returns
  True on any uncertainty past a 100k safety bound, because wrongly keeping media costs
  a delay while wrongly deleting it destroys a good upload. Three shapes are now checked
  live in `verify_queue.py`: no message at all, a message taken but unacked, and a
  message past the first page.
  A probe bug surfaced with it: those checks originally ran on the probe topic, which
  `has_message_for` does not scan, so every answer was False and the two negative cases
  passed for no reason. The positive cases failing is what exposed it. A check that can
  only pass is not a check.

- **A crash between the Postgres status write and the Redis push strands the job.**
  `measured: yes`, and NARROWER than this file implied. Redis being unavailable does
  **not** strand anything: `enforce_rate_limit` is the first line of `mark_uploaded`
  and the limiter is Redis-backed, so with Redis stopped the request fails with 500 at
  the door and the status write never happens. Measured 2026-08-18: job stayed
  `awaiting_upload`, stream length 0, nothing stranded.
  The strand needs Redis alive enough to pass the rate limiter AND the status publish,
  then to fail in the window between the Postgres commit and the XADD — microseconds,
  not "Redis crashed". `sweep_stalled_jobs` stays: the window is real and the defence
  is cheap. But it defends a far rarer event than "Redis went down", and the 6-hour
  threshold should be read in that light rather than as protection against a routine
  outage.
- **Two consumers running different model versions during a rolling deploy.**
  `measured: no (reasoned)` — the mixed-model state was *constructed* in
  `verify_attribution.py` by writing differing item sets directly. That the refusal
  path works is `measured: yes`; that a real rolling deploy produces this state is not,
  and with atomic `insert_items` it may be hard to reach naturally. Kept as
  defence-in-depth, not described as a live risk.

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

## Model trust level, and how the caveat is produced
Every result carries a caveat stating how far its score can be trusted, and the
mechanism **fails closed**. `ModelVersion.validation` is one of `placeholder`,
`research-checkpoint`, `production-validated`; it is a required field, so a new
detector cannot ship without declaring one. It is written onto each item row and
derived by the router from the rows that produced the score — same rule as
`model_version_id`: the rows are the evidence, the queue message is hearsay.
`jobs.model_validation` carries it on the audit row.

`_public_job` derives the advisory from that column. **NULL or any unrecognised
value produces the strongest caveat, not none.** The previous version matched the
substring `stub` against `model_version_id`, which failed OPEN: loading a real
checkpoint changes the id to `face-efficientnet_b4-<hash>`, the substring vanishes,
and every caveat disappears silently — at exactly the moment scores start looking
plausible enough to be believed. Do not reintroduce a check keyed on how a model is
named.

`production-validated` takes two keys: `DF_MODEL_VALIDATION` plus a non-empty
`DF_MODEL_VALIDATION_SIGNOFF` naming who validated the weights against this
pipeline. The "Cannot claim" list below still forbids the claim; the two-key gate
makes reaching it a deliberate, attributable act rather than a one-character change.

## Weights: decided, not yet integrated
Chosen placeholder: the **DFDC-winner EfficientNet** (`selimsef/dfdc_deepfake_challenge`).

**Licence caveat, accepted knowingly.** The repository code is MIT (verified). The
*weights* are trained on Meta's DFDC dataset, whose terms are not published on the
dataset page and whose download is gated behind an account and an agreement. Whether
a dataset licensor's terms flow through to derived model weights is unsettled and
jurisdiction-dependent. **Do not ship these weights commercially without legal
review.** This is a stated, nameable risk; that is why it is written here.

Rejected, with reasons AND sources so the call can be re-checked rather than taken
on trust. All assessed 2026-08-17 by reading the licence text, not from memory:

- **FaceForensics++** — non-commercial research/education only, binds a for-profit
  employer, requires manual approval via form. Hard blocker.
  <https://kaldir.vc.in.tum.de/faceforensics_tos.pdf> ·
  <https://github.com/ondyari/FaceForensics>
- **DeepfakeBench** — CC BY-NC-4.0 (non-commercial) *and* per-detector training data
  undocumented. Worse than the chosen option on both axes, which is why it lost
  despite shipping EfficientNetB4 weights.
  <https://github.com/SCLBD/DeepfakeBench>
- **Apache-2.0 HuggingFace checkpoints** (ViT/SigLIP) — clean licence, undocumented
  training data, and the wrong architecture for the existing torchvision wrapper.
  Clean-licence-unknown-provenance is not honesty, it is unverifiability: it cannot
  rule out FF++/DFDC-derived data, train/eval leakage, or simple unsuitability. A
  known, nameable licence risk beats an unknown provenance one; "clearly marked"
  would have labelled the licence and left the actual gap unlabelled.
  <https://huggingface.co/dima806/deepfake_vs_real_image_detection> ·
  <https://huggingface.co/prithivMLmods/Deep-Fake-Detector-v2-Model>

Chosen option's sources: <https://github.com/selimsef/dfdc_deepfake_challenge/blob/master/LICENSE>
(MIT, verified) · <https://ai.meta.com/datasets/dfdc/> (no published terms on the
dataset page; access gated behind an account and an agreement).

**Every entry here carries its source — existing ones too, not just new.** A rejection
with reasons but no citation is a claim, and this file's whole job is to not be that.
Stating the rule for future entries only would grandfather in exactly the claims that
have had longest to go stale; see "Provenance of the external-world claims in this
file", where doing that backfill immediately turned up a false one.

Audio has no checkpoint under this decision and stays on the stub. Expect a mixed
state: video/image at `research-checkpoint`, audio at `placeholder`. The advisory is
per-job and derived per-model, so it reports this correctly without special-casing.

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

**The first of the two gaps is now half closed, and the half that remains is the
larger one.** 2026-08-29: `tests/psycopg_shape.py` provides a psycopg-shaped
connection and `tests/test_db_api_shape.py` patches `psycopg.connect` so every
public `Db` method executes its **real body** — which no test had ever done,
because `FakeDb` replaces `Db` wholesale and so stands in for the layer *above*
the one that broke. The stand-in has no `executemany` on its Connection and no
`fetchone`, matching psycopg3.

What makes it more than another fake: its permitted surface is not written down.
`test_fake_surface_is_a_subset_of_real_psycopg` reads the public names off the
installed `psycopg.Connection`/`psycopg.Cursor` at runtime and fails if the
stand-in exposes anything the library does not. A hand-written list would have
been a fourth divergence — and, per the provenance rule, a state 3 claim about
psycopg wearing the appearance of a measured one. Both shipped bugs are now
mutations in `scripts/mutate.py` (`DB_MUTATIONS`) and both report RED,
witness-checked.

**What it still cannot see, and this is the load-bearing part: no SQL is
executed.** It proves the API shape and the `get_items` column list. It proves
nothing about whether a statement is valid, whether a column exists, or whether
a predicate selects the right rows. `verify_attribution.py`,
`verify_retention.py` and `smoke_compose.py` remain the only evidence of that,
and a passing pytest run is still not evidence the DB layer works.

The second gap is untouched:
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

This has drifted three times now — first still claiming a 5-day sprint and 114 tests,
then 139 against an actual 147, then **README.md at 106 against an actual 157** — and
every time it was caught by someone asking rather than by anything in the process. A
tracked document is worse than an untracked one when it is stale, because being in the
repo is itself a claim that it is current.
A written rule already lost to a human forgetting twice, so it is now mechanical.
`.githooks/pre-commit` runs `scripts/check_docs_current.py`, which fails the commit
when a tracked document claims a test count pytest does not collect, and separately
fails a commit that touches `migrations/` without touching `docs/`. Enable it once per
clone:

    git config core.hooksPath .githooks

The hook checks only what can be checked without judgement — the counted claims and
the schema case, which are exactly what drifted all three times. Whether the prose still
describes the system is still a person's job. `--no-verify` exists for the case where
a migration genuinely changes nothing the document describes; reaching for it
routinely means the rule is wrong and should be changed rather than bypassed.

**The third drift was the guard's own scope, and that is the lesson worth keeping.**
The checker was written to stop test-count drift and then pointed at `docs/` alone,
so `README.md` drifted 51 tests while `docs/` stayed correct and the hook passed
every commit. The rule was never "docs/ must be current" — it is "a tracked document
that is stale is worse than none" — and that applies to every tracked document.
`check_docs_current.DOCUMENTS` now lists `docs/solution-overview.html` **and**
`README.md`; any new document making a counted claim goes in that list on the day it
starts making it. The check also fails when a document matches *none* of its own
patterns, so a reworded claim surfaces as "silently stopped checking" rather than
passing — that direction is verified too, not just the wrong-number one.

Corrected 2026-08-26 in the same pass that found it, along with nine other stale
claims in `README.md` and `DECISIONS.md`: the compose gate described as unsatisfied,
the queue described as Redis lists, the upload grant described as a presigned PUT in
three places, and the model advisory described as matching the substring `stub`
against `model_version_id` — the fail-open check this file already says was removed.
A document repeating a mechanism that was deleted for being unsafe is how it gets
rebuilt.

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

**Witness the mutation before believing the verdict.** Use `scripts/mutate.py`, which
evaluates a witness snippet against the original and the mutated source and reports
`NO-OP` — withholding RED/GREEN entirely — when the two behave identically. This is
not ceremony: a mutation written as `return [] or [...]` edits the source, compiles,
and reads like a mutation in the diff, while evaluating to the original list. It was
reported as "the test is vacuous" when the test was fine and the mutation did nothing.
A no-op mutation and a vacuous test produce byte-identical output, so RED/GREEN cannot
distinguish them, and the practice that exists to stop tests being trusted blindly was
itself being trusted blindly. That mutation is kept in `ADVISORY_MUTATIONS` as a
permanent regression case: the harness must keep reporting NO-OP for it.

**Mutate a predicate in BOTH directions, and give every setup a positive control.**
Two distinct failures, learned the hard way one after the other:

- *One direction is not enough.* Mutating `has_message_for` to constant False turns
  only the positive checks red; constant True turns only the negative ones red. A
  suite mutated one way looks fully covered while half of it observes nothing.
  Demonstrated 2026-08-18; both directions are recorded in `verify_queue.py`.
- *A passing negative check proves nothing without a positive control in the same
  setup.* The `has_message_for` checks first ran on the probe topic, which the
  function does not scan, so every answer was False and both negative checks passed
  for no reason at all. No mutation of the function would have exposed that, because
  the function was never reached. What exposed it was the positive checks in the same
  setup failing. So: any block asserting "X is absent" needs a sibling asserting "X is
  present", or it is only evidence that the setup is broken in the convenient
  direction.

The harness has no check of its own either. A wrapper around this reported
"16 PASS / 0 FAIL" under a mutation that manually produces 2 FAIL, almost certainly
execing against a container that had not finished rebuilding. Witness the mutation
inside the environment the test actually runs in, not just in the source tree.

**Same class, found again 2026-08-29, and this one was inside the harness.**
`measured: yes`: mutating `floors = {"image": 1` to `{"image": 3` — one character,
identical byte length — made the harness report NO-OP for a mutation that plainly
changes behaviour. Afterwards `grep` showed `1` in the source while every fresh
interpreter reported `3`, until `__pycache__` was deleted by hand.

CPython validates a cached `.pyc` against the source's (mtime **in seconds**,
size). A mutation that preserves file size, written and then restored inside the
same one-second tick, leaves a `.pyc` whose header matches the restored file
exactly — so mutated bytecode is treated as current and reused indefinitely. Two
consequences, and the second is worse than a wrong verdict: the harness compared
two poisoned runs and withheld a true RED, and **pytest afterwards runs against
mutated bytecode with clean source on disk** — green proving nothing, or red
indicting code that is fine.

`-B` does not fix it: that stops Python *writing* a `.pyc`, not *reading* the
poisoned one already there. `scripts/mutate.py` now points every subprocess at a
scratch `PYTHONPYCACHEPREFIX` it empties before each run and after each restore,
and sweeps stale in-tree `__pycache__` too. The same-length mutation is kept in
`REPORTING_MUTATIONS` as a permanent regression case: it must report RED, and the
`[] or [...]` case must still report NO-OP — one of each, because a fix that made
everything report RED would look identical to success.

The general lesson, which is the third time this file has recorded a version of
it: **a mutation that changes no bytes of file length is the dangerous one**, and
"I edited the file and re-ran" is not evidence the edit was executed.
