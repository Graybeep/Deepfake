# Deepfake detection service

Three ingest pipelines (video, image, audio) converging on shared aggregation,
score-band routing, and retention logic.

**Read `CLAUDE.md` before changing anything touching deletion, retention, or
claims about what is built.** `DECISIONS.md` records the judgement calls made
during scaffolding and the two open questions that still need a human answer.

> **The detector is a placeholder.** The default inference backend
> (`DF_INFERENCE_BACKEND=stub`) is a deterministic hash-based scorer, not a
> trained model. It exists so the pipeline can be built and tested before
> weights land. Every stub result reports `is_real_detector=false` and declares
> `validation=placeholder`, and the API derives its advisory from that declared
> level — **never** from how the model is named. A name-based check fails open:
> load a real checkpoint, the id stops containing `stub`, and every caveat
> disappears at exactly the moment scores start looking believable.

---

## Run it without any infrastructure

The fastest way to see the pipeline behave — no Postgres, Redis, or S3:

```bash
python -m venv .venv && .venv/Scripts/pip install -r requirements-dev.txt

.venv/Scripts/python scripts/run_local.py video sample.mp4
.venv/Scripts/python scripts/run_local.py image sample.jpg --faces 0   # undetermined path
.venv/Scripts/python scripts/run_local.py image sample.jpg --faces 3   # worst-case rollup
```

## Tests

```bash
.venv/Scripts/python -m pytest
```

184 tests, no infrastructure required. The suite covers the two pipeline rules
that must never be relaxed (0 faces ⇒ undetermined, >1 face ⇒ worst-case
rollup), aggregation, band routing, the rate limiter, DLQ behaviour, that **TTL
deletion actually deletes**, and that **the hold flag blocks every delete path
that can touch preserved media**.

`tests/test_retention_ttl.py` asserts against the storage backend, not a mock's
call log: a mock proves a call was made, not that the bytes are gone.

`tests/test_db_api_shape.py` runs every `Db` method's **real body** against a
psycopg-shaped connection. Until this existed no test executed a single line of
`db.py` — `FakeDb` replaces `Db` wholesale, which is how `insert_items` shipped
calling `executemany` on a Connection (a psycopg3 *cursor* method) and
dead-lettered every job against a real database while the suite stayed green.
The stand-in's allowed surface is read off the installed `psycopg` classes at
runtime rather than hardcoded, so it cannot drift wider than the library the way
every previous fake did.

**It executes no SQL.** It proves the API shape and that `get_items` still
selects the columns the router reads. Whether a statement is valid or a
predicate selects the right rows is still only proven by the live probes.

`tests/test_end_to_end.py` drives the **real worker handlers** (preprocess →
inference → router) against in-memory Postgres/Redis/S3 stand-ins. It catches
broken queue payload contracts, missing DB writes, and a delete that never fires
in about a second, instead of after a multi-minute image build.

It does **not** cover container networking, image builds, real presigned URLs,
or the WebSocket transport. Those need compose.

## Run the full service

`docker compose up` **runs end to end** — `measured: yes` 2026-08-16 on the
development machine, gated by `scripts/smoke_compose.py`.

```bash
cp .env.example .env
docker compose up --build -d
python scripts/smoke_compose.py     # gates the "runs end to end" claim
```

### GPU passthrough is a separate prerequisite

Installing Docker Desktop does **not** give you GPU inference. The WSL2-enabled
NVIDIA driver installs on the **Windows host**, never inside the WSL distro.
Verify the chain before wiring the torch backend into compose — not on the day
the weights land:

```bash
wsl nvidia-smi                                              # driver visible in WSL
docker run --rm --gpus all nvidia/cuda:12.4.0-base nvidia-smi   # visible to containers
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build -d
```

`scripts/smoke_compose.py` exercises what the in-process tests cannot: the
presigned upload, the queue between containers, the audit trail on a real job,
that the bytes are actually gone from the bucket afterwards, and that the rate
limiter returns 429 under burst. It exits non-zero on the first failure.

The old gate — no Kubernetes manifests until compose runs end to end — is
**satisfied**, so K8s is schedulable rather than forbidden. It is still not
next: on a single GPU node it buys nothing compose does not already do, while
cluster provisioning, secrets, ingress and PVCs remain multi-day work. Per
CLAUDE.md the trigger is a second node, a real availability requirement, or
autoscaling the GPU pod — **not** the availability of time.

---

## Architecture

```
                  ┌──────────┐  presigned POST    ┌────────┐
   client ───────▶│ gateway  │───────────────────▶│  S3    │
                  └────┬─────┘                    └───┬────┘
                       │ enqueue                      │
                       ▼                              ▼
                 ┌───────────┐  face crops /   ┌──────────────┐
                 │    cpu-   │  spectrograms   │     gpu-     │
                 │ preprocess│────────────────▶│  inference   │
                 └───────────┘                 └──────┬───────┘
                  (network-isolated)                  │ job_items
                                                      ▼
                                              ┌───────────────┐
                                              │ router        │
                                              │ aggregate →   │
                                              │ band → verdict│
                                              │ → TTL delete  │
                                              └───────────────┘
```

| Pipeline | Path |
|---|---|
| Video | Ingest → Frame Sample → Face Extract → Align → face model → Aggregate → Router |
| Image | Ingest → Face Extract → Align → face model → Router (aggregation = identity) |
| Audio | Ingest → Chunk → Spectrogram → audio model → Aggregate → Router |

The face model is **shared** by video and image — same weights, one
`model_version_id`. Audio is a separate model with its own id and its own
calibration temperature.

### Aggregation

Confidence-weighted, symmetrically trimmed mean (`weighted_trimmed_mean.v1`).
Never a plain mean: a plain mean lets a handful of bad frames drag a verdict
around and gives low-confidence detections the same vote as clean ones.

The method name and exact params are written onto every job row.

**Two of the three mechanisms in that name have never once fired.**
`measured: yes` that they are inert; `measured: no` that the values are right.
Across 378 item rows the lowest confidence ever produced is 0.6, so
`min_confidence=0.3` has never dropped anything. Across 40 decisions and 228
items, zero were trimmed: `trim_frac=0.1` of a 6-item job floors to 0, and most
jobs carry fewer than 10 items. So in practice this has been a **weighted mean
with no trimming and no dropping**. The confidence *weighting* is real — the
weights genuinely differ. The two robustness mechanisms are the ones that have
never engaged. They are untuned defaults wearing the appearance of tuned ones,
and they cannot be tuned until real weights produce a real score and confidence
distribution.

### Score bands

Bands apply to the **aggregated** score, never a raw per-item score.

Bands are a **total partition** — routing has no undefined input.

| Score | Band | Class | Extended retention | Review flag |
|---|---|---|---|---|
| `< 20` | `likely_authentic` | authentic | – | – |
| `20–40` | `leaning_authentic` | authentic | – | – (auto-clears) |
| `40–60` | `uncertain` | uncertain | – | ✓ normal |
| `60–80` | `leaning_manipulated` | manipulated | – | ✓ **low** |
| `> 80` | `likely_manipulated` | manipulated | ✓ | ✓ normal |
| no score | `undetermined` | undetermined | – | ✓ normal |

`20–40` auto-clearing is deliberate: being wrong there means a probably-real item
gets deleted on schedule. `60–80` is the riskier gap — it sits next to the `>80`
threshold whose calibration isn't trusted enough to show as a raw percentage, so
it gets the same DB-flag-plus-alert treatment as `40–60` at low urgency. It still
deletes on the normal schedule; the flag is the record that it happened. It is
never a silent pass-through.

---

## Retention

- **Tier 1** — raw media and face crops are deleted when inference completes.
- **Tier 2** — a `>80` result opens an **extended retention window**: a fixed
  30-day timer that protects **the media that drove the score** — the face crops
  (or spectrogram chunks) that survived confidence-dropping and trimming, copied
  to `cold/<job_id>/` *before* the Tier 1 delete runs. The **full raw source is
  still deleted** for every band. Preserving only the job row would leave the
  `>80` branch with nothing a dispute could actually use.
- The window **auto-expires, including mid-dispute**. It is **not a legal hold**
  and must never be called one. It still needs legal sign-off before being relied
  on for a real dispute.

### The hold flag gates every delete path

Three code paths can delete retained media. All three read `retention_hold`
first, unconditionally:

| Path | Trigger |
|---|---|
| `delete_media_for_job()` | inference completion |
| `sweep_undeleted()` | crash recovery — a worker that died between verdict and delete |
| `expire_extended_retention()` | the 30-day timer running out |

The crash-recovery sweeper matters because without it "deleted on completion"
quietly stops being true after any router crash. The expiry sweeper matters
because a fixed timer nothing enforces silently becomes "retained forever".

`tests/test_retention_hold_gate.py` covers each path **twice** — held (media
survives) and unheld (media goes). Testing only the held case would pass against
a delete path that was simply broken.

## API

| Method | Path | Notes |
|---|---|---|
| `POST` | `/v1/jobs` | Rate-limited. Returns a presigned **POST policy** (url + form fields), not a PUT URL; media never transits the API. |
| `POST` | `/v1/jobs/{id}/uploaded` | Client calls after the upload POST succeeds. Idempotent. |
| `GET` | `/v1/jobs/{id}` | Result + status. **Polling fallback** after a dropped socket. |
| `WS` | `/v1/jobs/{id}/ws` | Live status. Sends current state on connect so a reconnecting client never waits on a transition it missed. |
| `GET` | `/healthz` | |

Every result carries advisories: scores are manipulable by adversarial
perturbation (no pre-classifier — Tier 3, not built), and whether a placeholder
model produced the score.

### Why the upload grant is a POST policy, not a PUT URL

This is load-bearing, not a style choice. A presigned **PUT** signs the URL but
not the body length, so an issued grant is an **unbounded write** to the bucket.
A **POST policy** can carry a `content-length-range` condition that object
storage itself enforces.

These bytes go straight from the browser to storage and never traverse the
gateway, so ingress rate limiting never sees them. That makes the policy
condition the *only* place `DF_MAX_UPLOAD_BYTES` can be enforced at all — the
`size_bytes` check on `POST /v1/jobs` is a courtesy 413 on the client's own
word, and a client that lies simply skips it.

`measured: yes` — `scripts/smoke_compose.py` mints a grant with a 1 KiB bound,
then shows storage accepting a body under it and rejecting one over it with 400,
writing nothing. `tests/test_presign_policy.py` proves the code still asks for
the condition; only the live probe proves storage honours it.

## Layout

```
migrations/          ordered SQL, 001-005; 001 creates jobs + hold flag
scripts/             migrate.py, run_local.py, mutate.py,
                     check_docs_current.py, and the live probes:
                     smoke_compose.py, verify_queue.py,
                     verify_retention.py, verify_attribution.py
src/df/
  aggregation.py     confidence-weighted trimmed mean
  bands.py           score-band routing
  retention.py       TTL delete + hold-flag check + sweeper
  rollup.py          multi-face worst-case rollup
  ratelimit.py       Redis token bucket (atomic Lua)
  queue.py           Redis Streams (default) or lists + retry limit + DLQ
  jobstatus.py       status key + pub/sub + persistence assertion
  storage.py         presigned uploads, deletes, in-memory backend
  inference/         detector interface, stub, EfficientNet, calibration
  pipelines/         video / image / audio + media extraction
  gateway/app.py     FastAPI
  workers/           cpu_preprocess, gpu_inference, router, retention_sweeper
tests/
```

## Not built — deliberately

- **Adversarial-input pre-classifier.** Scores are manipulable. Documented, not
  hidden.
- **Human review dashboard.** Substitute: `review_flags` table + Slack/email.
- **AV scanning.** Substitute: the CPU worker runs on an `internal: true`
  network with no route off the host, `read_only`, `cap_drop: ALL`,
  `no-new-privileges`, non-root, with pid and memory caps. It is the process
  that parses untrusted media, so this shipped with it rather than later.
- **Calibration.** Temperature scaling is wired but **not yet fitted** — both
  temperatures are `T=1.0`, i.e. uncalibrated. Launch snapshot only when fitted;
  never "production-validated".
