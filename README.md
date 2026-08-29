# Deepfake detection service

Three ingest pipelines (video, image, audio) converging on shared aggregation,
score-band routing, and retention logic.

**Read `CLAUDE.md` before changing anything touching deletion, retention, or
claims about what is built.** `DECISIONS.md` records the judgement calls made
during scaffolding. Its two open questions were answered on 2026-08-29 — both
by recording the inputs to the decision rather than by locking a reduction.

> **The default detector is a placeholder; real weights are opt-in.** With the
> default `DF_INFERENCE_BACKEND=stub` the scorer is a deterministic hash, not a
> trained model: `is_real_detector=false`, `validation=placeholder`.
>
> Real weights are wired and verified — the DFDC-winner EfficientNet-B7, loaded
> via `docker-compose.weights.yml`. **That does not make the output validated.**
> It reports `validation=research-checkpoint`, calibration is still `T=1.0`, and
> the score bands have never been measured against this model. A real model's
> output on unvalidated calibration is *more* dangerous than obvious nonsense,
> because it looks like a finding.
>
> The API derives its advisory from the declared level — **never** from how the
> model is named. A name-based check fails open: load a real checkpoint, the id
> stops containing `stub`, and every caveat disappears at exactly the moment
> scores start looking believable. That is no longer hypothetical — the id is
> now `face-tf_efficientnet_b7_ns-9db77ab93188`, and the caveat correctly
> escalated instead of vanishing.

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

241 tests, no infrastructure required. The suite covers the two pipeline rules
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

## Running on real weights

The stub is the default so `docker compose up` works on a machine with no
weights on disk. To run the real detector:

```bash
mkdir -p models/weights
BASE=https://github.com/selimsef/dfdc_deepfake_challenge/releases/download/0.0.1
curl -L -o models/weights/dfdc_b7_ns_seed111.pth "$BASE/final_111_DeepFakeClassifier_tf_efficientnet_b7_ns_0_36"

docker compose -f docker-compose.yml -f docker-compose.weights.yml up -d --build
```

254 MiB, no account needed. `models/` is gitignored — weights are never
committed, and the job row records their sha256 instead.

| | stub (default) | weights overlay |
|---|---|---|
| face model | `face-stub-v0` | `face-tf_efficientnet_b7_ns-<sha12>` |
| face validation | `placeholder` | `research-checkpoint` |
| audio model | `audio-stub-v0` | `audio-stub-v0` (unchanged) |
| audio validation | `placeholder` | `placeholder` |

**Audio does not switch.** The chosen checkpoint is a *face* model; there is no
audio checkpoint under that decision, so `get_audio_model()` falls back to the
stub and logs it rather than loading a face architecture for spectrograms. That
mixed state is deliberate and it fails closed — the stub carries the strongest
caveat.

**The overlay proves plumbing and provenance, not accuracy.** The CPU worker
stays on the stub extractor, so the crops are synthetic; a real model scoring
synthetic input returns a real number with no detection meaning. Treat scores
from this configuration as evidence that the wiring works, never as findings.

**Licence.** The upstream code is MIT. The weights are trained on Meta's DFDC
dataset, whose terms are unpublished and whose flow-through to derived weights
is unsettled. Do not ship commercially without legal review.

### Running the whole pipeline for real

The weights overlay switches only the GPU worker, which leaves the CPU worker
emitting synthetic crops — a real model scoring fake input. To make both halves
real, add the third overlay:

```bash
docker compose -f docker-compose.yml -f docker-compose.weights.yml                -f docker-compose.realpipeline.yml up -d --build
```

Frames then come from `OpenCVFrameSampler` and faces from the Haar cascade.
**Expect more `undetermined` results**: Haar misses profile and small faces, and
0 faces routes to `undetermined` by design rather than guessing.

This still is not accuracy. The detector is a research checkpoint, the
temperature is unfitted, and Haar has always been documented as a placeholder
for RetinaFace/SCRFD. It is real components on a real decode path — a
prerequisite for accuracy work, not a substitute for it.

**The face crop is emitted at native resolution.** Geometry belongs to the
detector, which does the isotropic resize and zero-padded centring that match
upstream preprocessing. The extractor used to resize crops to the model input
size, which both crashed (the constant became an int) and — when it worked —
destroyed the aspect ratio *before* the careful resize, silently making it a
no-op.

**Detection confidence is uncalibrated.** Haar's reject level is an unbounded
internal cascade score, not a probability; it is squashed to 0–1 by dividing by
10, which is arbitrary and monotone and nothing more. It becomes the aggregation
weight, so it is load-bearing. Every confidence distribution measured in this
repo so far came from the *stub* extractor, so "the weights genuinely differ"
has never been observed on the real path.

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

### Coverage, reported on every verdict

`item_count` is what produced the score; `items_total` is what was extracted
before confidence drops and trimming; `coverage` is the ratio. All three are on
the job row and in the API.

Without it a verdict off 1 usable frame of 50 and one off 50 of 50 are the same
response, which is what forced the minimum-items floor to be the only protection
a reader had: with no way to say *scored, but barely*, the only way to protect
the consumer is to refuse to answer. With coverage published, a consumer can set
its own bar.

The floor is now per modality — **1 for image, 3 for video and audio**. An image
is a complete observation of its subject; a video frame is one sample from a
distribution over frames, and a rule about sampling variance should not apply to
something that was not sampled. The image path already behaved this way by
accident (`aggregate_identity` never consulted the floor), but recorded
`min_items_for_score: 3` on every image job — a parameter the code had not
applied to that result.

**3 is an unvalidated placeholder** for video and audio. Deriving it means
measuring score variance at k=1,2,3,5,10 on validation clips and taking the
point where it flattens. There are no validation clips, and every score this
system has produced comes from a hash of the input bytes.

### Per-face evidence, not just a label

A `>80` crowd scene and a `>80` close-up are the same word. `face_evidence` on a
completed video/image result reports `faces_total`, how many carry recorded
geometry, and the top faces by score with frame index, confidence and pixel
size — so *"the highest is 38×41px in frame 412 of 47 faces"* is one read away.

It does **not** decide anything. There is deliberately no per-face threshold and
no `flagged` count: a per-face bar needs a false-positive rate measured per size
bucket, which needs labelled validation data. Inventing one to make the field
look complete would bake a second unvalidated constant into the contract while
claiming to fix the first.

Face geometry is new in migration 006 and this is why it was missing: `bbox` had
been populated by the extractor since the first commit and **dropped on the
floor in the CPU worker**, so across every job this system has ever run there is
no record of how big any face was. It was not un-thresholded, it was
unmeasurable. `face_w`/`face_h` are absolute pixels; relative-to-frame area is
the better bucketing feature and is still unavailable, because frame dimensions
are not recorded either.

### Calibration: the machinery exists, the fit does not

Temperature scaling turns a raw logit into a calibrated probability. It is fitted
by minimising negative log-likelihood **against ground-truth labels** on a
held-out set — labels are not an input that can be approximated, and without them
there is no loss surface and nothing to minimise.

Real weights landed 2026-08-29 and removed one of the two blockers. **The other
one stands: there is no labelled held-out set here**, and the public evaluation
sets already assessed for licensing (FF++, DeepfakeBench, DFDC) are gated,
non-commercial, or both.

So both temperatures are still `T=1.0`, which is the identity. What exists is the
fitter, ready to run the day labels do:

```bash
python scripts/fit_calibration.py --scores held_out.jsonl --model face
```

It is verified against synthetic data whose true temperature is known by
construction — distort a calibrated set by a factor of k and the fit must recover
k. It does, at k = 0.5, 1.0, 1.8 and 3.0, and on a 5000-item demo it recovered
2.435 against a true 2.4 while cutting ECE from 0.101 to 0.010. **That tests the
optimiser and nothing else.** It says nothing about whether any real detector is
calibrated.

**Do not invent a temperature.** A fabricated T is worse than 1.0: 1.0 is visibly
the identity and reads as "nothing applied", while a plausible 1.7 reads as
measured and nothing in the system could contradict it.

#### The runbook, once the data is on disk

The chosen set is the **DFDC validation split** — 4,000 clips, 50/50, with
`metadata.json`, and using 214 subjects **none of which appear in the training
set**. Not the Kaggle `test_videos` folder, which is unlabelled by design.
Getting it needs an AWS account and accepted terms at the dfdc.ai portal; that
step needs a person.

```bash
# 1. score the labelled set with the REAL pipeline, emitting {logit, label}
docker compose -f docker-compose.yml -f docker-compose.weights.yml     run --rm -v /path/to/dfdc_validation:/data:ro gpu-inference     python scripts/extract_logits.py --dir /data --out /tmp/heldout.jsonl

# 2. fit
python scripts/fit_calibration.py --scores /tmp/heldout.jsonl --model face     --describe "DFDC validation split, 4000 clips"

# 3. paste the printed Temperature(...) into src/df/inference/calibration.py
```

`extract_logits.py` reuses the production sampler, extractor and detector rather
than reimplementing them: a temperature is only valid for the distribution it was
fitted on, and preprocessing is part of that distribution.

Two limits to carry with any temperature this produces. It fits **per face crop**,
which is the level `Temperature.apply` acts on — but the bands apply to the
*aggregated* score, and a weighted trimmed mean of calibrated probabilities is not
itself guaranteed calibrated. And DFDC is paid actors under controlled lighting,
so the fit describes that distribution, not real uploads.

`calibration` is now on the job row, the item rows, and the API. It has to be
per-item and rolled up like `model_version_id`, because `model_version_id` is
keyed on the *weights hash* — refit the temperature and every score changes while
the id stays identical. This column is the only thing that tells those results
apart, and a job whose rows carry two calibrations is refused rather than
averaged across two scales.

Every result from a real model now also carries an **UNCALIBRATED SCORE**
advisory: the number is a raw model output, not a percentage likelihood, and the
score bands were not drawn against that scale.

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
migrations/          ordered SQL, 001-007; 001 creates jobs + hold flag
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
