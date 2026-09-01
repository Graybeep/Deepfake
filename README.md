# Deepfake detection service

Three ingest pipelines (video, image, audio) converging on shared aggregation,
score-band routing, and retention logic. **Only the image pipeline is offered as
a product** — see the table below before reading further.

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

## What actually works

| pipeline | in the product? | in the backend? | blocker |
|---|---|---|---|
| **Image** | **yes** — `/app`, phone-tested | complete | none; this is the working path |
| **Video** | no | **complete and verified** | container memory, not code |
| **Audio** | no | **complete except the scorer** | no audio checkpoint chosen |

The UI accepts photographs only (`accept="image/*"`, and the client sends
`media_type: 'image'` — it cannot request anything else). Video and audio are
reachable through the API by posting `media_type: "video"` / `"audio"`, and both
run end to end, but neither is fit to offer. What follows is what each needs.

### Image — the working path

Real DFDC-winner EfficientNet-B7, ~2–3 s end to end, verified on a phone over
cellular. Detection retries on contrast-enhanced and alternate Haar cascades when
the primary finds nothing, which took hard-case detection from 20/23 to 23/23
(`measured: yes`, glasses and bad lighting — the two cases originally reported
broken).

Known limits, measured rather than assumed, in `docs/EVALUATION.md`: it produces
**confident false positives on grainy and badly-lit photographs** (an authentic
1879 portrait scores 73, and 90 after recompression), compression pushes scores
up monotonically, and calibration is unfitted so no score is a probability.

### Video — works, but not on 1000 MB

`Ingest → Frame Sample → Face Extract → Align → EfficientNet-B7 → Aggregate →
Router`. `measured: yes`: a 4 s 720p clip yields 8 sampled frames, all 8 scored,
`coverage 1.0`, `likely_authentic` in 15.4 s.

**Why it is not offered.** `measured: yes` on the deployed service, the same 20 s
clip five times: `54.5 s CRASHED / 4.6 s clean / 4.1 s clean / 4.1 s clean /
99.4 s CRASHED`. Identical input, different outcomes, because the binding
constraint is the container's remaining memory. The container has **1000 MB**
(read from the cgroup at boot and logged by `deploy.py`), shared by five
processes one of which holds a 66 M-parameter model — and `cv2.VideoCapture.read()`
alone, retaining nothing, peaks at **123 MB** on a 1080p file against 21 MB for
twenty face extractions. The cost is inside OpenCV's decoder.

**To implement it:**

1. **Give the container more memory.** This is the actual fix and it is a hosting
   setting, not a patch. Everything below is optimisation around a ceiling.
2. Already done and insufficient: frames stream one at a time instead of being
   materialised as a list (peak 205 → 147 MB), and they pass to the extractor as
   decoded arrays instead of a PNG encode/decode round trip (32 % faster, no
   memory change). Three OpenCV levers were tested and none help —
   `CAP_PROP_BUFFERSIZE=1` changes nothing, frame-seeking is slightly worse.
3. If more memory is impossible: shrink the resident model. int8 dynamic
   quantisation takes B7 from ~266 MB to ~67 MB, but it perturbs **every score**,
   so the demo table and `docs/EVALUATION.md` would both need regenerating and
   the accuracy effect measuring. Not a free win.
4. Then enable it in the UI: `accept="image/*,video/*"` and derive `media_type`
   from the picked file's type instead of hardcoding `'image'`
   (`web/index.html`). Two lines, and only worth doing after (1).
5. Verify with **repeated** runs, not one. Runs 2–4 of the five above were all
   clean; a single check would have reported success.

`DF_VIDEO_MAX_FRAMES` (8) and `DF_DETECT_MAX_SIDE` (1600) bound the damage
meanwhile.

### Audio — the pipeline is real, the scorer is not

`Ingest → Chunk → Spectrogram → Scorer → Aggregate → Router`. `measured: yes`: a
21 s WAV yields 7 log-mel spectrogram chunks, all 7 scored, `coverage 1.0`,
banded and the media deleted. librosa loads it, the chunks are real, the routing
is real.

**Why it is not offered.** The scorer is `_hash_score()` in
`src/df/inference/stub.py` — a SHA-256 of the input bytes mapped onto 0–100.
Deterministic, stable for the same file, and uncorrelated with whether the audio
was manipulated. The single checkpoint this project uses is a **face** model;
`get_audio_model()` therefore falls back to the stub and logs it, and
`build_audio_detector()` raises `NotImplementedError` on purpose rather than
guessing an architecture. The stub declares `validation=placeholder`, which
produces the strongest caveat on every audio result — pointing a face model at
spectrograms would have produced confident numbers from something that has never
heard audio.

**To implement it:**

1. **Choose a checkpoint with checkable provenance and a usable licence.** This
   is the real blocker and it is the same one that has calibration stalled: most
   audio anti-spoofing weights are fine-tuned on ASVspoof, whose terms and
   train/eval overlap both need reading. `DECISIONS.md` §5 has the method used
   for the face weights; apply it rather than inventing one.
2. **Decide the input representation first, because it decides the work.** The
   pipeline currently hands the scorer **log-mel spectrogram PNGs**. A
   spectrogram-CNN drops into that seam unchanged. The strong models — AASIST,
   RawNet2, wav2vec2/XLSR fine-tunes — take **raw waveform**, which means
   changing `AudioChunk`, `LibrosaAudioChunker`, the CPU worker's audio branch
   and the queue payload together. Pick the family before writing anything.
3. Implement `build_audio_detector()` in `src/df/inference/efficientnet.py` (or a
   new module) and set `DF_AUDIO_WEIGHTS`. The registry already routes to it and
   already fails closed if it is absent.
4. **Fit its own calibration.** `CLAUDE.md` is explicit that face and audio do
   not share a temperature curve. Until that is done the audio advisory must stay
   at `UNCALIBRATED`.
5. Then enable it in the UI, and expect a mixed advisory state — the API derives
   the caveat per model, so video/image at `research-checkpoint` alongside audio
   at whatever audio earns is already handled without special-casing.

### Where the code is

| | |
|---|---|
| sampling, extraction, chunking | `src/df/pipelines/extract.py` |
| the model registry and fail-closed fallback | `src/df/inference/registry.py` |
| the stub scorer (audio's current scorer) | `src/df/inference/stub.py` |
| the unimplemented audio builder | `src/df/inference/efficientnet.py` |
| per-modality branching | `src/df/workers/cpu_preprocess.py` |
| aggregation, bands, routing | `src/df/aggregation.py`, `src/df/bands.py` |

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

480 tests, no infrastructure required. The suite covers the two pipeline rules
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
10, which is arbitrary and monotone and nothing more.

**It gates rather than reweights, and the gate is RELATIVE.** A detection is
dropped when its confidence is below `DF_DETECTION_CONFIDENCE_RATIO` (default
`0.4`) times the best detection **in the same frame**. Worst-case rollup over the
survivors is unchanged.

Three things are deliberate there.

*Gating, not reweighting*: a non-face region entering the model returns an
arbitrary number, and there is no principled way to combine an arbitrary number
with a real one — averaging it is not better than maxing it, only less alarming.

*Worst-case preserved*: one manipulated face is what makes an image manipulated.
A confidence-weighted mean across faces would drag a swapped face's score toward
the crowd in a group photo, trading a visible false positive for **invisible
false negatives** on precisely the case the tool exists for.

*Relative, not an absolute floor*: these confidences are OpenCV `levelWeights`
from `detectMultiScale3(outputRejectLevels=True)` — unbounded stage-rejection
scores from inside the cascade, squashed into 0–1 by dividing by 10. That is a
monotone transform of an internal score, **not a probability**, so no absolute
threshold is more justified than any other; "which floor is correct" is a
question the number cannot answer. A ratio is invariant to that untrusted scale
and compares detections only against each other, which is the comparison Haar's
weights can actually support.

It also **cannot empty a non-empty detection set** — the best detection is always
ratio 1.0. A lone marginal face (bad light, a turned head, glasses) is kept and
reported with its low confidence rather than gated into `undetermined`. That is a
structural property, not a fallback branch.

**What is gated is recorded.** An earlier version of this gate lived in the
extractor and was removed because an extraction-time drop left no row and no
count. It now writes to the `preprocess.complete` event and is surfaced per-face
by the API, with the ratio and the frame's best confidence alongside — `0.316`
means nothing unless you know it was compared against `0.968`. (`job_items`
cannot hold them: `score` is `NOT NULL`, and they were never scored.)

Measured end to end on a public-domain portrait: Haar returned the real face at
`0.968` (B7 scored it `0.54`, authentic) plus artefacts at `0.316` and `0.075`.
Before the gate, the `0.316` artefact scored `55.79` and worst-case rollup made
the whole image `uncertain`. After: `likely_authentic`, `0.54`, with
`face 1 discarded, conf 0.316, rel 0.326` in the response.

`0.4` is still a chosen number, chosen on failure asymmetry rather than evidence.
The real repair is a detector returning a genuine detection probability —
RetinaFace/SCRFD — not a better constant here.

### Phone uploads

**HEIC works.** A phone camera roll is HEIC by default and OpenCV has no HEIF
codec at all (`measured: yes`: its build lists JPEG, PNG, WEBP only), so
`cv2.imdecode` returned `None`, extraction returned no faces, and the job
completed as `undetermined` — the service told you no face was found in a photo
of your face. `decode_image()` now falls back to `pillow-heif`. Verified end to
end: a real HEIC uploaded as `IMG_0042.HEIC` decodes, the face is found and
scored `0.55`, `likely_authentic`.

**EXIF orientation needs no handling on the OpenCV path.** `cv2.imdecode`
honours it (`measured: yes`: a 300×100 image tagged `orientation=6` decodes as
100×300), so a sideways phone photo arrives upright. The `pillow-heif` branch
does *not* get that for free, so it calls `exif_transpose` explicitly.

**"Could not decode" is reported separately from "no face found."** Those are
different facts and were previously the same response. An undecodable upload now
carries a `MEDIA NOT DECODED` advisory naming the sniffed format and stating
that the result means *not analysed*, not *no face present*.

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

## Deploying (single container)

`docker compose` is the real topology. For a demo platform there is a
single-container image that bakes the weights and runs every process in one
tree — see **[docs/DEPLOY.md](docs/DEPLOY.md)** for the env matrix, measured
sizing, and an explicit list of what the trade gives up (most importantly the CPU
worker's network isolation, which is the AV-scanning substitute).

```bash
docker build -t deepfake-deploy .
```

**Weights are baked in.** The compose stack bind-mounts `./models`, which cannot
work where there is no host filesystem — the worker hits `FileNotFoundError` at
boot on every deploy.

**Storage is local disk** (`DF_S3_ENDPOINT=file:///data/media`), so no object
store is needed when every process shares a filesystem. The upload grant then
points at this service's own `POST /v1/uploads` rather than at S3, and the size
cap is enforced there instead of by a signed policy condition — the client flow
is unchanged either way. Ephemeral: a redeploy wipes it while Postgres rows
survive, which is survivable only because the evidence display reads from
Postgres and never re-reads the image.

**Sizing:** ≥2 GB RAM (peak RSS measured at 1541 MB), ≥1 vCPU, always-on.
Images only — video on CPU is minutes per clip.

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

Video and audio are **not** in this list: both pipelines are built and verified,
they are simply not offered as a product. See "What actually works" above for
what each one needs.


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
