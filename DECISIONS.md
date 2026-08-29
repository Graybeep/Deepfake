# Decisions and open questions

Judgement calls made while scaffolding, and the things that need a human answer.
CLAUDE.md is the contract; this file records where the contract was silent and
what was assumed in its place.

---

## Resolved by CLAUDE.md

### 1. The score band gaps — RESOLVED

CLAUDE.md now specifies that bands are a total partition and how each gap
behaves. Implemented as:

- **20–40** (`leaning_authentic`) auto-clears like `<20`. No flag, no window.
  Being wrong here means a probably-real item gets deleted on schedule.
- **60–80** (`leaning_manipulated`) gets the same DB-flag-plus-alert pattern as
  40–60, at **low urgency**. It sits next to the `>80` threshold whose
  calibration is not trusted enough to show as a raw percentage, so it must
  never pass silently into normal deletion. It still deletes on the normal
  schedule and does **not** open the extended retention window — the flag is the
  record that it happened.

`ReviewUrgency` (`none` / `low` / `normal`) carries this through to
`review_flags.urgency` and into the alert, so on-call can tell a low-urgency
60–80 flag apart from a 40–60 or `>80` one.

Where: `src/df/bands.py`, `tests/test_bands.py`.

### 2. Extended retention window scope — RESOLVED, reversing my earlier assumption

I had assumed record-only. CLAUDE.md is explicit that the window protects **the
flagged media itself — specifically the face crop(s) that drove the score, not
the full raw source** — because the row and per-item scores are already retained
for every job regardless of band, so a record-only window would leave the `>80`
branch with nothing a dispute could use.

Implemented as:

- On a `>80` verdict the router copies the driving crops to `cold/<job_id>/`
  **before** the Tier 1 delete runs. Ordering is load-bearing — reversed, the
  window would open over crops deleted a moment earlier.
- "Driving" = the items that survived confidence-dropping and trimming
  (`AggregationResult.used_items`), resolved to storage keys via
  `job_items.object_key`. Keys are read from the DB rather than rebuilt by
  naming convention, so a layout change can't silently preserve nothing.
- The raw source is still deleted on completion for every band. Tier 1 is
  unchanged.
- A third delete path now exists — cold-storage expiry — and it carries the same
  unconditional hold-flag check as the other two.

**Audio:** CLAUDE.md says "face crop(s)". For audio there are no faces, so the
faithful generalisation is the driving **spectrogram chunks**. Flagged here
rather than assumed silently.

**Still needs legal sign-off** before being relied on for an actual dispute.
CLAUDE.md says so directly; this is the engineering default, not the final word.

Where: `src/df/retention.py`, `src/df/workers/router.py`,
`tests/test_retention_hold_gate.py`.

---

## Answered 2026-08-29 — and neither was answered as asked

Both of these were put up as a choice between three reductions. Both came back
with the same objection, which was the right one: **picking between three lossy
reductions bakes the loss into the contract**, and every downstream consumer
then inherits it. That is more expensive to undo later than getting the choice
wrong, because the choice can be revisited and a discarded input cannot.

So neither is locked. What changed instead is that the inputs to the decision
are now recorded and published, so the policy can be fitted later without a
schema change or a contract break.

**The blocker underneath both is the same, and it is not a missing decision.**
There is no labelled validation data in this repository, and the only scores
this system has ever produced come from `_hash_score` — a SHA-256 of the input
bytes mapped onto 0–100, explicitly uncorrelated with whether the input was
manipulated. So there is no score distribution to fit a per-size-bucket
threshold against, and none to derive a video floor from either. Both questions
reduce to "we need weights and a labelled held-out set", which is the same
dependency the calibration work has been blocked on all along.

### 3. Worst-case multi-face rollup — kept as the aggregator, demoted as the label

CLAUDE.md flags this one itself: >1 face rolls up to worst-case severity, "a
default, not fixed — confirm before anything downstream assumes otherwise."

**Implemented as:** highest manipulation score across faces wins; the rolled-up
item keeps the confidence of the face that *set* the score, so a worst case
driven by a marginal detection is weighted down rather than counted at full
strength. Individual face scores are still stored in `job_items`.

**Live consequence:** a crowd scene where one background face scores high makes
the whole video `manipulated`.

**Resolved by making the reduction non-exclusive rather than by replacing it.**
Worst-case stays as the aggregator — it is the conservative default and the
false-positive cost is a review flag, not a deletion. What changed is that it is
no longer the only artifact: `face_evidence` on a completed video/image result
reports `faces_total`, how many carry geometry, and the top faces by score with
frame index, confidence and pixel size. "3 of 47 faces, the highest is 38×41px
in frame 412" is triageable; "manipulated" is not. The flag was never the
problem. The uninformative flag was.

**What is deliberately NOT built: a per-face threshold.** The better rule is a
bar that varies with face size and quality, fitted so the false-positive rate is
flat across size buckets rather than flat across scores — plus an N-correction
(Šidák or similar) on the per-face bar, which removes most crowd false positives
while preserving worst-case semantics. None of it can be fitted here. Adding an
invented per-face constant to make the field look finished would bake a *second*
unvalidated number into the contract while claiming to fix the first.

**The prerequisite that was missing, and is now fixed.** Face size was not
merely un-thresholded, it was *unmeasurable*: `FaceCrop.bbox` has been populated
by the extractor since the first commit and was dropped on the floor in the CPU
worker, never entering the manifest or `job_items`. Across every job this system
has ever run there is no record of how big any face was, so even with labels the
bucketing feature would have had to be regenerated from scratch. Migration 006
adds `job_items.face_w` / `face_h` and the workers now carry them through.

Absolute pixels only. Relative-to-frame area is the better feature and remains
unavailable, because source frame dimensions are not recorded anywhere either —
flagged rather than approximated, since a threshold fitted against a fabricated
feature is worse than no threshold.

**Still open, and now cheap to answer once data exists:** the per-bucket
false-positive numbers. Nothing downstream has to change to adopt them.

Where: `src/df/rollup.py` (`face_evidence`), `migrations/006`,
`src/df/workers/cpu_preprocess.py`, `src/df/gateway/app.py`,
`tests/test_coverage_and_evidence.py`.

### 4. Minimum items before a verdict — split by modality, and made non-load-bearing

**Was:** fewer than 3 usable items ⇒ `undetermined`, for every modality.

**Now, and why the gate was the wrong lever:** the objection to a k=1 verdict is
not that the score is wrong, it is that a one-frame verdict and a fifty-frame
verdict are indistinguishable in the response. The fix for that is to *mark the
difference*, not to withhold the verdict. So every result now carries
`items_total` and `coverage` alongside `item_count`, at every k. Once coverage
is published the floor stops being the only protection a reader has, and a
consumer can set its own bar instead of inheriting one this codebase cannot
defend.

**The floor is now per modality: 1 for image, 3 for video and audio.** This is a
real distinction, not a concession: an image is a complete observation of its
subject, while a video frame is one sample from a distribution over frames. A
rule about sampling variance should not apply to something that was not sampled.

The image path already behaved this way — `aggregate_identity` never consulted
the floor at all — but it recorded `min_items_for_score: 3` on every image job.
So the audit row asserted a parameter that had not governed the result, which is
the row claiming a rule the code did not run. That was the part that was
actually wrong, and it is fixed.

**3 (video/audio) is an unvalidated placeholder and is now labelled as one** in
`AggregationParams.min_items_for_score` itself rather than only in a document.
It was picked as the smallest number that is more than a couple. Deriving it
means measuring score variance at k=1,2,3,5,10 on validation clips and taking
the point where it flattens; there are no validation clips and no meaningful
scores. An honest constant is cheaper to revisit than one that has acquired the
authority of having been written down.

**What would change this again:** if usable-item filtering turns out to be
strict enough that sub-threshold media is genuinely unscoreable rather than
merely noisy — a 2-frame verdict near-random rather than just wide — then
abstaining is right and the fix belongs upstream in what counts as usable, not
in this constant.

Where: `src/df/aggregation.py` (`for_media`, `coverage`), `migrations/006`,
`src/df/workers/router.py`, `src/df/gateway/app.py`,
`tests/test_coverage_and_evidence.py`.

---

## Decisions taken (not blocking, but worth knowing)

**Stub inference backend is the default.** No EfficientNet weights are vendored.
`DF_INFERENCE_BACKEND=stub` runs a deterministic hash-based scorer so the whole
pipeline can be built and tested; it reports `is_real_detector=False` and
declares `validation=placeholder`. Switch to `torch` once weights exist.

**CORRECTED.** This used to say the API "attaches a placeholder advisory to any
result carrying a stub model id" — i.e. it matched the substring `stub` against
`model_version_id`. That failed **open**. Loading a real checkpoint changes the
id to `face-efficientnet_b4-<hash>`, the substring vanishes, and every caveat
disappears silently — at exactly the moment scores start looking plausible
enough to be believed.

The advisory is now derived from `ModelVersion.validation`, a required field
(`placeholder` / `research-checkpoint` / `production-validated`) written onto
each item row and rolled up by the router from the rows that actually produced
the score. NULL or any unrecognised value produces the **strongest** caveat, not
none. Do not reintroduce a check keyed on how a model is named.

Where: `src/df/gateway/app.py` (`_validation_advisories`),
`tests/test_validation_advisory.py`, and `ADVISORY_MUTATIONS` in
`scripts/mutate.py`, which keeps the fail-open version as a regression case.

**Confidence weighting uses detection/alignment confidence, not model
confidence.** The model's own output says nothing about whether it got a clean
face to look at. The GPU worker deliberately carries the preprocessing
confidence through instead of the model's.

**Result is committed before media is deleted.** If the delete fails the sweeper
retries it; if the result write had failed after deletion, the inputs needed to
reproduce it would already be gone.

**Redis Streams, not lists — REVERSED 2026-08-16.** This originally read "Redis
lists, not Streams; Streams consumer groups are week 2+ per CLAUDE.md". Streams
is now the default (`DF_QUEUE_BACKEND=streams`) and lists are kept only as a
rollback path; both write the same `q:<topic>:dead` list, so there is one place
to look either way.

What the reversal buys: any live consumer can reclaim a message whose worker
stopped without acking it, once idle past `DF_QUEUE_RECLAIM_MS` (XAUTOCLAIM). So
recovery no longer requires a worker restart, and a topic can have more than one
consumer — which is what makes the GPU worker horizontally scalable. Under
lists, a message stranded while all workers stayed up was invisible until the
retention sweeper caught the job hours later.

`measured: yes` — `scripts/verify_queue.py`, which must run inside a container:
Redis is on the internal network, and publishing a host port to test it would
weaken the isolation being tested.

**Plain SQL migrations, not Alembic.** The schema is small and the retention
logic needs to be readable by anyone reviewing it. Revisit if it starts churning.

**Rate-limit identity is API key, else client IP.** Behind a proxy this must read
a *trusted* forwarded-for header or every request buckets to the proxy — wire
that with the ingress config; `identity_of()` carries the note.

---

## Tier 3 — deferred, and visible in the product

Per CLAUDE.md these are marked, not half-built:

- **Adversarial-input pre-classifier — not built.** Scores are manipulable by
  adversarial perturbation. Every API result carries an advisory saying so.
- **Human-in-the-loop dashboard — not built.** Substitute is the `review_flags`
  table plus a Slack/email alert (`src/df/notify.py`). The DB row is written
  first and a failed notification never loses the flag.
- **AV scanning — not built.** Substitute is container isolation, and per
  CLAUDE.md it ships in the same phase as the CPU worker: `internal: true`
  network with no route off the host, `read_only`, `cap_drop: ALL`,
  `no-new-privileges`, non-root, pid and memory caps.

## 5. A labelled held-out set for calibration — assessed 2026-08-29

Calibration, the score bands, the per-face size threshold and the video
minimum-items floor are four open questions with **one** shared dependency: a
labelled held-out set. This is the survey. Every entry carries its source, per
CLAUDE.md, and all of it was read on 2026-08-29.

**The structural finding first: there is no ungated, permissively-licensed,
provenance-documented face-manipulation dataset.** Every option that is usable
requires a human to accept an agreement. That is not a gap in the search; it is
the state of the field, and it means this step cannot be automated away.

### The constraint that eliminates most candidates is not the licence

A calibration set must be **held out from the model's training data**. Ours is
the DFDC-winner B7, trained on DFDC. Fitting a temperature on data the model
memorised produces a confident-looking, well-calibrated-looking number that is
wrong in deployment — and unlike a licence problem, nothing downstream would
ever surface it. So "clean licence, undocumented provenance" is not a
half-acceptable compromise here, it is disqualifying: if the provenance is
unknown, overlap with DFDC cannot be ruled out.

This is the same unverifiability argument CLAUDE.md already used to reject the
Apache-2.0 HuggingFace checkpoints. It is stronger for a calibration set, because
a leaked calibration set fails silently in the direction of overconfidence.

The second eliminator is **task match**. The model detects face manipulation in
video frames. A temperature fitted on text-to-image synthetic pictures is fitted
for a distribution the model was never meant to score.

### Recommended: Deepfake-Eval-2024 — with one blocking question

- Licence **CC-BY-SA-4.0**, which permits commercial use with attribution and
  share-alike. That alone makes it the only serious candidate found.
- 44h video, 56.5h audio, 1,975 images. 88 sources, 52 languages, **manually
  labelled**, covering faceswap, lipsync and diffusion.
- Collected **in the wild in 2024**, so overlap with DFDC (2019–20 paid actors)
  is not merely unlikely, it is chronologically impossible. It is also far closer
  to deployment distribution than any academic set.
- Purpose-built as an evaluation benchmark, which is exactly what a held-out
  calibration set should be.

**BLOCKER, and it must be resolved before use, not after.** The terms as
summarised on the dataset card include *"use only for evaluation purposes, not
training"*. Temperature scaling fits a parameter on the data — a one-parameter
logistic regression. Whether post-hoc calibration counts as "training" under
those terms is genuinely ambiguous, and it is not this repository's call to
decide. Ask the authors directly; it is one email with a definitive answer, and
the answer determines whether this option exists at all.

`measured: no (source)`, and note the limitation: the full terms of use sit
**behind the access gate**, so the clause above is from the public dataset card,
not from the agreement text. That is precisely the sort of second-hand reading
this file exists to flag.
<https://huggingface.co/datasets/nuriachandra/Deepfake-Eval-2024> ·
<https://github.com/nuriachandra/Deepfake-Eval-2024> ·
<https://arxiv.org/abs/2503.02857>

Access gate: HuggingFace, requiring an institutional or company email plus
evidence of work related to deepfake detection. This project is that work.

### Fallback: the DFDC public test set

Adds **no new licence category**. CLAUDE.md already accepted the DFDC terms
knowingly for the weights; the test set is the same licensor and the same
agreement, so it introduces no risk that has not already been taken and written
down. It is genuinely held out from the training split the model was fitted on.

The cost is distributional, not legal: DFDC is paid actors under controlled
lighting and framing. A temperature fitted there is a launch snapshot for *that*
distribution, and real uploads do not look like it. Defensible, and it must be
recorded as what it is.

Access: accept the Kaggle competition rules.
`measured: no (source)` <https://www.kaggle.com/c/deepfake-detection-challenge>
· <https://ai.meta.com/datasets/dfdc/>

### Rejected, with reasons and sources

- **Celeb-DF v2** — non-commercial research only, with an explicit undertaking
  not to "exploit any portion of the videos or any derived data for any
  purpose". A fitted temperature is derived data. Same blocker as FF++.
  `measured: no (source)`
  <https://github.com/yuezunli/celeb-deepfakeforensics>
- **DF40**, **DeepfakeBench**, **SynthForensics** — all CC BY-NC-4.0.
  Non-commercial, as CLAUDE.md already recorded for DeepfakeBench.
  <https://github.com/YZY-stack/DF40> · <https://github.com/SCLBD/DeepfakeBench>
- **prithivMLmods HuggingFace sets** (`Deepfake-vs-Real-60K` and siblings) —
  Apache-2.0, and **zero provenance**. The card names two "curated subsets" and
  never says where the images came from, what generated the fakes, or which
  datasets they derive from. Overlap with DFDC is unknowable, which is fatal per
  the section above. Clean licence, unverifiable data — the exact trade CLAUDE.md
  already refused for checkpoints.
  <https://huggingface.co/datasets/prithivMLmods/Deepfake-vs-Real-60K>
- **OpenFake** — fails twice. Subsets from proprietary generators are
  non-commercial under provider non-compete clauses, and the content is
  AI-generated imagery broadly (Midjourney, Imagen, Stable Diffusion), not face
  manipulation. Wrong task.
  <https://huggingface.co/datasets/ComplexDataLab/OpenFake>

### Building our own was considered and rejected

Swapping faces ourselves over permissively-licensed real images is technically
possible and produces a worthless calibration: every fake would carry one
generator's artifacts, so the fitted temperature would describe our own tooling
rather than the threat. That is a fabricated calibration wearing better clothes
than a hand-picked constant, not an alternative to a real held-out set.

### What is true whichever is chosen

A temperature is only valid for the distribution it was fitted on. This is
inherent to a launch snapshot, is already stated in CLAUDE.md, and does not go
away by picking a better dataset. `fitted_on` records which set it was, and it
should be read every time the number is.

## Claims this codebase must not make

Repeated from CLAUDE.md because it is easy to drift on:

- Not "adversarial robustness" — not built.
- Not "legal hold" — it is a fixed-timer **extended retention window** that
  auto-expires, including mid-dispute.
- Not "production-validated calibration" — launch snapshot only, and the
  temperatures are not yet fitted (both currently `T=1.0`, i.e. uncalibrated).
- Not "BIPA/GDPR compliant" — partial deletion plus a partial audit trail is
  meaningfully better than nothing, but compliance is a legal determination this
  codebase does not get to assert.
