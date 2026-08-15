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

## Open questions — answer these before building further on them

### 3. Worst-case multi-face rollup — confirm the default

CLAUDE.md flags this one itself: >1 face rolls up to worst-case severity, "a
default, not fixed — confirm before anything downstream assumes otherwise."

**Implemented as:** highest manipulation score across faces wins; the rolled-up
item keeps the confidence of the face that *set* the score, so a worst case
driven by a marginal detection is weighted down rather than counted at full
strength. Individual face scores are still stored in `job_items`.

**Live consequence:** a crowd scene where one background face scores high makes
the whole video `manipulated`. If that is the wrong trade, the alternative is a
largest-face or highest-confidence-face rule.

Where: `src/df/rollup.py`, `src/df/pipelines/video.py`, `src/df/pipelines/image.py`.

### 4. Minimum items before a verdict

**Assumed:** fewer than 3 usable items ⇒ `undetermined` rather than a score.
A verdict off one or two frames is noise. This makes very short clips and
heavily-occluded video come back undetermined, which will show up as a support
question. Tunable via `AggregationParams.min_items_for_score`.

---

## Decisions taken (not blocking, but worth knowing)

**Stub inference backend is the default.** No EfficientNet weights are vendored.
`DF_INFERENCE_BACKEND=stub` runs a deterministic hash-based scorer so the whole
pipeline can be built and tested; it reports `is_real_detector=False` and its
`model_version_id` contains `stub`, so no stored result can be mistaken for a
real detection. The API attaches a placeholder advisory to any result carrying a
stub model id. Switch to `torch` once weights exist.

**Confidence weighting uses detection/alignment confidence, not model
confidence.** The model's own output says nothing about whether it got a clean
face to look at. The GPU worker deliberately carries the preprocessing
confidence through instead of the model's.

**Result is committed before media is deleted.** If the delete fails the sweeper
retries it; if the result write had failed after deletion, the inputs needed to
reproduce it would already be gone.

**Redis lists, not Streams.** Streams consumer groups are week 2+ per CLAUDE.md.
Lists with a processing-list handoff give at-least-once delivery, which is enough
while workers are idempotent.

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
