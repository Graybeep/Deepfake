# Demo notes

Everything here was measured against the deployed service, not locally.
`https://deepfake-detection-production-049c.up.railway.app/app`

## What to say about the scores

> **The pipeline separates clean images from resampled or spliced ones. It does
> not separate manipulation from compression.**

That sentence is chosen because it survives the table below. Two weaker versions
do not:

- ~~"Treat the scores as an ordering, not probabilities."~~ **False, by our own
  numbers.** The highest score in the table (69.53) belongs to an *authentic*
  photo — a screenshot. The actual manipulation scored lower (62.08). The
  ordering is wrong, so any claim resting on it dies the moment someone reads
  the table.
- ~~"It separates clean images from degraded ones."~~ **Over-general.** The
  blurred image is degraded and scored 2.72, right down with the clean ones.
  Blur does not trigger it; resampling and JPEG recompression do.

Calibration is unfitted (`temperature.v1:unfitted`), so these are raw model
outputs, not probabilities. A "62" is not "62% likely fake".

## The numbers

| case | verdict | score | time |
|---|---|---|---|
| clean portrait | likely_authentic | 0.54 | 2.1 s |
| poor lighting | likely_authentic | 0.37 | 2.0 s |
| face at an angle | likely_authentic | 0.65 | 2.2 s |
| blurred | likely_authentic | 2.72 | 2.1 s |
| group photo, 6 faces | likely_authentic | 1.34 | 5.5 s |
| spliced face | leaning_manipulated | 62.08 | 3.6 s |
| **screenshot of a face** | **leaning_manipulated** | **69.53** | 2.1 s |

Clean and blurred cluster at 0.3–2.7. Recompressed and spliced cluster at 62–70.
Nothing lands in between, which is why the separation claim holds and the
ordering claim does not.

The spliced image is a feathered-ellipse composite of two public-domain
portraits — **not** a GAN deepfake. It is a different artifact class from the
model's training data. It shows the pipeline responds to manipulation; it is not
an accuracy claim.

## Demo order: show the screenshot case yourself

After their own selfie, the next thing anyone reaches for is an image off the
web — downscaled, recompressed, the same artifact profile as the screenshot
case. Two of the first three uploads can land in that bucket.

So **run it deliberately, before anyone asks.** Naming the limitation and
showing the per-face confidence behind it reads as rigor. Getting caught by it
live reads as a broken model. Identical result, opposite impression, and the
only cost is the order you click things in.

The line: *"compression artifacts look like manipulation artifacts to this
model. That's a known limitation of an uncalibrated research checkpoint, and
it's why the response shows per-face detection confidence instead of just a
verdict."*

## Suggested sequence

1. **Clean portrait** — 0.54, `likely_authentic`. Point at the per-face table:
   score, detection confidence, pixel size.
2. **Screenshot** — 69.53, `leaning_manipulated`. Name the limitation first.
3. **Spliced face** — 62.08. The manipulation case.
4. **Group photo** — 6 faces, one verdict via worst-case rollup, 5.5 s.
5. `/healthz` — worker liveness, if anyone asks how you know it is up.

## Things that are true and worth saying

- Every result carries advisories: adversarial manipulability, research
  checkpoint, uncalibrated score. They are derived from what was recorded on the
  job row, and an unrecognised or missing value produces the *strongest* caveat,
  not none.
- A face detected below 40% of the strongest detection in the frame is not
  scored, and the response says so. That gate exists because a non-face region
  once scored 55.79 and set the verdict for a clean portrait.
- Media is deleted when inference completes (`media_deleted: true` in every
  result above). The per-face evidence survives in Postgres; the image does not.
- Audio falls back to the stub even with real face weights loaded, and reports
  `placeholder`. That mixed state is deliberate and fails closed.

## Rate limiting: fixed and verified on the deployed service

It was effectively disabled earlier tonight — the limiter keyed on the socket
peer, which behind Railway is a rotating proxy pool, so 45 rapid requests spread
across 20+ near-fresh buckets and never limited anything.

Now keyed on the real client via `DF_TRUSTED_PROXY_HOPS=2`. `measured: yes`
2026-09-01 against the live URL:

- 45 rapid `POST /v1/jobs` → **first 429 at request 42**, `Retry-After: 2`.
  42 is the arithmetic: capacity 30, plus ~12 refilled at 0.5/s during the burst.
- A client sending `X-Forwarded-For: 1.2.3.4, 5.6.7.8, 9.9.9.9` is **ignored** —
  identity still resolves to its real address, so nobody can pick their own
  bucket.

**Why 2 and not 1:** Railway's `X-Forwarded-For` is `<client>, <edge>` — the edge
appends its *own* address, and that address rotates too. One hop bucketed on the
rotating edge; two hops reaches the client. `GET /v1/whoami` shows the resolved
identity and the headers behind it, which is how this was determined rather than
guessed — the first guess was wrong and failed silently.

## Do not

- **Do not redeploy.** A container restart puts a rollout between a judge and
  their result, and strands any in-flight job at `status=inference`.
- Do not quote a score as a probability or a percentage.
- Do not call it "production-validated" or "adversarially robust" — neither is
  built, and both are on the forbidden list in CLAUDE.md.

## Before presenting

Load `/app`, upload one image, watch the result render. That warms the model and
the connections so the first judge is not the request that pays for it.
