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

1. **Clean portrait** — 0.54, `likely_authentic`. Read the headline and the
   scale: "no signs" at 0 and "strong signs" at 100, with the score pinned. The
   per-face table is deliberately absent here — with one face it only restated
   the headline. Open **Technical details** to show the three advisories, the
   trust level and the weights hash sitting behind the plain answer.
2. **Screenshot** — 69.53, `leaning_manipulated`. Name the limitation first.
3. **Spliced face** — 62.08. The manipulation case.
4. **Group photo** — 6 faces, one verdict via worst-case rollup, 5.5 s. This
   is where the per-face table appears, and where a skipped detection reads as
   "skipped — 31% as strong" rather than a raw confidence number.
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

## If asked about audio or video

Neither is offered in the UI. The file input says "Photos only" and the landing
page describes two pipelines, not three. That is deliberate and recent -- both
were advertised until 2026-09-01.

**Audio**, if asked: "The audio pipeline is built end to end -- chunking,
spectrograms, aggregation, routing, retention -- but the scorer is a stub, so
it's not in the product. The checkpoint we use is a face model, and pointing it
at spectrograms would produce confident numbers from something that has never
seen audio."

That is verifiable if anyone wants it: `measured: yes`, a 21 s WAV through the
API gave 7 log-mel spectrogram chunks, all 7 scored, coverage 1.0, `uncertain`
at 40.2638, media deleted. Every stage real, only the scorer a SHA-256 of the
bytes -- and the advisory says exactly that.

**Video**, if asked: "The video pipeline is real and produces correct results,
but it is not reliable on our current host, so it is not in the product. The
frame sampler materialises every sampled frame in memory at once, and this
container is already holding a 66-million-parameter model, so whether a clip
completes depends on how much memory the container has left rather than on the
clip. I would rather say that than ship a button that works when you are not
watching."

`measured: yes` 2026-09-01, and the last row is the point:

| test | outcome |
|---|---|
| 4 s @ 720p, 8 frames | clean, all 8 scored, coverage 1.0, 15.4 s |
| 20 s @ 720p, 40 frames | `gpu-inference exited with -9` — OOM |
| 10 s @ 1080p, 12 frames | 107 s, container down and back twice |
| **same 20 s clip, 3 runs, 8-frame cap** | **201 s never finished / 53 s crashed-then-recovered / 4.2 s clean** |

Three identical requests, three different outcomes. Lowering
`DF_VIDEO_MAX_FRAMES` from 300 to 12 to 8 did not make it deterministic, because
frame count is not the variable — accumulated container memory is. That is why
"short clips work" would have been the wrong thing to claim.

**The image path is unaffected**, which is the check that matters: three
consecutive uploads at 3.3 / 3.7 / 3.8 s, identical scores, container never
dropped. Video jobs cannot be created from the UI at all
(`media_type: 'image'` is hardcoded), so none of the above is reachable during
a demo.

**Both answers are stronger than the feature would have been.** Naming a limit
with numbers behind it reads as knowing your own system; a video button that
kills the service mid-demo does not.

## Known, not a problem tomorrow: slow jobs get delivered twice

`measured: yes` — the first audio job after a deploy was reclaimed:

    WARNING df.queue reclaimed topic=inference job=8269d576... delivery=2
    -- previous consumer stopped without acking

librosa JIT-compiles on first use, the message went unacked past the idle
timeout, and another consumer reclaimed it. It completed on the retry, so the
reclaim path did its job — but the mechanism is **generic, not an audio bug**:
any job slower than `DF_QUEUE_RECLAIM_MS` is processed twice. Audio only
surfaced it.

Tomorrow's margin is fine — the slowest measured demo case is the 6-face group
photo at 5.5 s, far inside the timeout, and the face model is warmed at boot.
Not worth a deploy tonight. Worth knowing it exists.

## Full-resolution phone photos: fixed 2026-09-01, verified

A 12.2 MP upload used to sit in `preprocessing` for 85+ seconds and then take
the container down (`gpu-inference exited with -9`, SIGKILL), returning
`undetermined`. Haar ran on the full-resolution image: +143.9 MB peak RSS for a
single extract, next to a resident B7. That is what a phone camera produces, so
it was the demo path.

`DF_DETECT_MAX_SIDE=1600` now bounds the image Haar sees; crops still come from
the full-resolution original. `measured: yes` on the deployed service after the
fix:

| upload | before | after |
|---|---|---|
| 12.2 MP | 85 s, container SIGKILLed, `undetermined` | 8.8 s, **container stayed up** |
| 8.4 MP real photo | (never got a verdict) | `likely_authentic` 1.3357, 1 face 516x516 |
| 1.2 MP baseline | 0.7898, 523x523 | 0.7898, 523x523 — **unchanged** |

The small-image path is untouched, which is the check that matters for the score
table above: it is below the cap, so the numbers in it still stand.

The 8.4 MP run is worth showing if anyone asks about the confidence gate. Three
detections, two dropped, each recorded with why:

    516x516 conf 0.573  <- best in frame, scored
    265x265 conf 0.179  relative_to_best 0.312  discarded (< 0.4)
    135x135 conf 0.085  relative_to_best 0.149  discarded (< 0.4)

`DF_QUEUE_RECLAIM_MS` was also lowered 120000 -> 45000. The UI gives up at 90 s
and the queue could not reclaim an orphaned job until 120 s, so recovery was
structurally unable to reach the person watching — a job that did complete came
back 30 s after the page had already said "timed out". Both are env vars, so
either can be retuned on the running deployment without a rebuild.

## Do not

- **Do not redeploy from here on.** A container restart puts a rollout between a
  judge and their result, and strands any in-flight job at `status=inference`.
  The deploy above was taken deliberately, before the demo, because the phone
  path was broken; that reason is now spent.
- Do not quote a score as a probability or a percentage.
- Do not call it "production-validated" or "adversarially robust" — neither is
  built, and both are on the forbidden list in CLAUDE.md.

## Before presenting

Load `/app`, upload one image, watch the result render. That warms the model and
the connections so the first judge is not the request that pays for it.
