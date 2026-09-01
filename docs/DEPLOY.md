# Deploying the single-container build

For a demo. The compose topology is the real deployment; this trades it for one
build and one boot. **Read "What this gives up" before using it for anything
public.**

## Environment

| Variable | Value | Why |
|---|---|---|
| `DF_PG_DSN` | `${{Postgres.DATABASE_URL}}` | Postgres backs the audit row and the per-face evidence the UI renders |
| `DF_REDIS_URL` | `${{Redis.REDIS_URL}}` | Queue, job status, and the rate limiter |
| `DF_S3_ENDPOINT` | `file:///data/media` | Selects `LocalDiskStorage`; no object storage service |
| `DF_LOCAL_STORAGE_ROOT` | `/data/media` | Created at image build time, world-writable |
| `DF_PUBLIC_BASE_URL` | `https://<app>.up.railway.app` | Upload grants are absolute; a relative URL breaks a browser on another origin |
| `DF_CORS_ORIGINS` | `https://<app>.vercel.app,https://<preview>.vercel.app` | **Must** be set or the browser fails preflight |
| `DF_INFERENCE_BACKEND` | `torch` | Real weights; the default is the stub |
| `DF_FACE_WEIGHTS` | `/models/weights/dfdc_b7_ns_seed111.pth` | Baked into the image |
| `PORT` | injected by the platform | The launcher reads it; do not hardcode 8000 |
| `DF_TRUSTED_PROXY_HOPS` | `2` on Railway | Proxies between client and app. **Determine it, do not guess** — see below |
| `DF_DETECT_MAX_SIDE` | `1600` | Longest side Haar detects on. Crops stay native. 0 disables |
| `DF_MIN_ITEM_CONFIDENCE` | `0.0` | Absolute aggregation floor. 0 = none, deliberately: 0.30 refused verdicts on ordinary portraits |
| `DF_DETECT_FALLBACK` | `true` | Retry detection on contrast-enhanced/alternate cascades when the primary finds nothing. 20/23 hard cases -> 23/23 |
| `DF_MAX_FACES_SCORED` | `5` | Faces scored per item. A 24-face photo crashed the container at 8; 5 is 3/3 clean. Capped faces are reported, not hidden |
| `DF_INFERENCE_BATCH_SIZE` | `1` | B7 forwards per batch. 2 gave 1/3 on a many-face photo, 1 gave 2/3; combined with the cap, 3/3 |
| `DF_VIDEO_MAX_FRAMES` | `8` | Frames sampled per video. The sampler holds them ALL in memory at once, so this bounds damage — it does not make video reliable |
| `DF_QUEUE_RECLAIM_MS` | `45000` | Must stay BELOW the UI's 90 s deadline, or an orphaned job recovers after the page has already given up |

`DF_AUDIO_WEIGHTS` stays unset — audio has no checkpoint and falls back to the
stub, which is the documented mixed state and fails closed.

## Deploys are manual

`git push` does NOT deploy. The Railway service has no GitHub source
(`railway status --json` reports `source.repo = null`), so a build happens only
when someone runs:

    railway up --detach

That is worth knowing in both directions: committing during a demo is safe, and
a fix is not live merely because it is pushed. Setting an environment variable
DOES restart the container -- faster than a build, but still a restart.

## Determining `DF_TRUSTED_PROXY_HOPS`

Get this wrong and rate limiting silently does nothing — no error, no warning,
just 201s forever. It happened here twice: first at the default 0 (socket peer,
which is a rotating proxy), then at a guessed 1.

`GET /v1/whoami` exists to settle it. It returns the resolved identity, the
socket peer, and the forwarding headers actually received:

```json
{"identity":"ip:61.12.82.214","socket_peer":"100.64.0.3","trusted_proxy_hops":2,
 "forwarding_headers":{"x-forwarded-for":"61.12.82.214, 152.233.15.123", ...}}
```

Count the entries in `x-forwarded-for` that were added by infrastructure and set
hops so `identity` lands on the real client. On Railway the header is
`<client>, <edge>` — the edge appends its **own** address — so the answer is 2,
not the 1 you would guess from "there is one proxy".

Verify with a burst: 45 rapid `POST /v1/jobs` should produce 429s.
Then verify spoofing fails: send your own `X-Forwarded-For` and confirm
`identity` is unchanged. Taking the *leftmost* entry would let any caller choose
its own bucket, which is worse than no limiting because it looks like protection.

## Sizing

`measured: yes` on the development machine:

| | |
|---|---|
| Peak RSS, 8 × 800px batch | **1541 MB** |
| After model load, idle | 836 MB |
| Cold start (load + first forward pass) | 10.68 s + 0.48 s |
| Steady state | 0.32 s/face at 8 threads |

**≥2 GB RAM.** Below that the container OOMs during model load, which in a
platform log looks identical to a crash. **Do not take 0.5 vCPU** — the 0.32 s
figure is on 8 threads, so a fractional core is a 4–16× cut and puts a single
face at 1.5–3 s.

**Images only.** Video sampling is 12 frames × faces each; on CPU that is
minutes per clip, not seconds.

## Health check

`/healthz` reports the **workers**, not just the gateway, and that distinction is
the difference between a visible failure and the worst one this service can have.

The job flow is asynchronous, so nothing blocks on the model warming and a fast
200 is correct for the gateway itself. But that same property means a dead
inference worker is invisible from the gateway: the platform sees a healthy
service, the gateway accepts the upload, the job lands in Redis, and nothing ever
picks it up. No error, no failed status, no red anything — a client watching a
spinner forever, looking exactly like a model thinking hard.

Two layers, covering different failures. Both verified against a running
container:

| Layer | Catches | Speed | Verified |
|---|---|---|---|
| `df.deploy` supervisor | a worker **process exiting** (e.g. OOM-kill) | ~2 s | `SIGKILL` → `gpu-inference exited with -9; shutting down the container` |
| heartbeat + `/healthz` | a worker **alive but wedged**, or Redis lost | ~45 s | `SIGSTOP` → `503 degraded:["inference"]`; `SIGCONT` → back to 200 |

Workers refresh a Redis key with a 45 s TTL on every poll (~5 s). `/healthz`
returns 503 when a required worker's key has expired, and the platform's restart
policy takes it from there.

**A 150 s boot grace** tolerates missing heartbeats at startup — workers take
seconds to appear and the inference worker loads a 254 MB model first. Without it
the first health check fails and the deploy is rolled back before anything had a
chance to start. Degraded workers are still *named* in the body during the grace,
so a slow boot can be told from a stuck one.

The retention sweeper is deliberately **excluded** from the required set: it runs
on a timer, so its absence delays cleanup rather than stranding a job, and it
must not be able to roll back a deploy.

### Memory, per process

`measured: yes` inside the running container — and this answers whether the model
is loaded more than once. It is not:

| Process | RSS |
|---|---|
| `gpu_inference` | **685 MB** ← the only one holding B7 |
| uvicorn gateway | 60 MB |
| retention_sweeper | 42 MB |
| router | 42 MB |
| cpu_preprocess | 39 MB |
| **idle total** | **~880 MB** |

A 1–2 face upload peaks around 1.1–1.3 GB, so a 2 GB tier holds. 4 GB is
comfortable. An 8 × 800px batch peaked at 1541 MB in the worker alone, so do not
size on the idle figure.

## What this gives up

- **The CPU worker's network isolation.** In compose it runs on an
  `internal: true` network with no route off the host, `read_only`,
  `cap_drop: ALL`, non-root. CLAUDE.md is explicit that this isolation *is* the
  AV-scanning substitute, shipped with the worker because a parser of untrusted
  media without it is an open compromise window. Here it shares a container with
  the gateway. **Fine for a time-boxed demo over known inputs; not fine for
  public traffic.**
- **Per-service scaling.** The GPU worker is the cost driver and the thing worth
  scaling alone. Here it scales with everything or not at all.
- **The presigned-upload property.** With S3/MinIO the browser POSTs straight to
  storage and the bytes never traverse the gateway — which is why the
  `content-length-range` policy condition is the only place the size cap can be
  enforced there. On local disk the grant points back at `POST /v1/uploads` and
  the cap is enforced by that endpoint, streaming and aborting mid-read. A
  different mechanism, not a weaker one, but not the same one.

## Ephemeral storage

The container filesystem is wiped on redeploy while Postgres rows survive, so
job rows will reference media that no longer exists.

That is survivable **only because the evidence display reads from Postgres** —
scores, per-face confidences, discarded detections — and never re-reads the
uploaded image. Tier 1 deletes the media on completion anyway, and
`media_deleted` already models exactly this. If a UI is ever changed to
re-render the uploaded file, a redeploy mid-demo will empty the screen.

## Managed Postgres and Redis, not Neon and Upstash

`railway add --database postgres` and `--database redis`. Same private network,
no external signups, and no scale-to-zero cold start stacking on top of the model
warm-up. Reference them as `${{Postgres.DATABASE_URL}}` / `${{Redis.REDIS_URL}}`
rather than pasting connection strings, so a rotated credential does not silently
break the app.

Note the internal Redis URL is `redis://`, **not** `rediss://` — private
networking, no TLS termination to negotiate. The `rediss://` requirement applies
to Upstash, not to this.

**Setting variables from Git Bash mangles absolute paths.** `DF_FACE_WEIGHTS=/models/...`
was stored as `C:/Program Files/Git/models/...` and the worker died at boot with
a `FileNotFoundError` naming that path. Prefix with `MSYS_NO_PATHCONV=1`:

```bash
MSYS_NO_PATHCONV=1 railway variables --set "DF_FACE_WEIGHTS=/models/weights/dfdc_b7_ns_seed111.pth"
```

## Deploy checklist

Run in this order. Do not proceed past a failure.

```bash
railway login
railway init            # or `railway link` for an existing project
railway up              # builds Dockerfile, uploads ~254MB of context
railway domain          # get the public URL
```

Then set the variables from the table above in the Railway dashboard, and
redeploy. `DF_PUBLIC_BASE_URL` needs the URL from `railway domain`, so it cannot
be set before the first deploy.

**Expect a slow first deploy.** The build context carries ~254MB of weights (a
build-time fetch was tried and reverted — see `Dockerfile`), and
the image is 3.3GB.

### Things that behave differently than on localhost

- **`$PORT`** is injected. The launcher reads it and binds `0.0.0.0`; a hardcoded
  port or a `127.0.0.1` bind gives a container that runs fine and a public URL
  that 502s, which is indistinguishable from a crash. *(Verified: reads `PORT`,
  binds `0.0.0.0`.)*
- **Health check vs warm-up.** `/healthz` answers as soon as uvicorn binds, which
  is ~6s before the model finishes warming. That is deliberate — the queue
  absorbs a job submitted in that window — but it means healthy does not mean
  warm. `healthcheckTimeout` is 300s so a slow boot is not killed mid-load.
- **Neon needs SSL, and has two endpoints.** Use the **direct** endpoint for
  migrations and the **pooled** one for the app. Five processes each opening a
  pool against the direct endpoint is how a free tier runs out of connections.
- **Upstash needs `rediss://`**, not `redis://`, or TLS fails. *(Verified: every
  client goes through `redis.Redis.from_url`, which handles `rediss://`.)*
- **Migrations do not race.** `df.deploy` runs them once in the parent process
  before starting any worker. *(Verified.)*

### DO NOT REMOVE THE THREAD PINNING

`df.deploy._pin_threads()` sets `OMP_NUM_THREADS` before spawning any worker.
That line looks like cargo cult. It is not. Measured on Railway, before and
after, on the same image:

| | before | after |
|---|---|---|
| model load | 41.82 s | 3.01 s |
| first inference | **225.04 s** | **0.21 s** |
| end to end | 222.2 s | 4.0 s |

**`os.cpu_count()` reports the HOST's cores inside a container, not the cgroup
quota.** torch sized its thread pool for cores it did not have, and those threads
contended over a 2-core allocation. A single image took nearly four minutes,
which for a demo is indistinguishable from broken — and nothing in the logs says
"too many threads", it just runs slowly and looks like a big model on a small
machine.

Three details that matter if you touch this:

- It must be set **before torch is imported**. The OpenMP runtime reads
  `OMP_NUM_THREADS` at import; setting it afterwards in the child does nothing.
  That is why it lives in the launcher and not in the worker.
- `cpu_quota()` returns `None` on a real host, where there is no quota, and
  `usable_threads()` then falls back to `os.cpu_count()` — correct outside a
  container.
- Existing values are respected (`setdefault`), so deliberate operator tuning
  is not overruled.

**This is not specific to torch or to Railway.** Any thread-pool sizing from
`os.cpu_count()` inside a container has the same bug. The rest of this tree was
checked; nothing else sizes a pool that way.

### Fallback, if the platform is sick

The container is verified working locally, which makes a tunnel a free fallback
rather than an architecture:

```bash
cloudflared tunnel --url http://localhost:8080
```

Point the UI at the tunnel URL and add it to `DF_CORS_ORIGINS`. Primary stays the
platform; this is for the morning when something is down and there is no time.

## Before presenting

Everything here idles down: Neon scale-to-zero, Upstash, the platform's own
container, and the model warm-up. **Run one real upload sixty seconds before you
present.** Do not let the first judge be the request that warms the stack.
