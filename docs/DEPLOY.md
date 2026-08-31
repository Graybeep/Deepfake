# Deploying the single-container build

For a demo. The compose topology is the real deployment; this trades it for one
build and one boot. **Read "What this gives up" before using it for anything
public.**

## Environment

| Variable | Value | Why |
|---|---|---|
| `DF_PG_DSN` | Neon connection string | Postgres backs the audit row and the per-face evidence the UI renders |
| `DF_REDIS_URL` | Upstash connection string | Queue, job status, and the rate limiter |
| `DF_S3_ENDPOINT` | `file:///data/media` | Selects `LocalDiskStorage`; no object storage service |
| `DF_LOCAL_STORAGE_ROOT` | `/data/media` | Created at image build time, world-writable |
| `DF_PUBLIC_BASE_URL` | `https://<app>.up.railway.app` | Upload grants are absolute; a relative URL breaks a browser on another origin |
| `DF_CORS_ORIGINS` | `https://<app>.vercel.app,https://<preview>.vercel.app` | **Must** be set or the browser fails preflight |
| `DF_INFERENCE_BACKEND` | `torch` | Real weights; the default is the stub |
| `DF_FACE_WEIGHTS` | `/models/weights/dfdc_b7_ns_seed111.pth` | Baked into the image |
| `PORT` | injected by the platform | The launcher reads it; do not hardcode 8000 |

`DF_AUDIO_WEIGHTS` stays unset — audio has no checkpoint and falls back to the
stub, which is the documented mixed state and fails closed.

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

## Deploy checklist

Run in this order. Do not proceed past a failure.

```bash
railway login
railway init            # or `railway link` for an existing project
railway up              # builds docker/deploy.Dockerfile, uploads ~254MB of context
railway domain          # get the public URL
```

Then set the variables from the table above in the Railway dashboard, and
redeploy. `DF_PUBLIC_BASE_URL` needs the URL from `railway domain`, so it cannot
be set before the first deploy.

**Expect a slow first deploy.** The build context carries ~254MB of weights (a
build-time fetch was tried and reverted — see `docker/deploy.Dockerfile`), and
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
