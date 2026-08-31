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

`/healthz` binds as soon as the gateway starts, which is *before* the GPU worker
finishes warming. The platform will report healthy while the first inference
would still block for ~11 s. `healthcheckTimeout` is set high so a slow boot is
not killed mid-warm, but if you want strict readiness the gate belongs on the
worker's warm-up completing, not on the port opening.

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

## Before presenting

Everything here idles down: Neon scale-to-zero, Upstash, the platform's own
container, and the model warm-up. **Run one real upload sixty seconds before you
present.** Do not let the first judge be the request that warms the stack.
