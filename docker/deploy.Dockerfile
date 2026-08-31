# Single-container deploy image. Everything in one process tree.
#
# The compose stack runs five services on three images. A platform deploy would
# mean five builds and five boots to get wrong before a deadline, so this trades
# the topology for one build and one boot.
#
# WHAT THAT TRADE COSTS, stated rather than absorbed: the CPU-preprocess worker
# loses its network isolation. In compose it runs on an `internal: true` network
# with no route off the host, read_only, cap_drop ALL, non-root -- and CLAUDE.md
# is explicit that this isolation IS the AV-scanning substitute, shipped in the
# same phase as the worker because a parser of untrusted media without it is an
# open compromise window. Here it shares a container with the gateway. That is
# acceptable for a time-boxed demo over known inputs and is NOT acceptable for
# public traffic. Deploy the compose topology for that.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app/src \
    HOME=/tmp \
    XDG_CACHE_HOME=/tmp \
    TORCH_HOME=/tmp

WORKDIR /app

# libGL/glib are what OpenCV needs at import time; slim has neither.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 curl ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# HEAVY dependency layers first, pinned inline rather than via a file.
# Deliberate: these have no build-context dependency, so editing
# requirements.txt cannot invalidate them. The first version of this file copied
# requirements.txt above the torch install, so adding one small pure-Python
# package re-downloaded 175MB of torch. 350s -> 109s after reordering. Nothing
# above this line should change often.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
      torch==2.5.1 torchvision==0.20.1 \
 && pip install --no-cache-dir \
      timm==1.0.11 \
      opencv-python-headless==4.10.0.84 \
      librosa==0.10.2.post1 \
      pillow-heif==0.18.0

# Weights are COPIED from the build context, in a layer above COPY src/ so
# application rebuilds never touch them.
#
# A build-time `curl` from the public release URL was tried and reverted: it
# would keep 254MB out of the context (which a platform re-uploads on every
# deploy), but curl failed with exit 35, an SSL connect error, inside this
# machine's build network. Possibly TLS interception locally; it may well work
# on a platform builder. Not worth diagnosing against a deadline when COPY is
# verified working, so the cost is accepted: expect ~254MB of context upload
# per deploy. If you want it back, the pinned values were:
#   URL    https://github.com/selimsef/dfdc_deepfake_challenge/releases/download/0.0.1/final_111_DeepFakeClassifier_tf_efficientnet_b7_ns_0_36
#   SHA256 9db77ab9318863e2f8ab287c8eb83c2232584b82dc2fb41f1d614ddd7900cccb
# and the sha256 check is worth keeping if you do -- a truncated asset would
# otherwise produce a model whose id, derived from that hash, silently
# disagrees with every job row already stored.
COPY models/weights/dfdc_b7_ns_seed111.pth /models/weights/dfdc_b7_ns_seed111.pth
# Light, fast-changing deps last.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY migrations/ ./migrations/
COPY scripts/ ./scripts/
COPY src/ ./src/

# Uploads and derived crops live here. Created at build time so the directory
# exists and is writable before the first upload -- a permission error on first
# write is indistinguishable from a broken API from the outside.
RUN mkdir -p /data/media && chmod 777 /data/media

EXPOSE 8000
CMD ["python", "-m", "df.deploy"]
