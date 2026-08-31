# Single-container deploy image. Everything in one process tree.
#
# The compose stack runs five services on three images. Railway would mean five
# deploys of a ~2GB image at midnight before a morning deadline, so this trades
# the topology for one build and one boot.
#
# WHAT THAT TRADE COSTS, stated rather than absorbed: the CPU-preprocess worker
# loses its network isolation. In compose it runs on an `internal: true` network
# with no route off the host, read_only, cap_drop ALL, non-root -- and CLAUDE.md
# is explicit that this isolation IS the AV-scanning substitute, shipped in the
# same phase as the worker because a parser of untrusted media without it is an
# open compromise window. Here it shares a container with the gateway. That is
# acceptable for a time-boxed demo with known inputs and is NOT acceptable for
# anything public. Deploy the compose topology for that.
#
# WEIGHTS ARE BAKED IN, deliberately. The compose setup bind-mounts ./models,
# which cannot work on a platform with no host filesystem: the worker would hit
# FileNotFoundError at boot on every deploy. Baking costs ~254MB of image and
# saves a ~250MB fetch inside the platform's health-check window on every
# restart.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app/src \
    HOME=/tmp \
    XDG_CACHE_HOME=/tmp \
    TORCH_HOME=/tmp

WORKDIR /app

# ffmpeg/libGL are what OpenCV's video decode needs; slim has neither.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

# Dependency layers first and separately, so an application code change does not
# reinstall torch. This is the slow half of the build; keep it above COPY src/.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
 && pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
      torch==2.5.1 torchvision==0.20.1 \
 && pip install --no-cache-dir \
      timm==1.0.11 \
      opencv-python-headless==4.10.0.84 \
      librosa==0.10.2.post1 \
      pillow-heif==0.18.0

# Weights before source: they change far less often than code, so this layer
# survives every application rebuild.
COPY models/weights/dfdc_b7_ns_seed111.pth /models/weights/dfdc_b7_ns_seed111.pth

COPY migrations/ ./migrations/
COPY scripts/ ./scripts/
COPY src/ ./src/

# Uploads and derived crops live here. Created at build time so the directory
# exists and is writable before the first upload -- a permission error on first
# write is indistinguishable from a broken API from the outside.
RUN mkdir -p /data/media && chmod 777 /data/media

EXPOSE 8000
CMD ["python", "-m", "df.deploy"]
