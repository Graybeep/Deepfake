# GPU inference worker -- the cost driver, and the pod HPA scales on post-MVP.
#
# Built on python-slim with the CPU torch wheel by default so `docker compose up`
# works on any machine. Swap the base to nvidia/cuda:12.4-runtime and drop the
# --index-url line for real GPU inference.
FROM python:3.11-slim

# Runs read_only with only /tmp writable; torch writes kernel and hub caches.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app/src \
    HOME=/tmp \
    XDG_CACHE_HOME=/tmp \
    TORCH_HOME=/tmp

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
 && pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
      torch==2.5.1 torchvision==0.20.1 \
 && pip install --no-cache-dir timm==1.0.11 opencv-python-headless==4.10.0.84

COPY src/ ./src/

# Weights are NOT baked into the image: they are large, they change independently
# of code, and the job row records their sha256. Mount them and point
# DF_FACE_WEIGHTS / DF_AUDIO_WEIGHTS at the mount.
VOLUME ["/models"]

RUN useradd --uid 10003 --no-create-home --shell /usr/sbin/nologin worker
USER 10003

CMD ["python", "-m", "df.workers.gpu_inference"]
