# CPU preprocessing worker.
#
# This image decodes untrusted, attacker-supplied media. There is no AV scanning
# in front of it (Tier 3, deferred), so the container itself is the control:
# non-root, no shell utilities beyond what OpenCV needs, and run with
# read_only + cap_drop ALL + an internal-only network (see docker-compose.yml).
FROM python:3.11-slim

# The container runs read_only with only /tmp writable. numba (pulled in by
# librosa) and joblib both try to write caches at import time and will hard fail
# on a read-only filesystem, so every cache is pointed at the tmpfs.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app/src \
    OPENCV_VIDEOIO_PRIORITY_MSMF=0 \
    HOME=/tmp \
    XDG_CACHE_HOME=/tmp \
    NUMBA_CACHE_DIR=/tmp \
    MPLCONFIGDIR=/tmp

WORKDIR /app

# libGL/libglib are OpenCV's runtime deps; ffmpeg decodes the video containers.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 ffmpeg \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
 && pip install --no-cache-dir opencv-python-headless==4.10.0.84 librosa==0.10.2.post1 \
 && pip install --no-cache-dir pillow-heif==0.18.0

COPY src/ ./src/

RUN useradd --uid 10002 --no-create-home --shell /usr/sbin/nologin worker
USER 10002

CMD ["python", "-m", "df.workers.cpu_preprocess"]
