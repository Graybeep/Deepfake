# Aggregation / routing / retention. No media decoding, no ML stack -- this
# service only ever sees numbers, so it stays small.
FROM python:3.11-slim

# Runs read_only with only /tmp writable.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app/src \
    HOME=/tmp \
    XDG_CACHE_HOME=/tmp

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/

RUN useradd --uid 10004 --no-create-home --shell /usr/sbin/nologin worker
USER 10004

CMD ["python", "-m", "df.workers.router"]
