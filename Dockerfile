# Multi-stage: a build layer for wheels, a slim runtime that ships no compiler.
FROM python:3.12-slim AS builder

ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /build

RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential libxml2-dev libxslt1-dev \
 && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY creditlens ./creditlens
RUN python -m pip install --upgrade pip build \
 && python -m pip wheel --wheel-dir /wheels .

# ---------------------------------------------------------------- runtime ---
FROM python:3.12-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    CREDITLENS_DATA_DIR=/app/data \
    CREDITLENS_LOG_FORMAT=json

RUN apt-get update \
 && apt-get install -y --no-install-recommends libxml2 libxslt1.1 curl \
 && rm -rf /var/lib/apt/lists/* \
 && useradd --create-home --uid 10001 creditlens

WORKDIR /app
COPY --from=builder /wheels /wheels
RUN python -m pip install --no-index --find-links=/wheels creditlens && rm -rf /wheels

COPY web ./web
COPY creditlens/eval/datasets ./creditlens/eval/datasets
RUN mkdir -p /app/data && chown -R creditlens:creditlens /app

USER creditlens
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
  CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["python", "-m", "uvicorn", "creditlens.api.app:app", \
     "--host", "0.0.0.0", "--port", "8000"]
