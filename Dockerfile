# STAGE 1: Builder
FROM python:3.11-slim-bookworm AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    unixodbc-dev \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.4.0 /uv /uvx /bin/

WORKDIR /build

COPY pyproject.toml ./

RUN uv venv /opt/venv && \
    uv pip install --python /opt/venv -r pyproject.toml

# STAGE 2: Runtime
FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    ACCEPT_EULA=Y \
    PATH="/opt/venv/bin:$PATH"

RUN apt-get update && apt-get install -y --no-install-recommends curl gnupg2 apt-transport-https && \
    curl -fsSL https://packages.microsoft.com/keys/microsoft.asc | gpg --dearmor -o /usr/share/keyrings/microsoft-prod.gpg && \
    curl -fsSL https://packages.microsoft.com/config/debian/12/prod.list > /etc/apt/sources.list.d/mssql-release.list && \
    apt-get update && \
    ACCEPT_EULA=Y apt-get install -y --no-install-recommends \
    msodbcsql18 \
    unixodbc \
    && apt-get purge -y --auto-remove gnupg2 apt-transport-https \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --shell /bin/bash worker

WORKDIR /ai-foundry

COPY --from=builder /opt/venv /opt/venv

COPY ./app ./app
COPY ./ocr-model ./ocr-model
COPY ./Classification-model ./Classification-model

RUN chown -R worker:worker /ai-foundry

USER worker

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/ || exit 1

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
