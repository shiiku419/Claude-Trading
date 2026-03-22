# ---------------------------------------------------------------------------
# Stage 1: dependency resolver / builder
# ---------------------------------------------------------------------------
FROM python:3.12-slim-bookworm AS builder

# Pull uv from its official image so we don't depend on PyPI during the build.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Copy only the files uv needs to resolve and install dependencies so that
# Docker's layer cache is reused when application code changes but deps don't.
COPY pyproject.toml uv.lock ./

# Install production dependencies into /app/.venv; skip editable installs and
# dev extras so the venv stays small.
RUN uv sync --frozen --no-dev --no-editable

# ---------------------------------------------------------------------------
# Stage 2: lean runtime image
# ---------------------------------------------------------------------------
FROM python:3.12-slim-bookworm

# Bring across the pre-built virtualenv from the builder stage.
COPY --from=builder /app/.venv /app/.venv

# Prepend the venv's bin directory so every python/pip invocation uses it.
ENV PATH="/app/.venv/bin:$PATH"

WORKDIR /app

# Copy application source after the venv layer so code changes don't
# invalidate the dependency cache.
COPY . .

# Run as an unprivileged user; no home directory needed for a headless bot.
RUN useradd --no-create-home --shell /bin/false trader

USER trader

# Default entrypoint uses the paper-trading configuration; override CMD in
# docker-compose or at runtime to switch profiles.
CMD ["python", "main.py", "config/paper.yaml"]
