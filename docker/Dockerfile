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

# Export requirements from uv.lock then install with system Python into /usr/local
# This avoids venv symlink issues when building on macOS for Linux containers
RUN uv export --frozen --no-dev > requirements.txt && \
    uv pip install --system -r requirements.txt

# ---------------------------------------------------------------------------
# Stage 2: lean runtime image
# ---------------------------------------------------------------------------
FROM python:3.12-slim-bookworm

# Copy installed packages from builder stage
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

WORKDIR /app

# Copy application source after the dependency layer so code changes don't
# invalidate the dependency cache.
COPY . .

# Run as an unprivileged user; no home directory needed for a headless bot.
RUN useradd --no-create-home --shell /bin/false trader

USER trader

# Default entrypoint uses the paper-trading configuration; override CMD in
# docker-compose or at runtime to switch profiles.
CMD ["python", "main.py", "config/paper.yaml"]
