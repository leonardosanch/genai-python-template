# Dockerfile
# Multi-stage build for production GenAI API

# --- Builder Stage ---
FROM python:3.12-slim-bookworm AS builder

LABEL org.opencontainers.image.source="https://github.com/your-org/genai-python-template"
LABEL org.opencontainers.image.description="GenAI Python Template API"
LABEL org.opencontainers.image.licenses="MIT"

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Set working directory
WORKDIR /app

# Install dependencies using uv
# Copy only dependency files first to cache the layer
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --no-dev

# Copy the rest of the application
COPY . .
RUN uv sync --frozen --no-dev

# --- Runner Stage ---
FROM python:3.12-slim-bookworm AS runner

LABEL org.opencontainers.image.source="https://github.com/your-org/genai-python-template"
LABEL org.opencontainers.image.description="GenAI Python Template API"

WORKDIR /app

# Create a non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Copy virtual environment from builder
COPY --from=builder /app/.venv /app/.venv

# Copy application code
COPY --from=builder /app/src /app/src
# Copy config files
COPY --from=builder /app/pyproject.toml /app/pyproject.toml

# Set environment variables
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app"
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Change ownership to non-root user
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8000

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Command to run the application
CMD ["uvicorn", "src.interfaces.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers"]
