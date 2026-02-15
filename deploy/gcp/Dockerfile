FROM python:3.11-slim

WORKDIR /app

# Install uv
RUN pip install uv

# Copy project files
COPY pyproject.toml uv.lock ./
COPY src/ ./src/

# Install dependencies
RUN uv sync --frozen

# Expose port
EXPOSE 8080

# Run with uvicorn
CMD ["uv", "run", "uvicorn", "src.examples.cloud.gcp_cloud_run_agent:app", "--host", "0.0.0.0", "--port", "8080"]
