# Deployment

## Docker

### Dockerfile

```dockerfile
FROM python:3.12-slim AS base

# Instalar uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Copiar dependencias primero (cache layer)
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# Copiar código
COPY src/ src/

# Non-root user
RUN useradd -m appuser
USER appuser

EXPOSE 8000
CMD ["uv", "run", "uvicorn", "src.interfaces.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Docker Compose (desarrollo)

```yaml
services:
  app:
    build: .
    ports:
      - "8000:8000"
    env_file: .env
    volumes:
      - ./src:/app/src  # Hot reload
    depends_on:
      - redis
      - qdrant

  qdrant:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
```

---

## Kubernetes

### Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: genai-app
spec:
  replicas: 3
  selector:
    matchLabels:
      app: genai-app
  template:
    metadata:
      labels:
        app: genai-app
    spec:
      containers:
        - name: app
          image: genai-app:latest
          ports:
            - containerPort: 8000
          resources:
            requests:
              memory: "256Mi"
              cpu: "250m"
            limits:
              memory: "512Mi"
              cpu: "500m"
          env:
            - name: OPENAI_API_KEY
              valueFrom:
                secretKeyRef:
                  name: llm-secrets
                  key: openai-api-key
          livenessProbe:
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 10
          readinessProbe:
            httpGet:
              path: /ready
              port: 8000
            initialDelaySeconds: 5
```

---

## Terraform

```hcl
# deploy/terraform/main.tf
module "eks" {
  source          = "terraform-aws-modules/eks/aws"
  cluster_name    = "genai-cluster"
  cluster_version = "1.29"
  vpc_id          = module.vpc.vpc_id
  subnet_ids      = module.vpc.private_subnets
}

module "secrets" {
  source = "./modules/secrets"
  secrets = {
    "openai-api-key"    = var.openai_api_key
    "anthropic-api-key" = var.anthropic_api_key
  }
}
```

---

## CI/CD Pipeline

### GitHub Actions

```yaml
# .github/workflows/ci.yml
name: CI

on:
  push:
    branches: [main]
  pull_request:

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v4
      - run: uv sync --frozen
      - run: uv run ruff check .
      - run: uv run ruff format --check .
      - run: uv run mypy src/

  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v4
      - run: uv sync --frozen
      - run: uv run pytest tests/unit/ --cov=src --cov-report=xml
      - uses: codecov/codecov-action@v4

  security:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v4
      - run: uv sync --frozen
      - run: uv pip audit

  sonarqube:
    runs-on: ubuntu-latest
    needs: [test]
    steps:
      - uses: actions/checkout@v4
      - uses: SonarSource/sonarqube-scan-action@v3
        env:
          SONAR_TOKEN: ${{ secrets.SONAR_TOKEN }}

  deploy:
    needs: [lint, test, security]
    if: github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: docker build -t genai-app:${{ github.sha }} .
      - run: docker push genai-app:${{ github.sha }}
      - run: kubectl set image deployment/genai-app app=genai-app:${{ github.sha }}
```

---

## Environments

| Entorno | Propósito | LLM Model | Guardrails |
|---------|-----------|-----------|------------|
| local | Desarrollo | gpt-4o-mini / local | Relajados |
| staging | Pre-producción | gpt-4o | Activos |
| production | Producción | gpt-4o | Estrictos |

```python
# src/infrastructure/config.py
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    environment: str = "local"
    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.0
    max_tokens: int = 4096
    token_budget_daily: int = 1_000_000
    enable_guardrails: bool = True

    class Config:
        env_file = ".env"
```

---

## Health Checks

```python
@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.get("/ready")
async def ready():
    checks = {
        "vector_store": await vector_store.ping(),
        "llm": await llm.ping(),
    }
    all_healthy = all(checks.values())
    return {"ready": all_healthy, "checks": checks}
```

Ver también: [TOOLS.md](TOOLS.md), [OBSERVABILITY.md](OBSERVABILITY.md)
