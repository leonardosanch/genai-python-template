# Deployment

## Quick Start

### Local Development
```bash
docker compose up -d
# App at http://localhost:8000
# Health: http://localhost:8000/health
```

### Kubernetes (Kustomize)
```bash
# Dev
kubectl apply -k deploy/k8s/overlays/dev/

# Staging
kubectl apply -k deploy/k8s/overlays/staging/

# Production
kubectl apply -k deploy/k8s/overlays/prod/
```

### Terraform (AWS)
```bash
cd deploy/terraform
cp terraform.tfvars.example terraform.tfvars
# Edit terraform.tfvars with your values
terraform init
terraform plan
terraform apply
```

---

## Docker

### Build
```bash
docker build -t genai-app:latest .
```

The Dockerfile uses:
- **Multi-stage build** — builder + runner stages
- **Non-root user** — `appuser` (UID 1000)
- **uv** for dependency management
- **Python-based healthcheck** (no curl dependency needed)

### Docker Compose (Development)
```bash
docker compose up -d        # Start all services
docker compose logs -f app  # Follow app logs
docker compose down         # Stop all
```

Services: PostgreSQL, Redis, app with healthchecks.

---

## Kubernetes

### Structure (Kustomize)
```
deploy/k8s/
├── base/                    # Shared manifests
│   ├── kustomization.yaml
│   ├── namespace.yaml
│   ├── deployment.yaml
│   ├── service.yaml
│   ├── hpa.yaml
│   ├── networkpolicy.yaml
│   └── pdb.yaml
└── overlays/
    ├── dev/                 # 1 replica, debug logging
    ├── staging/             # 2 replicas, TLS ingress
    └── prod/                # 3+ replicas, strict resources, TLS
```

### Environment Differences

| | Dev | Staging | Production |
|---|---|---|---|
| Replicas | 1 | 2 | 3 |
| CPU request | 100m | 250m | 500m |
| Memory request | 128Mi | 512Mi | 1Gi |
| LOG_LEVEL | DEBUG | INFO | WARNING |
| TLS | No | Yes | Yes |
| HPA max | 3 | 5 | 10 |

### Preview Manifests
```bash
kubectl kustomize deploy/k8s/overlays/dev/
kubectl kustomize deploy/k8s/overlays/staging/
kubectl kustomize deploy/k8s/overlays/prod/
```

### Secrets
Base `secret.yaml` is a **template only**. In real clusters:
- Dev/Staging: Use `kubectl create secret` or sealed-secrets
- Production: Use AWS Secrets Manager / Vault with external-secrets-operator

---

## Terraform (AWS)

### Architecture
- **VPC**: 2 AZs, public + private subnets, NAT gateway
- **ECS Fargate**: Serverless containers, ALB, auto-scaling
- **RDS PostgreSQL 15**: Encrypted, automated backups (7 days)
- **Secrets Manager**: DB credentials (auto-generated), API keys

### Modules
```
deploy/terraform/
├── main.tf
├── variables.tf
├── outputs.tf
├── terraform.tfvars.example
└── modules/
    ├── networking/    # VPC, subnets, NAT, route tables
    ├── ecs/           # Cluster, task def, service, ALB, IAM
    ├── rds/           # PostgreSQL, subnet group, security group
    └── secrets/       # Secrets Manager entries
```

### Remote State (Recommended)
Uncomment the S3 backend in `main.tf` and create:
```bash
aws s3api create-bucket --bucket your-terraform-state-bucket --region us-east-1
aws dynamodb create-table \
  --table-name terraform-locks \
  --attribute-definitions AttributeName=LockID,AttributeType=S \
  --key-schema AttributeName=LockID,KeyType=HASH \
  --billing-mode PAY_PER_REQUEST
```

### Environment Overrides
```bash
# Production
terraform apply -var="environment=production" \
  -var="container_cpu=1024" \
  -var="container_memory=2048" \
  -var="desired_count=3" \
  -var="db_instance_class=db.r6g.large"
```

---

## CI/CD Pipeline

### GitHub Actions (`.github/workflows/ci.yml`)

| Job | Trigger | What |
|---|---|---|
| `quality` | Push + PR | Ruff lint + format, mypy |
| `test` | After quality | pytest + coverage + audit |
| `build` | Push to main | Docker build + push to ghcr.io |
| `deploy` | Commented out | K8s or ECS deployment |

### Container Registry
Images are pushed to `ghcr.io/<org>/<repo>` with tags:
- `sha-<commit>` — immutable per commit
- `latest` — latest main build

---

## Health Checks

| Endpoint | Purpose | K8s Probe |
|---|---|---|
| `/health` | App is alive | livenessProbe |
| `/ready` | App can serve traffic | readinessProbe |

---

## Environments

| Environment | LLM Model | Guardrails |
|---|---|---|
| development | gpt-4o-mini / local | Relaxed |
| staging | gpt-4o | Active |
| production | gpt-4o | Strict |

See also: [TOOLS.md](TOOLS.md), [OBSERVABILITY.md](OBSERVABILITY.md)
