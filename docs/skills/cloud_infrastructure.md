# Skill: Cloud & Infrastructure

## Description
This skill covers the automation, containerization, and orchestration of cloud-native applications. Use this when defining infrastructure, writing Dockerfiles, configuring Kubernetes, or setting up CI/CD pipelines.

## Executive Summary

**Critical infrastructure rules:**
- ALWAYS run containers as non-root (`USER appuser` in Dockerfile) — security vulnerability otherwise
- Multi-stage builds mandatory — minimize image size and never include build tools in production image
- Kubernetes: Resource Requests and Limits MUST be set — prevents OOMKilled and node starvation
- Liveness AND Readiness probes required — ensures traffic only goes to healthy pods
- Secrets NEVER in images or code — use K8s Secrets, AWS Secrets Manager, or environment injection at runtime

**Read full skill when:** Writing Dockerfiles, configuring K8s deployments, setting up Terraform modules, implementing HPA, or deploying to production environments.

---

## Versiones y Seguridad de Imagen

| Herramienta | Versión Mínima | Estabilidad |
|-------------|----------------|-------------|
| Docker | >= 24.0.0 | ✅ Estable |
| Terraform | >= 1.5.0 | ✅ Estable |
| Helm | >= 3.12.0 | ✅ Estable |
| AWS CLI | >= 2.13.0 | ✅ Estable |

### Docker Rootless Requirement

```dockerfile
# ✅ SIEMPRE crear y usar usuario no-root
RUN groupadd -g 10001 appgroup && \
    useradd -u 10001 -g appgroup -m -s /bin/bash appuser

# Cambiar permisos de archivos necesarios
RUN chown -R appuser:appgroup /app

USER appuser
```

### Resource Limits (K8s)

```yaml
resources:
  requests:
    cpu: "250m"
    memory: "512Mi"
  limits:
    cpu: "1000m"
    memory: "1Gi"
```

---

## Deep Dive

## Core Concepts

1.  **Infrastructure as Code (IaC)**: Provisioning infrastructure via declarative code (Terraform, Pulumi).
2.  **Immutability**: Containers and infrastructure components are replaced, not updated in-place.
3.  **Observability**: Metrics, logs, and traces are first-class citizens.
4.  **GitOps**: Git as the single source of truth for declarative infrastructure and applications.

---

## External Resources

### 🐳 Containerization

#### Docker
- **Docker Documentation**: [docs.docker.com](https://docs.docker.com/)
    - *Best for*: Core concepts, CLI, Compose
- **Dockerfile Best Practices**: [docs.docker.com/develop/develop-images/dockerfile_best-practices/](https://docs.docker.com/develop/develop-images/dockerfile_best-practices/)
    - *Best for*: Layer caching, image size optimization, security
- **Hadolint**: [github.com/hadolint/hadolint](https://github.com/hadolint/hadolint)
    - *Best for*: Linting Dockerfiles for best practices

#### OCI & Runtimes
- **Open Container Initiative (OCI)**: [opencontainers.org](https://opencontainers.org/)
    - *Best for*: Container standards
- **Podman**: [podman.io](https://podman.io/)
    - *Best for*: Daemonless, rootless container engine

---

### ☸️ Kubernetes & Orchestration

#### Core Kubernetes
- **Kubernetes Documentation**: [kubernetes.io/docs/home/](https://kubernetes.io/docs/home/)
    - *Best for*: API reference, concepts (Pods, Services, Deployments)
- **kubectl Cheat Sheet**: [kubernetes.io/docs/reference/kubectl/cheatsheet/](https://kubernetes.io/docs/reference/kubectl/cheatsheet/)
    - *Best for*: Common commands

#### K8s Ecosystem
- **Helm**: [helm.sh](https://helm.sh/)
    - *Best for*: Package management, templating charts
- **Kustomize**: [kustomize.io](https://kustomize.io/)
    - *Best for*: Configuration management, overlays without templates
- **K9s**: [k9scli.io](https://k9scli.io/)
    - *Best for*: Terminal UI for Kubernetes clusters

---

### ☁️ Infrastructure as Code (IaC)

#### Terraform / OpenTofu
- **Terraform Registry**: [registry.terraform.io](https://registry.terraform.io/)
    - *Best for*: Finding providers (AWS, Azure, GCP) and modules
- **Terraform Best Practices**: [www.terraform-best-practices.com](https://www.terraform-best-practices.com/)
    - *Best for*: File structure, naming conventions, state management
- **OpenTofu**: [opentofu.org](https://opentofu.org/)
    - *Best for*: Open-source alternative to Terraform

#### Other tools
- **Pulumi**: [www.pulumi.com](https://www.pulumi.com/)
    - *Best for*: IaC using general-purpose languages (Python, TS)
- **Ansible**: [docs.ansible.com](https://docs.ansible.com/)
    - *Best for*: Configuration management, server provisioning

---

### 🔄 CI/CD & GitOps

#### CI/CD Platforms
- **GitHub Actions**: [docs.github.com/en/actions](https://docs.github.com/en/actions)
    - *Best for*: Integrated CI/CD, workflow automation
- **GitLab CI/CD**: [docs.gitlab.com/ee/ci/](https://docs.gitlab.com/ee/ci/)
    - *Best for*: Integrated DevOps platform

#### GitOps
- **Argo CD**: [argo-cd.readthedocs.io](https://argo-cd.readthedocs.io/)
    - *Best for*: Declarative GitOps deployment for K8s
- **Flux**: [fluxcd.io](https://fluxcd.io/)
    - *Best for*: Continuous delivery for K8s

---

### 🌩️ Cloud Providers (Reference)

#### AWS
- **AWS Well-Architected Framework**: [aws.amazon.com/architecture/well-architected/](https://aws.amazon.com/architecture/well-architected/)
    - *Best for*: Operational excellence, security, reliability, performance, cost. **Continuously updated** (web version is authoritative).
- **AWS CloudFormation**: [docs.aws.amazon.com/cloudformation/](https://docs.aws.amazon.com/cloudformation/)

#### Google Cloud Platform (GCP)
- **Google Cloud Architecture Framework**: [cloud.google.com/architecture/framework](https://cloud.google.com/architecture/framework)
    - *Best for*: Complete guide to GCP cloud architecture. **Continuously updated** (no static annual PDF).
- **Google Kubernetes Engine (GKE)**: [cloud.google.com/kubernetes-engine](https://cloud.google.com/kubernetes-engine)

#### Azure
- **Azure Architecture Center**: [learn.microsoft.com/en-us/azure/architecture/](https://learn.microsoft.com/en-us/azure/architecture/)
- **Azure Kubernetes Service (AKS)**: [learn.microsoft.com/en-us/azure/aks/](https://learn.microsoft.com/en-us/azure/aks/)

---

### 📖 Books & Guides

#### Books
- **Kubernetes: Up and Running** (Brendan Burns, Joe Beda, Kelsey Hightower)
    - *Best for*: Understanding K8s from its creators
- **Terraform: Up & Running** (Yevgeniy Brikman)
    - *Best for*: Deep dive into Terraform/IaC
- **Effective DevOps** (Jennifer Davis, Ryn Daniels)
    - *Best for*: Culture and practices

#### Guides
- **The Twelve-Factor App**: [12factor.net](https://12factor.net/)
    - *Best for*: Cloud-native application methodology
- **CNCF Setup**: [landscape.cncf.io](https://landscape.cncf.io/)
    - *Best for*: Exploring the cloud-native ecosystem

---

## Decision Trees

### Decision Tree 1: Dónde desplegar

```
¿Cuál es tu escenario?
├── Prototipo / demo / equipo < 3
│   └── Cloud Run (GCP) o AWS App Runner
│       ├── Container-based, auto-scaling, zero ops
│       └── Pay per request — ideal para tráfico variable
├── Producción con tráfico predecible
│   └── ¿Ya tienes cluster Kubernetes?
│       ├── SÍ → Deploy en K8s existente
│       └── NO → ¿Necesitas orchestration compleja?
│           ├── SÍ → EKS/GKE/AKS (managed K8s)
│           └── NO → ECS Fargate / Cloud Run
├── GPU workloads (model serving, fine-tuning)
│   └── ¿Managed o self-hosted?
│       ├── Managed → AWS SageMaker / GCP Vertex AI
│       └── Self-hosted → K8s con GPU node pools
└── On-premise / air-gapped
    └── K8s self-managed + Docker registry privado
```

### Decision Tree 2: IaC — Terraform vs Pulumi vs CloudFormation

```
¿Cuál es tu contexto?
├── Multi-cloud o cloud-agnostic
│   └── Terraform / OpenTofu
│       ├── HCL, mayor ecosystem, más madurez
│       └── State management requiere configuración (S3+DynamoDB)
├── Equipo Python-first, quiere IaC en Python
│   └── Pulumi
│       ├── Python, TypeScript, Go como lenguaje
│       └── Menor ecosystem que Terraform
├── Solo AWS, equipo AWS-native
│   └── CloudFormation / CDK
│       └── Integración nativa, pero vendor lock-in
└── Configuración de servidores existentes
    └── Ansible (configuration management, no provisioning)
```

---

## Instructions for the Agent

1.  **Container Security**:
    - Always use non-root users in Dockerfiles (`USER appuser`).
    - Minimize image size (Multi-stage builds, Distroless or Alpine/Slim base images).
    - Scan images for vulnerabilities (Trivy, Docker Scout).
2.  **Code Reproducibility**:
    - Pin versions for EVERYTHING (Base images, system packages via apt/apk, Python deps via uv.lock).
    - Avoid `latest` tag in production manifests.
3.  **Kubernetes Best Practices**:
    - Always define **Resource Requests and Limits** (prevent OOMKilled/CPU starvation).
    - Include **Liveness** (restart if stuck) and **Readiness** (don't send traffic until ready) probes.
    - Use ConfigMaps and Secrets for configuration (12-Factor).
4.  **IaC Discipline**:
    - Treat infrastructure code like application code (Linting, PRs, Versioning).
    - Never modify infrastructure manually (Console ClickOps) -> Drift.
5.  **Observability Integration**:
    - Ensure apps log to stdout/stderr (JSON format preferred).
    - Expose metrics (Prometheus format) where possible.

---

## Code Examples

### Example 1: Multi-Stage Dockerfile for GenAI

```dockerfile
# Dockerfile
# Stage 1: Builder
FROM python:3.11-slim as builder

WORKDIR /app
COPY pyproject.toml uv.lock ./
# Install uv and dependencies
RUN pip install uv && \
    uv export --format requirements-txt > requirements.txt && \
    pip wheel --no-cache-dir --no-deps --wheel-dir /app/wheels -r requirements.txt

# Stage 2: Runtime
FROM python:3.11-slim

WORKDIR /app
COPY --from=builder /app/wheels /wheels
COPY --from=builder /app/requirements.txt .

RUN pip install --no-cache /wheels/*

COPY src/ ./src/

ENV PYTHONUNBUFFERED=1
ENV PORT=8000

# Non-root user for security
RUN adduser --disabled-password --gecos '' appuser
USER appuser

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Example 2: Kubernetes HPA for LLM Workloads

```yaml
# k8s/hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: genai-service-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: genai-service
  minReplicas: 2
  maxReplicas: 10
  metrics:
  # Scale on CPU usage (standard)
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  # Scale on Custom Metric (e.g., Active Requests or Queue Depth)
  - type: Pods
    pods:
      metric:
        name: active_requests
      target:
        type: AverageValue
        averageValue: 5
```

### Example 3: Terraform AWS ECS Setup

```hcl
# terraform/ecs.tf
resource "aws_ecs_task_definition" "app" {
  family                   = "genai-app"
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                      = 1024
  memory                   = 2048
  execution_role_arn       = aws_iam_role.ecs_execution.arn
  task_role_arn            = aws_iam_role.ecs_task.arn

  container_definitions = jsonencode([{
    name  = "app"
    image = "${aws_ecr_repository.app.repository_url}:latest"
    environment = [
      { name = "ENVIRONMENT", value = "production" }
    ]
    secrets = [
      { name = "OPENAI_API_KEY", valueFrom = aws_secretsmanager_secret.openai_key.arn }
    ]
    logConfiguration = {
      logDriver = "awslogs"
      options = {
        "awslogs-group" = "/ecs/genai-app"
        "awslogs-region" = var.region
        "awslogs-stream-prefix" = "ecs"
      }
    }
  }])
}
```

---

## Cost Optimization & FinOps

### FinOps Principles
- **Inform**: Full visibility of cloud spend by team, service, and environment
- **Optimize**: Continuous right-sizing, commitment-based discounts, waste elimination
- **Operate**: Cost-aware engineering culture, budgets with alerts, showback/chargeback

### Strategies

1.  **Spot/Preemptible Instances**: Use for stateless worker agents (savings up to 90%). Always with fallback to on-demand.
2.  **Right Sizing**: Monitor CPU/Memory heavily. LLM apps often need high concurrency (async) but low CPU compared to training.
3.  **Commitment Discounts**: Reserved Instances (1-3yr) for baseline, Savings Plans for flexible workloads.
4.  **Auto-Scaling**: Aggressive scale-down policies during off-hours. Schedule-based scaling for predictable patterns.
5.  **Token Budgeting**: Implement hard caps on daily token usage per service at the API gateway level.
6.  **Storage Lifecycle**: S3 Intelligent-Tiering, auto-archive to Glacier after 90 days.
7.  **Tagging Strategy**: Mandatory tags for cost allocation (`team`, `env`, `service`, `cost-center`).

### Cost Monitoring Setup

```hcl
# terraform/cost_alerts.tf
resource "aws_budgets_budget" "monthly" {
  name         = "monthly-budget"
  budget_type  = "COST"
  limit_amount = "5000"
  limit_unit   = "USD"
  time_unit    = "MONTHLY"

  notification {
    comparison_operator       = "GREATER_THAN"
    threshold                 = 80
    threshold_type            = "PERCENTAGE"
    notification_type         = "FORECASTED"
    subscriber_email_addresses = [var.billing_alert_email]
  }

  notification {
    comparison_operator       = "GREATER_THAN"
    threshold                 = 100
    threshold_type            = "PERCENTAGE"
    notification_type         = "ACTUAL"
    subscriber_email_addresses = [var.billing_alert_email]
  }

  cost_filter {
    name   = "TagKeyValue"
    values = ["user:env$production"]
  }
}
```

### Tagging Enforcement

```hcl
# terraform/tagging.tf
locals {
  required_tags = {
    Environment = var.environment
    Team        = var.team
    Service     = var.service_name
    CostCenter  = var.cost_center
    ManagedBy   = "terraform"
  }
}

# Apply to all resources via default_tags
provider "aws" {
  region = var.region
  default_tags {
    tags = local.required_tags
  }
}
```

---

## Disaster Recovery & Business Continuity

### RTO / RPO Definitions

| Tier | RTO | RPO | Estrategia | Costo Relativo |
|------|-----|-----|-----------|----------------|
| Tier 1 (Critical) | < 1h | < 15min | Multi-region active-active | $$$$$ |
| Tier 2 (Important) | < 4h | < 1h | Multi-region active-passive (warm standby) | $$$ |
| Tier 3 (Standard) | < 24h | < 4h | Pilot light + automated restore | $$ |
| Tier 4 (Non-critical) | < 72h | < 24h | Backup & restore from snapshots | $ |

### Decision Tree: DR Strategy

```
¿Cuál es tu RTO requerido?
├── < 1 hora (mission-critical)
│   └── Active-Active Multi-Region
│       ├── Global load balancer (Route53, Cloud DNS)
│       ├── Database replication síncrona (Aurora Global, CockroachDB)
│       └── Costo: ~2x infraestructura base
├── < 4 horas (business-critical)
│   └── Warm Standby
│       ├── Réplicas reducidas en región DR (e.g., 2 pods vs 10)
│       ├── Database replication asíncrona (RDS cross-region read replica)
│       └── Scale-up automático en failover
├── < 24 horas (important)
│   └── Pilot Light
│       ├── Infraestructura core provisionada pero apagada
│       ├── Data replication continua
│       └── Terraform apply para escalar en evento DR
└── > 24 horas (non-critical)
    └── Backup & Restore
        ├── Snapshots diarios a otra región
        ├── IaC para recrear desde cero
        └── Restore manual con runbook documentado
```

### Multi-Region Failover (Terraform)

```hcl
# terraform/dr/route53_failover.tf
resource "aws_route53_health_check" "primary" {
  fqdn              = "api-primary.example.com"
  port               = 443
  type               = "HTTPS"
  resource_path      = "/health"
  failure_threshold  = 3
  request_interval   = 10

  tags = {
    Name = "primary-health-check"
  }
}

resource "aws_route53_record" "api_failover_primary" {
  zone_id = var.hosted_zone_id
  name    = "api.example.com"
  type    = "A"

  failover_routing_policy {
    type = "PRIMARY"
  }

  set_identifier = "primary"
  health_check_id = aws_route53_health_check.primary.id

  alias {
    name                   = aws_lb.primary.dns_name
    zone_id                = aws_lb.primary.zone_id
    evaluate_target_health = true
  }
}

resource "aws_route53_record" "api_failover_secondary" {
  zone_id = var.hosted_zone_id
  name    = "api.example.com"
  type    = "A"

  failover_routing_policy {
    type = "SECONDARY"
  }

  set_identifier = "secondary"

  alias {
    name                   = aws_lb.secondary.dns_name
    zone_id                = aws_lb.secondary.zone_id
    evaluate_target_health = true
  }
}
```

### Cross-Region Backup (K8s CronJob)

```yaml
# k8s/dr/backup-cronjob.yaml
apiVersion: batch/v1
kind: CronJob
metadata:
  name: db-backup-cross-region
spec:
  schedule: "0 */6 * * *"  # Every 6 hours
  concurrencyPolicy: Forbid
  jobTemplate:
    spec:
      backoffLimit: 2
      template:
        spec:
          serviceAccountName: backup-sa
          containers:
          - name: backup
            image: postgres:16-alpine
            command:
            - /bin/sh
            - -c
            - |
              pg_dump $DATABASE_URL | gzip | \
              aws s3 cp - s3://${BACKUP_BUCKET}/db-$(date +%Y%m%d-%H%M).sql.gz \
                --region ${DR_REGION}
            envFrom:
            - secretRef:
                name: db-credentials
            - configMapRef:
                name: backup-config
            resources:
              requests:
                cpu: "250m"
                memory: "512Mi"
              limits:
                cpu: "500m"
                memory: "1Gi"
          restartPolicy: OnFailure
```

---

## Cloud Migration Strategies

### The 7Rs Framework

```
¿Qué hacer con cada workload?
├── Retire — Eliminar (no migrar, ya no se usa)
├── Retain — Mantener on-premise (regulatorio, latencia, deuda técnica)
├── Rehost — "Lift & Shift" (VM → EC2, sin cambios de código)
│   └── Rápido pero no optimiza costos cloud
├── Relocate — Mover a managed platform (VMware on-prem → VMware Cloud on AWS)
├── Replatform — "Lift, Tinker & Shift" (ajustes menores)
│   ├── DB → RDS/Aurora (managed)
│   ├── App server → Container (ECS/EKS)
│   └── Cron jobs → Lambda / Cloud Functions
├── Refactor — Rearquitectar para cloud-native
│   ├── Monolito → Microservicios
│   ├── Sync → Event-driven
│   └── Mayor esfuerzo, máximo beneficio
└── Repurchase — Reemplazar con SaaS
    └── CRM custom → Salesforce, email server → SES
```

### Migration Waves Planning

```yaml
# migration-plan.yaml (documentation artifact)
waves:
  - name: "Wave 0 — Foundation"
    duration: "4 weeks"
    tasks:
      - Landing zone setup (accounts, networking, IAM)
      - CI/CD pipeline for IaC
      - Monitoring & logging infrastructure
      - VPN / Direct Connect to on-premise

  - name: "Wave 1 — Quick Wins"
    duration: "3 weeks"
    strategy: "Rehost / Replatform"
    workloads:
      - Static websites → S3 + CloudFront
      - Dev/staging environments → ECS Fargate
      - Cron jobs → Lambda + EventBridge

  - name: "Wave 2 — Core Services"
    duration: "6 weeks"
    strategy: "Replatform"
    workloads:
      - PostgreSQL → RDS Aurora
      - Redis → ElastiCache
      - API monolith → ECS with ALB

  - name: "Wave 3 — Refactor"
    duration: "8+ weeks"
    strategy: "Refactor"
    workloads:
      - Monolith decomposition → microservices
      - Sync processing → event-driven (SQS/SNS)
      - GenAI services → purpose-built infrastructure
```

### Migration Validation Checklist

- [ ] Application dependency map documented
- [ ] Performance baseline captured (latency, throughput, error rate)
- [ ] Data migration tested with production-like volume
- [ ] DNS cutover plan with rollback procedure
- [ ] Rollback tested end-to-end
- [ ] Post-migration performance matches or exceeds baseline
- [ ] Cost projection validated against actual spend (30 days)

---

## Hybrid Cloud Patterns

### Connectivity Options

| Option | Latency | Bandwidth | Cost | Use Case |
|--------|---------|-----------|------|----------|
| Site-to-Site VPN | ~50ms | < 1 Gbps | $ | Dev/staging, low-traffic |
| AWS Direct Connect | ~5ms | 1-100 Gbps | $$$$ | Production, data-heavy |
| ExpressRoute (Azure) | ~5ms | 1-100 Gbps | $$$$ | Azure hybrid |
| Cloud Interconnect (GCP) | ~5ms | 10-200 Gbps | $$$$ | GCP hybrid |
| Transit Gateway | Variable | Aggregated | $$$ | Multi-VPC hub-and-spoke |

### Workload Placement Decision

```
¿Dónde ubicar el workload?
├── Datos sensibles con regulación de residencia
│   └── On-premise o región cloud específica
├── Requiere GPU / capacidad elástica
│   └── Cloud (spot instances para training, on-demand para inference)
├── Latencia < 10ms a sistemas on-premise
│   └── On-premise o edge (AWS Outposts, Azure Stack)
├── Stateless, escala variable
│   └── Cloud (auto-scaling, serverless)
└── Legacy con dependencias no portables
    └── On-premise (candidato a Retain o Rehost futuro)
```

---

## Landing Zone Design

### Account Structure (AWS)

```
Organization Root
├── Security OU
│   ├── Log Archive Account (CloudTrail, Config, GuardDuty)
│   └── Security Tooling Account (SecurityHub, Inspector)
├── Infrastructure OU
│   ├── Networking Account (Transit Gateway, DNS, VPN)
│   └── Shared Services Account (CI/CD, Container Registry, Artifacts)
├── Workloads OU
│   ├── Production OU
│   │   ├── genai-prod Account
│   │   └── data-prod Account
│   └── Non-Production OU
│       ├── genai-dev Account
│       ├── genai-staging Account
│       └── sandbox Account
└── Suspended OU (decommissioned accounts)
```

### Landing Zone Terraform Module

```hcl
# terraform/landing-zone/main.tf
module "organization" {
  source = "./modules/organization"

  organizational_units = {
    security       = { parent = "root" }
    infrastructure = { parent = "root" }
    workloads      = { parent = "root" }
    production     = { parent = "workloads" }
    non_production = { parent = "workloads" }
  }

  accounts = {
    log-archive = {
      ou    = "security"
      email = "aws-logs@example.com"
    }
    networking = {
      ou    = "infrastructure"
      email = "aws-network@example.com"
    }
    genai-prod = {
      ou    = "production"
      email = "aws-genai-prod@example.com"
    }
    genai-dev = {
      ou    = "non_production"
      email = "aws-genai-dev@example.com"
    }
  }
}

module "guardrails" {
  source = "./modules/scp"

  deny_policies = [
    "deny-root-account-usage",
    "deny-leave-organization",
    "deny-disable-cloudtrail",
    "deny-non-approved-regions",
  ]

  approved_regions = ["us-east-1", "us-west-2", "eu-west-1"]
}

module "centralized_logging" {
  source = "./modules/logging"

  log_archive_account_id = module.organization.accounts["log-archive"].id
  retention_days         = 365
  enable_cloudtrail      = true
  enable_config          = true
  enable_guardduty       = true
}
```

### Network Topology (Hub-and-Spoke)

```hcl
# terraform/landing-zone/networking.tf
resource "aws_ec2_transit_gateway" "hub" {
  description                     = "Central hub for all VPCs"
  default_route_table_association = "disable"
  default_route_table_propagation = "disable"

  tags = { Name = "tgw-hub" }
}

module "vpc_production" {
  source = "./modules/vpc"

  name               = "vpc-production"
  cidr_block         = "10.1.0.0/16"
  availability_zones = ["us-east-1a", "us-east-1b", "us-east-1c"]
  private_subnets    = ["10.1.1.0/24", "10.1.2.0/24", "10.1.3.0/24"]
  public_subnets     = ["10.1.101.0/24", "10.1.102.0/24", "10.1.103.0/24"]
  transit_gateway_id = aws_ec2_transit_gateway.hub.id
  enable_nat_gateway = true
  single_nat_gateway = false  # HA: one NAT per AZ in production
}

module "vpc_dev" {
  source = "./modules/vpc"

  name               = "vpc-dev"
  cidr_block         = "10.2.0.0/16"
  availability_zones = ["us-east-1a", "us-east-1b"]
  private_subnets    = ["10.2.1.0/24", "10.2.2.0/24"]
  public_subnets     = ["10.2.101.0/24", "10.2.102.0/24"]
  transit_gateway_id = aws_ec2_transit_gateway.hub.id
  enable_nat_gateway = true
  single_nat_gateway = true  # Cost saving: single NAT in dev
}
```

---

## Project Scaffolding

This project includes production-ready deployment artifacts:

- **Kustomize** (`deploy/k8s/`): Base manifests + overlays for dev/staging/prod with NetworkPolicy, PDB, Ingress
- **Terraform** (`deploy/terraform/`): AWS modules for VPC, ECS Fargate, RDS PostgreSQL, Secrets Manager
- **CI/CD** (`.github/workflows/ci.yml`): Quality + test + Docker build/push to ghcr.io

Apply with:
```bash
# Kubernetes
kubectl apply -k deploy/k8s/overlays/dev/

# Terraform
cd deploy/terraform && terraform init && terraform apply
```

---

## Anti-Patterns to Avoid

### ❌ Secrets in Docker Images
**Problem**: API keys embedded in image layers  
**Solution**: Use runtime environment variables, K8s Secrets, or AWS Secrets Manager.

### ❌ Running as Root
**Problem**: Security risk if container is compromised  
**Solution**: `USER appuser` in Dockerfile.

### ❌ No Health Probes
**Problem**: Load balancer sends traffic to dead pods/containers  
**Solution**: Implement Liveness (restart if dead) and Readiness (don't send traffic until ready) probes.

### ❌ Unbounded Resources
**Problem**: One run-away container OOMs the entire node
**Solution**: Always set `resources.limits.memory` and `resources.requests.cpu`.

### ❌ No DR Testing
**Problem**: DR plan exists on paper but has never been executed
**Solution**: Quarterly DR drills with documented runbooks. Automate failover and validate RTO/RPO.

### ❌ Big Bang Migration
**Problem**: Migrating all workloads simultaneously — high risk, impossible rollback
**Solution**: Migration waves with independent rollback per wave. Validate each wave before proceeding.

### ❌ No Cost Visibility
**Problem**: $50K cloud bill surprise at end of month
**Solution**: Mandatory tagging, budget alerts at 80% forecasted, daily cost anomaly detection.

### ❌ Single-Account Everything
**Problem**: All environments (dev/staging/prod) in one cloud account — blast radius is total
**Solution**: Multi-account landing zone with OU-based guardrails and SCPs.

---

## Infrastructure Checklist

### Containerization
- [ ] Multi-stage build used
- [ ] Non-root user configured
- [ ] Image scanning enabled (Trivy/Clair)
- [ ] No secrets in build args or image

### Orchestration (K8s)
- [ ] Resource Requests/Limits set
- [ ] Liveness/Readiness probes configured
- [ ] Horizontal Pod Autoscalers active
- [ ] Pod Disruption Budgets set
- [ ] Network Policies restrictive

### IaC (Terraform)
- [ ] State file encrypted and locked (S3 + DynamoDB)
- [ ] Modules used for reusability
- [ ] `terraform validate` and `tflint` passing in CI
- [ ] Secrets passed via Secrets Manager/Vault

### Disaster Recovery
- [ ] RTO/RPO defined per service tier
- [ ] Cross-region backups automated
- [ ] Failover tested quarterly with runbook
- [ ] Rollback procedure documented and tested

### FinOps
- [ ] Mandatory tagging enforced (team, env, service, cost-center)
- [ ] Budget alerts at 80% forecasted and 100% actual
- [ ] Right-sizing review monthly
- [ ] Unused resources cleanup automated

### Landing Zone
- [ ] Multi-account structure with OUs
- [ ] SCPs enforce approved regions and deny dangerous actions
- [ ] Centralized logging (CloudTrail, Config, GuardDuty)
- [ ] Hub-and-spoke network topology

---

## Additional References

- **CNCF Best Practices**: [cncf.io/blog/](https://www.cncf.io/blog/)
    - *Best for*: Cloud-native standards
- **FinOps Foundation**: [finops.org](https://www.finops.org/)
    - *Best for*: Cloud cost management
- **Kubernetes Patterns**: [k8spatterns.io](https://k8spatterns.io/)
    - *Best for*: Deployment strategies
- **12 Factors**: [12factor.net](https://12factor.net/)
    - *Best for*: App methodology
- **FinOps Foundation**: [finops.org](https://www.finops.org/)
    - *Best for*: Cloud cost management practices and maturity model
- **AWS Disaster Recovery Whitepaper**: [docs.aws.amazon.com/whitepapers/latest/disaster-recovery-workloads-on-aws/](https://docs.aws.amazon.com/whitepapers/latest/disaster-recovery-workloads-on-aws/)
    - *Best for*: DR strategies (backup/restore, pilot light, warm standby, active-active)
- **AWS Landing Zone Accelerator**: [aws.amazon.com/solutions/implementations/landing-zone-accelerator-on-aws/](https://aws.amazon.com/solutions/implementations/landing-zone-accelerator-on-aws/)
    - *Best for*: Multi-account best practices, SCPs, guardrails
- **AWS Migration Hub**: [docs.aws.amazon.com/migrationhub/](https://docs.aws.amazon.com/migrationhub/)
    - *Best for*: Migration tracking, 7Rs assessment, dependency mapping
