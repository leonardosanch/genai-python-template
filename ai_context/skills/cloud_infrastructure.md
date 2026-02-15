# Skill: Cloud & Infrastructure

## Description
This skill covers the automation, containerization, and orchestration of cloud-native applications. Use this when defining infrastructure, writing Dockerfiles, configuring Kubernetes, or setting up CI/CD pipelines.

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

## Cost Optimization Strategies

1.  **Spot Instances**: Use Spot instances for stateless worker agents (savings up to 90%).
2.  **Right Sizing**: Monitor CPU/Memory heavily. LLM apps often need high concurrency (async) but low CPU compared to training.
3.  **Auto-Scaling**: Aggressive scale-down policies during off-hours.
4.  **Token Budgeting**: Implement hard caps on daily token usage per service at the API gateway level.

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
