variable "aws_region" {
  description = "AWS region for resource deployment"
  type        = string
  default     = "us-east-1"
}

variable "environment" {
  description = "Deployment environment (development, staging, production)"
  type        = string
  default     = "development"
}

variable "app_name" {
  description = "Application name used for resource naming"
  type        = string
  default     = "genai-app"
}

variable "vpc_cidr" {
  description = "CIDR block for the VPC"
  type        = string
  default     = "10.0.0.0/16"
}

variable "container_image" {
  description = "Docker image for the application"
  type        = string
  default     = "ghcr.io/your-org/genai-app:latest"
}

variable "container_cpu" {
  description = "CPU units for ECS task (1024 = 1 vCPU)"
  type        = number
  default     = 256
}

variable "container_memory" {
  description = "Memory in MiB for ECS task"
  type        = number
  default     = 512
}

variable "desired_count" {
  description = "Desired number of ECS tasks"
  type        = number
  default     = 1
}

variable "db_instance_class" {
  description = "RDS instance class"
  type        = string
  default     = "db.t3.micro"
}
