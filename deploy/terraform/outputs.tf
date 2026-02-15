output "region" {
  description = "AWS region used"
  value       = var.aws_region
}

output "environment" {
  description = "Deployment environment"
  value       = var.environment
}

output "vpc_id" {
  description = "VPC ID"
  value       = module.networking.vpc_id
}

output "alb_dns_name" {
  description = "ALB DNS name for the application"
  value       = module.ecs.alb_dns_name
}

output "db_endpoint" {
  description = "RDS PostgreSQL endpoint"
  value       = module.rds.db_endpoint
}

output "ecs_cluster_name" {
  description = "ECS cluster name"
  value       = module.ecs.cluster_name
}
