terraform {
  required_version = ">= 1.5.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }

  # Configure your backend for remote state:
  # backend "s3" {
  #   bucket         = "your-terraform-state-bucket"
  #   key            = "genai-app/terraform.tfstate"
  #   region         = "us-east-1"
  #   dynamodb_table = "terraform-locks"
  #   encrypt        = true
  # }
}

provider "aws" {
  region = var.aws_region

  default_tags {
    tags = {
      Project     = var.app_name
      Environment = var.environment
      ManagedBy   = "terraform"
    }
  }
}

module "networking" {
  source = "./modules/networking"

  app_name    = var.app_name
  environment = var.environment
  vpc_cidr    = var.vpc_cidr
}

module "secrets" {
  source = "./modules/secrets"

  app_name    = var.app_name
  environment = var.environment
}

module "rds" {
  source = "./modules/rds"

  app_name               = var.app_name
  environment            = var.environment
  vpc_id                 = module.networking.vpc_id
  private_subnet_ids     = module.networking.private_subnet_ids
  ecs_security_group_id  = module.networking.ecs_security_group_id
  db_instance_class      = var.db_instance_class
  db_password_secret_arn = module.secrets.db_password_secret_arn
}

module "ecs" {
  source = "./modules/ecs"

  app_name              = var.app_name
  environment           = var.environment
  vpc_id                = module.networking.vpc_id
  public_subnet_ids     = module.networking.public_subnet_ids
  private_subnet_ids    = module.networking.private_subnet_ids
  ecs_security_group_id = module.networking.ecs_security_group_id
  alb_security_group_id = module.networking.alb_security_group_id
  container_image       = var.container_image
  container_cpu         = var.container_cpu
  container_memory      = var.container_memory
  desired_count         = var.desired_count
  db_secret_arn         = module.secrets.db_password_secret_arn
  openai_secret_arn     = module.secrets.openai_api_key_secret_arn
  db_endpoint           = module.rds.db_endpoint
  db_name               = module.rds.db_name
}
