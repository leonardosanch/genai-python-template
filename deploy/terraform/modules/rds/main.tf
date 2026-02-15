resource "aws_db_subnet_group" "main" {
  name       = "${var.app_name}-${var.environment}"
  subnet_ids = var.private_subnet_ids

  tags = { Name = "${var.app_name}-${var.environment}-db-subnet-group" }
}

resource "aws_security_group" "rds" {
  name_prefix = "${var.app_name}-${var.environment}-rds-"
  vpc_id      = var.vpc_id

  ingress {
    from_port       = 5432
    to_port         = 5432
    protocol        = "tcp"
    security_groups = [var.ecs_security_group_id]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  lifecycle { create_before_destroy = true }
}

data "aws_secretsmanager_secret_version" "db_password" {
  secret_id = var.db_password_secret_arn
}

resource "aws_db_instance" "main" {
  identifier = "${var.app_name}-${var.environment}"

  engine         = "postgres"
  engine_version = "15"
  instance_class = var.db_instance_class

  allocated_storage     = 20
  max_allocated_storage = 100
  storage_encrypted     = true

  db_name  = replace("${var.app_name}_${var.environment}", "-", "_")
  username = "genai_admin"
  password = data.aws_secretsmanager_secret_version.db_password.secret_string

  db_subnet_group_name   = aws_db_subnet_group.main.name
  vpc_security_group_ids = [aws_security_group.rds.id]

  multi_az            = var.environment == "production" ? true : false
  skip_final_snapshot = var.environment != "production"

  backup_retention_period = 7
  backup_window           = "03:00-04:00"
  maintenance_window      = "Mon:04:00-Mon:05:00"

  tags = { Name = "${var.app_name}-${var.environment}-postgres" }
}
