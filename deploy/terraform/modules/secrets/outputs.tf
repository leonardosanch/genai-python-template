output "db_password_secret_arn" {
  description = "ARN of the DB password secret"
  value       = aws_secretsmanager_secret.db_password.arn
}

output "openai_api_key_secret_arn" {
  description = "ARN of the OpenAI API key secret"
  value       = aws_secretsmanager_secret.openai_api_key.arn
}

output "jwt_secret_arn" {
  description = "ARN of the JWT secret"
  value       = aws_secretsmanager_secret.jwt_secret.arn
}
