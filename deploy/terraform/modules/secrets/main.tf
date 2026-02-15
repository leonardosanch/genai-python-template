resource "random_password" "db_password" {
  length  = 32
  special = true
  override_special = "!#$%&*()-_=+[]{}<>:?"
}

resource "aws_secretsmanager_secret" "db_password" {
  name = "${var.app_name}/${var.environment}/db-password"
}

resource "aws_secretsmanager_secret_version" "db_password" {
  secret_id     = aws_secretsmanager_secret.db_password.id
  secret_string = random_password.db_password.result
}

resource "aws_secretsmanager_secret" "openai_api_key" {
  name = "${var.app_name}/${var.environment}/openai-api-key"
}

# Note: The OpenAI API key value must be set manually in the AWS console
# or via CLI after initial terraform apply:
#   aws secretsmanager put-secret-value \
#     --secret-id genai-app/development/openai-api-key \
#     --secret-string "sk-..."

resource "aws_secretsmanager_secret" "jwt_secret" {
  name = "${var.app_name}/${var.environment}/jwt-secret"
}

resource "random_password" "jwt_secret" {
  length  = 64
  special = false
}

resource "aws_secretsmanager_secret_version" "jwt_secret" {
  secret_id     = aws_secretsmanager_secret.jwt_secret.id
  secret_string = random_password.jwt_secret.result
}
