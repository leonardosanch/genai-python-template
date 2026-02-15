# src/infrastructure/config/settings.py
from functools import lru_cache

from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMSettings(BaseModel):
    """LLM related settings."""

    MODE: str = "openai"
    # OpenAI
    OPENAI_API_KEY: str = "YOUR_OPENAI_API_KEY"
    OPENAI_MODEL: str = "gpt-4o"
    # Anthropic
    ANTHROPIC_API_KEY: str = "YOUR_ANTHROPIC_API_KEY"
    ANTHROPIC_MODEL: str = "claude-3-opus-20240229"


class DatabaseSettings(BaseModel):
    """Database related settings."""

    URL: str = "postgresql+asyncpg://user:password@localhost:5432/genai"
    ECHO: bool = False


class RedisSettings(BaseModel):
    """Redis related settings."""

    URL: str = "redis://localhost:6379/0"


class DataEngineeringSettings(BaseModel):
    """Data engineering related settings."""

    DEFAULT_BATCH_SIZE: int = 1000
    MAX_RECORDS_IN_MEMORY: int = 100_000
    DEFAULT_FORMAT: str = "csv"
    CLEANING_STRIP_WHITESPACE: bool = True
    CLEANING_DROP_DUPLICATES: bool = False


class CloudSettings(BaseModel):
    """Cloud storage settings."""

    STORAGE_BACKEND: str = "local"
    S3_BUCKET: str = ""
    S3_REGION: str = "us-east-1"
    S3_ENDPOINT_URL: str = ""


class JWTSettings(BaseModel):
    """JWT authentication settings."""

    SECRET_KEY: str = "CHANGE-ME-IN-PRODUCTION"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    REFRESH_TOKEN_EXPIRE_MINUTES: int = 1440  # 24h
    ENABLED: bool = True


class SparkSettings(BaseModel):
    """Apache Spark related settings."""

    APP_NAME: str = "genai-spark"
    MASTER: str = "local[*]"
    DEPLOY_MODE: str = "client"
    EXECUTOR_MEMORY: str = "2g"
    EXECUTOR_CORES: int = 2
    DRIVER_MEMORY: str = "1g"
    SHUFFLE_PARTITIONS: int = 200
    JDBC_FETCH_SIZE: int = 10_000
    WAREHOUSE_DIR: str = "/tmp/spark-warehouse"
    DELTA_ENABLED: bool = False
    EXTRA_CONF: dict[str, str] = {}


class BackendSettings(BaseModel):
    """Backend selection for pluggable infrastructure components."""

    MESSAGE_BROKER: str = "memory"  # "memory" | "redis"
    EVENT_BUS: str = "memory"  # "memory" (future: "redis", "kafka")
    AUDIT_TRAIL: str = "memory"  # "memory" (future: "postgresql")
    COST_TRACKER: str = "memory"  # "memory" (future: "postgresql")


class ObservabilitySettings(BaseModel):
    """Observability related settings."""

    LOG_LEVEL: str = "INFO"
    OTEL_EXPORTER_OTLP_ENDPOINT: str = ""


class Settings(BaseSettings):
    """Main settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_nested_delimiter="__",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    APP_NAME: str = "GenAI API"
    ENVIRONMENT: str = "development"
    API_V1_PREFIX: str = "/api/v1"
    API_KEYS: list[str] = []
    RATE_LIMIT_RPM: int = 60
    RATE_LIMIT_BACKEND: str = "memory"  # "memory" | "redis"
    CORS_ORIGINS: list[str] = ["http://localhost:3000"]
    MAX_REQUEST_SIZE: int = 1_048_576  # 1MB

    llm: LLMSettings = LLMSettings()
    database: DatabaseSettings = DatabaseSettings()
    redis: RedisSettings = RedisSettings()
    data_engineering: DataEngineeringSettings = DataEngineeringSettings()
    cloud: CloudSettings = CloudSettings()
    jwt: JWTSettings = JWTSettings()
    spark: SparkSettings = SparkSettings()
    backends: BackendSettings = BackendSettings()
    observability: ObservabilitySettings = ObservabilitySettings()


@lru_cache
def get_settings() -> Settings:
    """Get the settings instance."""
    return Settings()
