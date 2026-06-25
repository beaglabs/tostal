from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    database_url: str = "postgresql+asyncpg://tostal:tostal_dev@localhost:5432/tostal"
    database_url_sync: str = "postgresql://tostal:tostal_dev@localhost:5432/tostal"

    azure_storage_account: str = "devstoreaccount1"
    azure_storage_key: str = ""
    azure_storage_connection_string: str = ""

    jwt_secret: str = "dev-secret-change-in-production"
    jwt_algorithm: str = "HS256"
    jwt_expiration_minutes: int = 60

    api_host: str = "0.0.0.0"
    api_port: int = 8000
    cors_origins: str = "http://localhost:3000,http://localhost:5173"

    stripe_secret_key: str = ""
    stripe_webhook_secret: str = ""

    temporal_host: str = "localhost:7233"
    temporal_namespace: str = "default"

    environment: str = "development"
    log_level: str = "INFO"

    default_storage_quota_bytes: int = 107374182400  # 100 GB

    model_config = {"env_file": ".env", "case_sensitive": False}


@lru_cache
def get_settings() -> Settings:
    return Settings()