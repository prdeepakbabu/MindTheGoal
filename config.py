"""Configuration management for MindTheGoal evaluation framework."""

import os
from typing import Optional, List
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application configuration loaded from environment variables."""

    # ===========================
    # AWS Configuration
    # ===========================
    aws_region: str = Field(default="us-west-2", alias="AWS_REGION")
    aws_profile: Optional[str] = Field(default=None, alias="AWS_PROFILE")

    # ===========================
    # Model Configuration
    # ===========================
    bedrock_model_id: str = Field(
        default="anthropic.claude-3-5-sonnet-20241022-v2:0",
        alias="BEDROCK_MODEL_ID"
    )
    judge_temperature: float = Field(default=0.1, alias="JUDGE_TEMPERATURE")
    chat_temperature: float = Field(default=0.7, alias="CHAT_TEMPERATURE")
    max_tokens: int = Field(default=4096, alias="MAX_TOKENS")

    # ===========================
    # Server Configuration
    # ===========================
    host: str = Field(default="0.0.0.0", alias="HOST")
    port: int = Field(default=8000, alias="PORT")
    debug: bool = Field(default=True, alias="DEBUG")
    cors_origins: str = Field(
        default="http://localhost:3000,http://localhost:5173",
        alias="CORS_ORIGINS"
    )

    # ===========================
    # Database Configuration
    # ===========================
    database_url: str = Field(
        default="sqlite:///./mindthegoal.db",
        alias="DATABASE_URL"
    )

    # ===========================
    # Logging Configuration
    # ===========================
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    log_format: str = Field(default="text", alias="LOG_FORMAT")

    # ===========================
    # Dataset Configuration
    # ===========================
    datasets_dir: str = Field(default="datasets/data", alias="DATASETS_DIR")
    default_sample_size: int = Field(default=100, alias="DEFAULT_SAMPLE_SIZE")

    # ===========================
    # Rate Limiting
    # ===========================
    rate_limit_requests_per_minute: int = Field(
        default=60,
        alias="RATE_LIMIT_REQUESTS_PER_MINUTE"
    )
    max_concurrent_evaluations: int = Field(
        default=5,
        alias="MAX_CONCURRENT_EVALUATIONS"
    )

    # ===========================
    # Session Configuration
    # ===========================
    session_timeout_minutes: int = Field(default=60, alias="SESSION_TIMEOUT_MINUTES")
    max_messages_per_session: int = Field(default=100, alias="MAX_MESSAGES_PER_SESSION")

    # ===========================
    # Feature Flags
    # ===========================
    enable_streaming: bool = Field(default=True, alias="ENABLE_STREAMING")
    enable_multi_teacher: bool = Field(default=False, alias="ENABLE_MULTI_TEACHER")
    enable_cache: bool = Field(default=True, alias="ENABLE_CACHE")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        populate_by_name = True
        extra = "ignore"

    @property
    def cors_origins_list(self) -> List[str]:
        """Get CORS origins as a list."""
        return [origin.strip() for origin in self.cors_origins.split(",")]

    def get_datasets_path(self, dataset_name: str = "") -> str:
        """Get the full path to a dataset directory."""
        return os.path.join(self.datasets_dir, dataset_name)

    def get_output_path(self) -> str:
        """Get the full path to the output directory."""
        return "output"

    def get_reports_path(self) -> str:
        """Get the full path to the reports directory."""
        return os.path.join(self.get_output_path(), "reports")


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get the global settings instance."""
    return settings


def reload_settings() -> Settings:
    """Reload settings from environment (useful for testing)."""
    global settings
    settings = Settings()
    return settings
