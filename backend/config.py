from __future__ import annotations

import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    database_url: str = "sqlite:///./data/presento.db"
    attendance_min_confidence: float = 0.48
    # Liveness is gated by state string (REAL_LOCKED), not a probability score.
    # This key is used by the AI Vision engine to authenticate against /api/attendance/events.
    presento_ai_api_key: str = "change-me-before-production-ai-key"
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"
    jwt_secret_key: str = "presento-secure-default-jwt-secret-replace-in-env"
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 1440
    api_token: str = "change-me-before-production"
    log_level: str = "INFO"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @property
    def cors_origin_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]


settings = Settings()
