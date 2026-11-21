"""
Backend Configuration for SmartAttend AI.
"""

import os
from typing import List
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    PROJECT_NAME: str = "SmartAttend AI Backend"
    API_V1_PREFIX: str = "/api/v1"
    VERSION: str = "1.0.0"
    
    # JWT & Auth
    SECRET_KEY: str = os.getenv("JWT_SECRET_KEY", "SmartAttendAI-SuperSecretKey-ChangeInProduction-2026")
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24  # 24 hours
    
    # Database
    MONGODB_URI: str = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
    DATABASE_NAME: str = os.getenv("DATABASE_NAME", "smartattend_db")
    USE_FALLBACK_JSON_DB: bool = True  # Automatically uses resilient local JSON if MongoDB is offline
    
    # Biometrics & AI
    BIOMETRIC_SECRET_KEY: str = os.getenv("SMARTATTEND_BIOMETRIC_KEY", "SmartAttendAI-SecretEncryptionKey-32Bytes!")
    DEFAULT_RECOGNITION_THRESHOLD: float = 0.982
    DEFAULT_LIVENESS_THRESHOLD: float = 0.58
    DEFAULT_LATE_GRACE_MINUTES: int = 10
    
    # CORS
    CORS_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
        "http://localhost:8000",
    ]
    
    # SMTP Email Alerts (Preserved from legacy final.py)
    SMTP_SERVER: str = os.getenv("SMTP_SERVER", "smtp.gmail.com")
    SMTP_PORT: int = int(os.getenv("SMTP_PORT", "587"))
    SMTP_USER: str = os.getenv("SMTP_USER", "notifications@smartattend.ai")
    SMTP_PASSWORD: str = os.getenv("SMTP_PASSWORD", "")
    SMTP_ENABLED: bool = os.getenv("SMTP_ENABLED", "false").lower() == "true"


settings = Settings()
