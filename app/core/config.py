from pydantic_settings import BaseSettings
from typing import List, Union
from pydantic import field_validator

class Settings(BaseSettings):
    ENV: str = "development"
    API_SECRET_KEY: str = "changeme"
    ALLOWED_ORIGINS: Union[str, List[str]] = "*"

    @field_validator("ALLOWED_ORIGINS", mode="before")
    @classmethod
    def parse_origins(cls, v):
        if isinstance(v, list):
            return v
        return [origin.strip() for origin in v.split(",")]

    # DeepFace defaults
    DEFAULT_MODEL: str = "ArcFace"
    DEFAULT_DETECTOR: str = "opencv"
    DEFAULT_METRIC: str = "cosine"
    LIVENESS_THRESHOLD: float = 0.5
    VERIFICATION_THRESHOLD: float = 0.68
    ENABLE_DEEPFACE_WARMUP: bool = False

    class Config:
        env_file = ".env"

settings = Settings()