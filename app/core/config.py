import json
from typing import List, Union

from pydantic import field_validator
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    ENV: str = "development"
    API_SECRET_KEY: str = "changeme"
    ALLOWED_ORIGINS: Union[str, List[str]] = "*"

    @field_validator("ALLOWED_ORIGINS", mode="before")
    @classmethod
    def parse_origins(cls, v):
        if isinstance(v, list):
            return v
        if not isinstance(v, str):
            return ["*"]

        value = v.strip()
        if not value:
            return ["*"]

        if value.startswith("["):
            try:
                parsed = json.loads(value)
                if isinstance(parsed, list):
                    return parsed
            except Exception:
                pass

        if value == "*":
            return ["*"]

        return [origin.strip().strip('"').strip("'") for origin in value.split(",") if origin.strip()]

    # DeepFace defaults
    DEFAULT_MODEL: str = "ArcFace"
    DEFAULT_DETECTOR: str = "opencv"
    DEFAULT_METRIC: str = "cosine"
    ANTI_SPOOFING: bool = False
    LIVENESS_THRESHOLD: float = 0.5
    VERIFICATION_THRESHOLD: float = 0.68
    ENABLE_DEEPFACE_WARMUP: bool = False
    HMAC_SECRET: str = ""
    REQUIRE_HMAC: bool = False

    class Config:
        env_file = ".env"

settings = Settings()