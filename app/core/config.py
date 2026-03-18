from pydantic_settings import BaseSettings
from typing import List

class Settings(BaseSettings):
    ENV: str = "development"
    API_SECRET_KEY: str = "changeme"
    ALLOWED_ORIGINS: List[str] = ["*"]
    
    # DeepFace defaults
    DEFAULT_MODEL: str = "ArcFace"          # ArcFace | VGG-Face | Facenet | DeepID
    DEFAULT_DETECTOR: str = "retinaface"    # retinaface | mtcnn | opencv
    DEFAULT_METRIC: str = "cosine"          # cosine | euclidean | euclidean_l2
    LIVENESS_THRESHOLD: float = 0.5
    VERIFICATION_THRESHOLD: float = 0.68    # ArcFace cosine

    class Config:
        env_file = ".env"

settings = Settings()
