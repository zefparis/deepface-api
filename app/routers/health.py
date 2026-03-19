import time

from fastapi import APIRouter

from app.core.config import settings
from app.services.deepface_service import get_deepface_service

router = APIRouter()
START_TIME = time.time()

@router.get("")
async def health(detailed: bool = False):
    base = {
        "status": "ok",
        "service": "deepface-api",
        "version": "2.0.0",
        "timestamp": time.time(),
    }

    if detailed:
        service = get_deepface_service()
        base["models_loaded"] = service._deepface is not None
        base["detector"] = settings.DEFAULT_DETECTOR
        base["model"] = settings.DEFAULT_MODEL
        base["uptime_seconds"] = int(time.time() - START_TIME)

    return base
