import asyncio
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import analyze, health
from app.core.config import settings
from app.services.deepface_service import get_deepface_service

app = FastAPI(
    title="DeepFace API",
    description="Microservice biométrique — Face recognition, liveness, anti-spoofing",
    version="1.0.0",
    docs_url="/docs" if settings.ENV != "production" else None,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    if settings.ENABLE_DEEPFACE_WARMUP:
        service = get_deepface_service()
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, service.warmup)

app.include_router(health.router, prefix="/health", tags=["health"])
app.include_router(analyze.router, prefix="/analyze", tags=["analyze"])
