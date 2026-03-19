import asyncio
from functools import partial
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
import base64
import numpy as np
import cv2
import time

from app.core.security import require_api_key
from app.core.config import settings
from app.services.deepface_service import get_deepface_service

router = APIRouter()


# ─── Schemas ─────────────────────────────────────────────────────────────────

class AnalyzeRequest(BaseModel):
    """Single image analysis — liveness, attributes, embedding"""
    image_b64: str = Field(..., description="Base64 encoded image (JPEG/PNG)")
    extract_embedding: bool = Field(False, description="Return face embedding vector")
    model: Optional[str] = Field(None, description="Override default model")

class VerifyRequest(BaseModel):
    """Face verification — are these two images the same person?"""
    image1_b64: str
    image2_b64: str
    model: Optional[str] = None

class AnalyzeResponse(BaseModel):
    success: bool
    face_detected: bool
    liveness: Optional[bool] = None
    liveness_score: Optional[float] = None
    age: Optional[int] = None
    gender: Optional[str] = None
    emotion: Optional[str] = None
    embedding: Optional[list] = None
    processing_ms: int

class VerifyResponse(BaseModel):
    success: bool
    verified: bool
    confidence: float
    distance: float
    threshold: float
    model: str
    processing_ms: int


# ─── Routes ──────────────────────────────────────────────────────────────────

@router.post("/", response_model=AnalyzeResponse, dependencies=[Depends(require_api_key)])
async def analyze(req: AnalyzeRequest):
    """
    Analyze a face image:
    - Detect presence
    - Liveness check (anti-spoofing)
    - Age / gender / emotion
    - Optional: embedding vector
    """
    t0 = time.time()
    try:
        img = _decode_image(req.image_b64)
        service = get_deepface_service()
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None,
            partial(
                service.analyze,
                img,
                req.model or settings.DEFAULT_MODEL,
                req.extract_embedding,
            ),
        )
        return AnalyzeResponse(
            **result,
            processing_ms=int((time.time() - t0) * 1000)
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@router.post("/verify", response_model=VerifyResponse, dependencies=[Depends(require_api_key)])
async def verify(req: VerifyRequest):
    """
    Compare two face images — returns verified=True/False + confidence score.
    Used for: enrollment vs. live capture comparison.
    """
    t0 = time.time()
    try:
        img1 = _decode_image(req.image1_b64)
        img2 = _decode_image(req.image2_b64)
        service = get_deepface_service()
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None,
            partial(
                service.verify,
                img1,
                img2,
                req.model or settings.DEFAULT_MODEL,
            ),
        )
        return VerifyResponse(
            **result,
            processing_ms=int((time.time() - t0) * 1000)
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Verification failed: {str(e)}")


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _decode_image(b64: str) -> np.ndarray:
    try:
        # Strip data URL prefix if present (data:image/jpeg;base64,...)
        if "," in b64:
            b64 = b64.split(",")[1]
        img_bytes = base64.b64decode(b64)
        arr = np.frombuffer(img_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Could not decode image")
        return img
    except Exception:
        raise ValueError("Invalid base64 image")
