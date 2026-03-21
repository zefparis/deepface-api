import asyncio
from functools import partial
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field
from typing import Optional
import base64
import numpy as np
import cv2
import time

from app.core.circuit_breaker import deepface_breaker
from app.core.logger import log_request
from app.core.rate_limit import rate_limiter
from app.core.security import require_api_key
from app.core.config import settings
from app.services.deepface_service import get_deepface_service

router = APIRouter()
TIMING_FLOOR_MS = 200


def _client_ip(request: Request) -> str:
    return request.client.host if request.client else "unknown"


def _request_id(request: Request) -> str:
    state = getattr(request, "state", None)
    return getattr(state, "request_id", "unknown")


async def with_timing_floor(coro, floor_ms: int = TIMING_FLOOR_MS):
    """Ensures response never returns faster than floor_ms."""
    t0 = time.perf_counter()
    try:
        return await coro
    finally:
        elapsed = (time.perf_counter() - t0) * 1000
        if elapsed < floor_ms:
            await asyncio.sleep((floor_ms - elapsed) / 1000)


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
    confidence: float = 0.0
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

@router.post("", response_model=AnalyzeResponse, dependencies=[Depends(require_api_key)])
async def analyze(req: AnalyzeRequest, request: Request):
    """
    Analyze a face image:
    - Detect presence
    - Liveness check (anti-spoofing)
    - Age / gender / emotion
    - Optional: embedding vector
    """
    try:
        client_ip = _client_ip(request)
        request_id = _request_id(request)
        t0 = time.time()
        if not rate_limiter.is_allowed(f"{client_ip}:/analyze", 30, 60):
            log_request(
                request_id=request_id,
                endpoint="/analyze",
                client_ip=client_ip,
                processing_ms=int((time.time() - t0) * 1000),
                face_detected=False,
                liveness=None,
                status=429,
            )
            raise HTTPException(status_code=429, detail="Rate limit exceeded")
        if not deepface_breaker.call_allowed():
            log_request(
                request_id=request_id,
                endpoint="/analyze",
                client_ip=client_ip,
                processing_ms=int((time.time() - t0) * 1000),
                face_detected=False,
                liveness=None,
                status=503,
            )
            raise HTTPException(
                status_code=503,
                detail="Service temporarily unavailable — circuit open",
            )

        async def do_analyze():
            img = _decode_image(req.image_b64)
            service = get_deepface_service()
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                None,
                partial(
                    service.analyze,
                    img,
                    req.model or settings.DEFAULT_MODEL,
                    req.extract_embedding,
                ),
            )

        result = await with_timing_floor(do_analyze())
        deepface_breaker.record_success()
        result.pop("_internal_status", None)
        processing_ms = int((time.time() - t0) * 1000)
        log_request(
            request_id=request_id,
            endpoint="/analyze",
            client_ip=client_ip,
            processing_ms=processing_ms,
            face_detected=bool(result.get("face_detected", False)),
            liveness=result.get("liveness"),
            status=200,
        )
        return AnalyzeResponse(
            **result,
            processing_ms=processing_ms,
        )
    except ValueError as e:
        if str(e) not in {"Invalid base64 image", "Could not decode image"}:
            deepface_breaker.record_failure()
        log_request(
            request_id=_request_id(request),
            endpoint="/analyze",
            client_ip=_client_ip(request),
            processing_ms=int((time.time() - t0) * 1000),
            face_detected=False,
            liveness=None,
            status=400,
        )
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        deepface_breaker.record_failure()
        log_request(
            request_id=_request_id(request),
            endpoint="/analyze",
            client_ip=_client_ip(request),
            processing_ms=int((time.time() - t0) * 1000),
            face_detected=False,
            liveness=None,
            status=500,
        )
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@router.post("/verify", response_model=VerifyResponse, dependencies=[Depends(require_api_key)])
async def verify(req: VerifyRequest, request: Request):
    """
    Compare two face images — returns verified=True/False + confidence score.
    Used for: enrollment vs. live capture comparison.
    """
    try:
        client_ip = _client_ip(request)
        request_id = _request_id(request)
        t0 = time.time()
        if not rate_limiter.is_allowed(f"{client_ip}:/analyze/verify", 20, 60):
            log_request(
                request_id=request_id,
                endpoint="/analyze/verify",
                client_ip=client_ip,
                processing_ms=int((time.time() - t0) * 1000),
                face_detected=False,
                liveness=None,
                status=429,
            )
            raise HTTPException(status_code=429, detail="Rate limit exceeded")
        if not deepface_breaker.call_allowed():
            log_request(
                request_id=request_id,
                endpoint="/analyze/verify",
                client_ip=client_ip,
                processing_ms=int((time.time() - t0) * 1000),
                face_detected=False,
                liveness=None,
                status=503,
            )
            raise HTTPException(
                status_code=503,
                detail="Service temporarily unavailable — circuit open",
            )

        async def do_verify():
            img1 = _decode_image(req.image1_b64)
            img2 = _decode_image(req.image2_b64)
            service = get_deepface_service()
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                None,
                partial(
                    service.verify,
                    img1,
                    img2,
                    req.model or settings.DEFAULT_MODEL,
                ),
            )

        result = await with_timing_floor(do_verify())
        deepface_breaker.record_success()
        processing_ms = int((time.time() - t0) * 1000)
        log_request(
            request_id=request_id,
            endpoint="/analyze/verify",
            client_ip=client_ip,
            processing_ms=processing_ms,
            face_detected=True,
            liveness=None,
            status=200,
        )
        return VerifyResponse(
            **result,
            processing_ms=processing_ms,
        )
    except ValueError as e:
        if str(e) not in {"Invalid base64 image", "Could not decode image"}:
            deepface_breaker.record_failure()
        log_request(
            request_id=_request_id(request),
            endpoint="/analyze/verify",
            client_ip=_client_ip(request),
            processing_ms=int((time.time() - t0) * 1000),
            face_detected=False,
            liveness=None,
            status=400,
        )
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        deepface_breaker.record_failure()
        log_request(
            request_id=_request_id(request),
            endpoint="/analyze/verify",
            client_ip=_client_ip(request),
            processing_ms=int((time.time() - t0) * 1000),
            face_detected=False,
            liveness=None,
            status=500,
        )
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
