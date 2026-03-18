"""
Tests d'intégration — à lancer avec : pytest tests/
Requiert un serveur local : uvicorn app.main:app --reload
"""
import base64
import httpx
import numpy as np
import cv2
import pytest

BASE_URL = "http://localhost:8000"
API_KEY = "dev-secret-change-in-prod"
HEADERS = {"X-API-Key": API_KEY}


def _encode_image(path: str) -> str:
    """Encode une image en base64"""
    img = cv2.imread(path)
    _, buffer = cv2.imencode(".jpg", img)
    return base64.b64encode(buffer).decode("utf-8")


def _fake_image_b64() -> str:
    """Image synthétique 224x224 pour test sans vraie photo"""
    img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    _, buffer = cv2.imencode(".jpg", img)
    return base64.b64encode(buffer).decode("utf-8")


# ─── Health ──────────────────────────────────────────────────────────────────

def test_health():
    r = httpx.get(f"{BASE_URL}/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


# ─── Auth ─────────────────────────────────────────────────────────────────────

def test_no_api_key_rejected():
    r = httpx.post(f"{BASE_URL}/analyze", json={"image_b64": "x"})
    assert r.status_code == 403


# ─── Analyze ─────────────────────────────────────────────────────────────────

def test_analyze_invalid_image():
    r = httpx.post(
        f"{BASE_URL}/analyze",
        json={"image_b64": "notbase64!!"},
        headers=HEADERS,
    )
    assert r.status_code == 400


def test_analyze_no_face():
    """Image valide mais sans visage — face_detected doit être False"""
    r = httpx.post(
        f"{BASE_URL}/analyze",
        json={"image_b64": _fake_image_b64()},
        headers=HEADERS,
    )
    assert r.status_code == 200
    data = r.json()
    assert data["success"] is True
    assert data["face_detected"] is False


# ─── Exemple d'appel depuis HCS-U7 (TypeScript) ──────────────────────────────
"""
// Dans ton backend HCS-U7 — un seul fetch, zéro dépendance Python

const response = await fetch(`${process.env.DEEPFACE_API_URL}/analyze`, {
  method: "POST",
  headers: {
    "Content-Type": "application/json",
    "X-API-Key": process.env.DEEPFACE_API_KEY,
  },
  body: JSON.stringify({ image_b64: capturedFrame }),
})

const {
  face_detected,
  liveness,
  liveness_score,
  age,
  gender,
  emotion,
} = await response.json()

// Intégrer au trust score HCS-U7
const biometricBonus = liveness ? 0.15 : 0
"""
