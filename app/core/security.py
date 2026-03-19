import hashlib
import hmac
import time

from fastapi import Security, HTTPException, status
from fastapi.security.api_key import APIKeyHeader
from app.core.config import settings

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def verify_hmac_signature(
    body: bytes,
    timestamp: str,
    signature: str,
    secret: str,
) -> bool:
    """
    Verify HMAC-SHA256 request signature.
    Used for server-to-server calls (hybrid-vector-api → deepface-api)
    Header: X-Signature: sha256=<hmac>
    Header: X-Timestamp: <unix_timestamp>
    """
    try:
        ts = int(timestamp)
        if abs(time.time() - ts) > 300:
            return False
    except (ValueError, TypeError):
        return False

    if not secret or not signature:
        return False

    mac = hmac.new(
        secret.encode(),
        f"{timestamp}.".encode() + body,
        hashlib.sha256,
    )
    expected = "sha256=" + mac.hexdigest()
    return hmac.compare_digest(expected, signature)


async def require_api_key(api_key: str = Security(api_key_header)):
    if not hmac.compare_digest(api_key or "", settings.API_SECRET_KEY):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid or missing API key"
        )
    return api_key
