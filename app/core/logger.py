import json
import time
from typing import Any


def log(level: str, event: str, **kwargs: Any) -> None:
    """
    NDJSON structured logging — same philosophy as HCS-U7.
    """
    entry = {
        "ts": time.time(),
        "level": level,
        "event": event,
        "service": "deepface-api",
        **kwargs,
    }
    print(json.dumps(entry), flush=True)


def log_request(
    request_id: str,
    endpoint: str,
    client_ip: str,
    processing_ms: int,
    face_detected: bool,
    liveness: bool | None,
    status: int,
) -> None:
    log(
        "INFO",
        "request_complete",
        request_id=request_id,
        endpoint=endpoint,
        client_ip=client_ip,
        processing_ms=processing_ms,
        face_detected=face_detected,
        liveness=liveness,
        status=status,
    )
