import uuid

from fastapi.responses import JSONResponse

from app.core.config import settings
from app.core.security import verify_hmac_signature


class RequestSecurityMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request_id = str(uuid.uuid4())
        state = scope.get("state")
        if not isinstance(state, dict):
            state = {}
            scope["state"] = state
        state["request_id"] = request_id

        method = scope.get("method", "").upper()
        inner_receive = receive

        if method == "POST" and settings.REQUIRE_HMAC:
            chunks = []
            while True:
                message = await receive()
                if message["type"] != "http.request":
                    continue
                chunks.append(message.get("body", b""))
                if not message.get("more_body", False):
                    break

            body = b"".join(chunks)
            headers = {
                key.decode("latin-1").lower(): value.decode("latin-1")
                for key, value in scope.get("headers", [])
            }
            if not verify_hmac_signature(
                body=body,
                timestamp=headers.get("x-timestamp", ""),
                signature=headers.get("x-signature", ""),
                secret=settings.HMAC_SECRET,
            ):
                response = JSONResponse(
                    status_code=403,
                    content={"detail": "Invalid or missing HMAC signature"},
                )
                response.headers["X-Request-ID"] = request_id
                await response(scope, receive, send)
                return

            sent = False

            async def replay_receive():
                nonlocal sent
                if not sent:
                    sent = True
                    return {"type": "http.request", "body": body, "more_body": False}
                return {"type": "http.request", "body": b"", "more_body": False}

            inner_receive = replay_receive

        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", []))
                headers.append((b"x-request-id", request_id.encode("utf-8")))
                message["headers"] = headers
            await send(message)

        await self.app(scope, inner_receive, send_wrapper)
