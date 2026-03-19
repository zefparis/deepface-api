# Sovereign Deployment Guide

## deepface-api v2.0

Production profile for a sovereign biometric microservice.

## Local Docker

```bash
docker build -t deepface-api:2.0 .
docker run -d \
  -p 8000:8000 \
  -e API_SECRET_KEY=<key> \
  -e HMAC_SECRET=<secret> \
  -e REQUIRE_HMAC=true \
  deepface-api:2.0
```

## Air-gapped deployment

- All models are pre-downloaded at build time.
- Runtime does not require external internet access.
- Detector backend defaults to `opencv` for speed.
- High-security deployments can override the detector via `DEEPFACE_DETECTOR=retinaface`.

## Security checklist

- `REQUIRE_HMAC=true` in production
- `HMAC_SECRET` is set to a strong random value
- `API_SECRET_KEY` is at least 32 characters
- `ALLOWED_ORIGINS` is restricted to trusted domains
- Rate limiting is enabled by default
- Request IDs are emitted on every response
- No external telemetry or runtime model downloads

## Notes

- The in-memory rate limiter and circuit breaker are process-local.
- Keep the deployment on a single worker for low-memory environments.
- Enable `ENABLE_DEEPFACE_WARMUP=true` only on hosts with enough RAM for startup preloading.
