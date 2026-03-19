# deepface-api

Microservice biométrique indépendant basé sur [DeepFace](https://github.com/serengil/deepface).

## Endpoints

| Méthode | Route | Description |
|---------|-------|-------------|
| GET | `/health` | Status du service |
| POST | `/analyze` | Analyse un visage (liveness, attributs, embedding) |
| POST | `/analyze/verify` | Compare deux visages |

## Auth

Header requis sur tous les endpoints sauf `/health` :
```
X-API-Key: <votre_clé>
```

En production, vous pouvez aussi activer la signature HMAC serveur-à-serveur :
```bash
X-Timestamp: <unix_timestamp>
X-Signature: sha256=<hmac_sha256>
```

Le service renvoie aussi `X-Request-ID` sur chaque réponse.

## Health

- `GET /health` — status rapide
- `GET /health?detailed=true` — statut détaillé, modèles chargés, détecteur, uptime

## Démarrage local

```bash
cp .env.example .env
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Swagger UI disponible sur : http://localhost:8000/docs

## Deploy Render

```bash
# Push le repo, connecter à Render, utiliser render.yaml
# Le Dockerfile pré-télécharge les modèles ArcFace au build
# Default prod: opencv detector, HMAC activable, warmup désactivé par défaut
```

## Intégration HCS-U7 (TypeScript)

```typescript
const res = await fetch(`${process.env.DEEPFACE_API_URL}/analyze`, {
  method: "POST",
  headers: {
    "Content-Type": "application/json",
    "X-API-Key": process.env.DEEPFACE_API_KEY,
  },
  body: JSON.stringify({ image_b64: capturedFrame }),
})
const { face_detected, liveness, liveness_score, age, gender, emotion } = await res.json()
```

## Stack

- **FastAPI** + **Uvicorn**
- **DeepFace** (ArcFace par défaut, backends interchangeables)
- **OpenCV** headless
- **Docker** ready, **Render** ready
