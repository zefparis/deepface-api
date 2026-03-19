import numpy as np
from typing import Optional, Dict, Any
from app.core.config import settings


class DeepFaceService:
    """
    Wrapper autour de DeepFace.
    Lazy-loaded : DeepFace est importé au premier appel pour éviter
    de ralentir le démarrage du serveur.
    """

    def __init__(self):
        self._deepface = None

    def _get_deepface(self):
        if self._deepface is None:
            from deepface import DeepFace
            self._deepface = DeepFace
        return self._deepface

    def analyze(
        self,
        img: np.ndarray,
        model: str = "ArcFace",
        extract_embedding: bool = False,
    ) -> Dict[str, Any]:
        DeepFace = self._get_deepface()

        try:
            faces = DeepFace.extract_faces(
                img_path=img,
                detector_backend=settings.DEFAULT_DETECTOR,
                anti_spoofing=True,
                enforce_detection=True,
            )
        except Exception:
            return {
                "success": True,
                "face_detected": False,
                "liveness": None,
                "liveness_score": None,
                "age": None,
                "gender": None,
                "emotion": None,
                "embedding": None,
            }

        if not faces:
            return {
                "success": True,
                "face_detected": False,
                "liveness": None,
                "liveness_score": None,
                "age": None,
                "gender": None,
                "emotion": None,
                "embedding": None,
            }

        face = faces[0]
        face_detected = True
        antispoof_score = face.get("antispoof_score", 0.5)
        liveness_score = float(0.5 if antispoof_score is None else antispoof_score)
        liveness = liveness_score >= settings.LIVENESS_THRESHOLD

        age = gender = emotion = None
        try:
            attrs = DeepFace.analyze(
                img_path=img,
                actions=["age", "gender", "emotion"],
                detector_backend=settings.DEFAULT_DETECTOR,
                enforce_detection=False,
                silent=True,
            )
            attr = attrs[0] if isinstance(attrs, list) else attrs
            age = int(attr.get("age", 0))
            gender = attr.get("dominant_gender", None)
            emotion = attr.get("dominant_emotion", None)
        except Exception:
            pass

        embedding = None
        if extract_embedding:
            try:
                emb_result = DeepFace.represent(
                    img_path=img,
                    model_name=model,
                    detector_backend=settings.DEFAULT_DETECTOR,
                    enforce_detection=False,
                )
                embedding = emb_result[0]["embedding"] if emb_result else None
                if hasattr(embedding, "tolist"):
                    embedding = embedding.tolist()
            except Exception:
                pass

        return {
            "success": True,
            "face_detected": face_detected,
            "liveness": liveness,
            "liveness_score": liveness_score,
            "age": age,
            "gender": gender,
            "emotion": emotion,
            "embedding": embedding,
        }

    def warmup(self) -> None:
        """
        Pre-load all ML models into memory on startup.
        Called once — eliminates cold-start latency on first request.
        """
        try:
            DeepFace = self._get_deepface()
            dummy = np.zeros((100, 100, 3), dtype=np.uint8)

            try:
                DeepFace.extract_faces(
                    img_path=dummy,
                    detector_backend=settings.DEFAULT_DETECTOR,
                    anti_spoofing=False,
                    enforce_detection=False,
                )
            except Exception:
                pass

            try:
                DeepFace.build_model("ArcFace")
            except Exception:
                pass

            print("✅ DeepFace models warmed up")
        except Exception as e:
            print(f"⚠️  Warmup failed: {e}")

    def verify(
        self,
        img1: np.ndarray,
        img2: np.ndarray,
        model: str = "ArcFace",
    ) -> Dict[str, Any]:
        DeepFace = self._get_deepface()

        result = DeepFace.verify(
            img1_path=img1,
            img2_path=img2,
            model_name=model,
            detector_backend=settings.DEFAULT_DETECTOR,
            distance_metric=settings.DEFAULT_METRIC,
            enforce_detection=True,
            silent=True,
        )

        distance = float(result["distance"])
        threshold = float(result["threshold"])
        verified = bool(result["verified"])
        # confidence : inversion normalisée de la distance
        confidence = round(max(0.0, 1.0 - distance / threshold), 4) if threshold > 0 else 0.0

        return {
            "success": True,
            "verified": verified,
            "confidence": confidence,
            "distance": distance,
            "threshold": threshold,
            "model": model,
        }


_instance: Optional["DeepFaceService"] = None


def get_deepface_service() -> "DeepFaceService":
    global _instance
    if _instance is None:
        _instance = DeepFaceService()
    return _instance
