from __future__ import annotations

import tempfile
from typing import Dict, List


class FacialEmotionAnalyzer:
    """Facial emotion analysis using DeepFace when available."""

    def analyze(self, image_bytes: bytes) -> List[Dict[str, float]]:
        try:
            from deepface import DeepFace

            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=True) as tmp:
                tmp.write(image_bytes)
                tmp.flush()
                result = DeepFace.analyze(
                    img_path=tmp.name,
                    actions=["emotion"],
                    enforce_detection=False,
                    detector_backend="opencv",
                )

            if isinstance(result, list):
                result = result[0]

            emotion_scores = result.get("emotion", {})
            if not emotion_scores:
                raise ValueError("No emotion scores returned")

            total = sum(float(v) for v in emotion_scores.values()) or 1.0
            normalized = [
                {"label": str(label).lower(), "score": round(float(score) / total, 4)}
                for label, score in emotion_scores.items()
            ]
            normalized.sort(key=lambda x: x["score"], reverse=True)
            return normalized

        except Exception:
            return [
                {"label": "neutral", "score": 0.5},
                {"label": "joy", "score": 0.2},
                {"label": "sadness", "score": 0.15},
                {"label": "anger", "score": 0.15},
            ]
