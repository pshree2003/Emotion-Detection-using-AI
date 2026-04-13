from __future__ import annotations

from typing import Dict, List


class TextEmotionAnalyzer:
    """Text emotion analyzer with transformer-first strategy and rule fallback."""

    LABEL_MAP = {
        "joy": ["joy", "happy", "optimism", "amusement", "love"],
        "sadness": ["sadness", "grief", "disappointment", "remorse"],
        "anger": ["anger", "annoyance", "disapproval"],
        "fear": ["fear", "nervousness"],
        "surprise": ["surprise", "realization", "curiosity"],
        "neutral": ["neutral"],
    }

    POSITIVE_WORDS = {
        "great", "excellent", "good", "love", "happy", "relaxed", "amazing", "wonderful"
    }
    NEGATIVE_WORDS = {
        "sad", "bad", "terrible", "angry", "upset", "depressed", "worried", "anxious"
    }

    def __init__(self) -> None:
        self._pipeline = None
        self.model_name = "j-hartmann/emotion-english-distilroberta-base"
        self._load_pipeline()

    def _load_pipeline(self) -> None:
        try:
            from transformers import pipeline

            self._pipeline = pipeline(
                "text-classification",
                model=self.model_name,
                top_k=6,
                truncation=True,
            )
        except Exception:
            self._pipeline = None

    def _normalize_label(self, label: str) -> str:
        lowered = label.lower()
        for canonical, aliases in self.LABEL_MAP.items():
            if lowered in aliases:
                return canonical
        return lowered

    def _rule_based_scores(self, text: str) -> Dict[str, float]:
        tokens = [t.strip(".,!?;:\"'()[]{}") for t in text.lower().split()]
        pos_hits = sum(1 for t in tokens if t in self.POSITIVE_WORDS)
        neg_hits = sum(1 for t in tokens if t in self.NEGATIVE_WORDS)

        if pos_hits == 0 and neg_hits == 0:
            return {
                "neutral": 0.6,
                "joy": 0.2,
                "sadness": 0.1,
                "anger": 0.1,
            }

        total = max(pos_hits + neg_hits, 1)
        joy_score = pos_hits / total
        sadness_score = neg_hits / total * 0.7
        anger_score = neg_hits / total * 0.3

        neutral_score = max(0.05, 1.0 - (joy_score + sadness_score + anger_score))

        return {
            "joy": round(joy_score, 4),
            "sadness": round(sadness_score, 4),
            "anger": round(anger_score, 4),
            "neutral": round(neutral_score, 4),
        }

    def analyze(self, text: str) -> List[Dict[str, float]]:
        if self._pipeline is not None:
            try:
                raw = self._pipeline(text)[0]
                aggregate: Dict[str, float] = {}
                for item in raw:
                    label = self._normalize_label(item["label"])
                    aggregate[label] = aggregate.get(label, 0.0) + float(item["score"])

                normalized = sorted(
                    (
                        {"label": k, "score": round(v, 4)}
                        for k, v in aggregate.items()
                    ),
                    key=lambda x: x["score"],
                    reverse=True,
                )
                return normalized
            except Exception:
                pass

        fallback_scores = self._rule_based_scores(text)
        return sorted(
            (
                {"label": label, "score": score}
                for label, score in fallback_scores.items()
            ),
            key=lambda x: x["score"],
            reverse=True,
        )
