from __future__ import annotations

from typing import Dict, List


def fuse_emotions(modality_predictions: Dict[str, List[Dict[str, float]]]) -> Dict[str, object]:
    """Weighted late-fusion over available modalities."""
    modality_weights = {
        "text": 0.35,
        "speech": 0.3,
        "facial": 0.35,
    }

    active_weights: Dict[str, float] = {}
    for modality in modality_predictions:
        active_weights[modality] = modality_weights.get(modality, 0.25)

    total_weight = sum(active_weights.values()) or 1.0
    normalized_weights = {
        k: round(v / total_weight, 4) for k, v in active_weights.items()
    }

    pooled_scores: Dict[str, float] = {}
    for modality, preds in modality_predictions.items():
        weight = normalized_weights[modality]
        for pred in preds:
            pooled_scores[pred["label"]] = pooled_scores.get(pred["label"], 0.0) + pred["score"] * weight

    ranked = sorted(pooled_scores.items(), key=lambda x: x[1], reverse=True)
    top_label, top_score = ranked[0] if ranked else ("neutral", 0.0)

    return {
        "top_emotion": top_label,
        "confidence": round(float(top_score), 4),
        "weights": normalized_weights,
        "ranked": [{"label": k, "score": round(float(v), 4)} for k, v in ranked],
    }
