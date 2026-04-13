from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class TextEmotionRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Input text for emotion analysis")


class EmotionPrediction(BaseModel):
    label: str
    score: float


class EmotionResponse(BaseModel):
    modality: str
    top_emotion: str
    confidence: float
    predictions: List[EmotionPrediction]
    metadata: Dict[str, str] = Field(default_factory=dict)


class MultimodalEmotionResponse(BaseModel):
    top_emotion: str
    confidence: float
    source_weights: Dict[str, float]
    modality_results: Dict[str, EmotionResponse]
    insights: Dict[str, str] = Field(default_factory=dict)


class HealthResponse(BaseModel):
    status: str
    available_modalities: List[str]
