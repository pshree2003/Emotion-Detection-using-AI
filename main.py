from __future__ import annotations

from typing import Dict, Optional

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi import Request

from app.schemas import (
    EmotionPrediction,
    EmotionResponse,
    HealthResponse,
    MultimodalEmotionResponse,
    TextEmotionRequest,
)
from app.services.facial_emotion import FacialEmotionAnalyzer
from app.services.fusion import fuse_emotions
from app.services.speech_emotion import SpeechEmotionAnalyzer
from app.services.text_emotion import TextEmotionAnalyzer


app = FastAPI(
    title="Emotion Detection AI",
    description="Multimodal emotion detection for text, speech, and facial inputs",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

templates = Jinja2Templates(directory="templates")

text_analyzer = TextEmotionAnalyzer()
speech_analyzer = SpeechEmotionAnalyzer()
facial_analyzer = FacialEmotionAnalyzer()


def _to_emotion_response(modality: str, predictions, metadata: Optional[Dict[str, str]] = None) -> EmotionResponse:
    top = predictions[0] if predictions else {"label": "neutral", "score": 0.0}
    return EmotionResponse(
        modality=modality,
        top_emotion=top["label"],
        confidence=round(float(top["score"]), 4),
        predictions=[EmotionPrediction(label=p["label"], score=float(p["score"])) for p in predictions],
        metadata=metadata or {},
    )


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(request=request, name="index.html", context={})


@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="ok",
        available_modalities=["text", "speech", "facial"],
    )


@app.post("/analyze/text", response_model=EmotionResponse)
async def analyze_text(payload: TextEmotionRequest):
    predictions = text_analyzer.analyze(payload.text)
    return _to_emotion_response(
        modality="text",
        predictions=predictions,
        metadata={"input_length": str(len(payload.text))},
    )


@app.post("/analyze/speech", response_model=EmotionResponse)
async def analyze_speech(audio: UploadFile = File(...)):
    audio_bytes = await audio.read()
    predictions = speech_analyzer.analyze(audio_bytes)
    return _to_emotion_response(
        modality="speech",
        predictions=predictions,
        metadata={"filename": audio.filename or "uploaded_audio"},
    )


@app.post("/analyze/facial", response_model=EmotionResponse)
async def analyze_facial(image: UploadFile = File(...)):
    image_bytes = await image.read()
    predictions = facial_analyzer.analyze(image_bytes)
    return _to_emotion_response(
        modality="facial",
        predictions=predictions,
        metadata={"filename": image.filename or "uploaded_image"},
    )


@app.post("/analyze/multimodal", response_model=MultimodalEmotionResponse)
async def analyze_multimodal(
    text: Optional[str] = Form(default=None),
    audio: Optional[UploadFile] = File(default=None),
    image: Optional[UploadFile] = File(default=None),
):
    results = {}
    predictions = {}

    if text and text.strip():
        text_preds = text_analyzer.analyze(text)
        predictions["text"] = text_preds
        results["text"] = _to_emotion_response(
            modality="text",
            predictions=text_preds,
            metadata={"input_length": str(len(text))},
        )

    if audio is not None:
        audio_bytes = await audio.read()
        speech_preds = speech_analyzer.analyze(audio_bytes)
        predictions["speech"] = speech_preds
        results["speech"] = _to_emotion_response(
            modality="speech",
            predictions=speech_preds,
            metadata={"filename": audio.filename or "uploaded_audio"},
        )

    if image is not None:
        image_bytes = await image.read()
        face_preds = facial_analyzer.analyze(image_bytes)
        predictions["facial"] = face_preds
        results["facial"] = _to_emotion_response(
            modality="facial",
            predictions=face_preds,
            metadata={"filename": image.filename or "uploaded_image"},
        )

    if not predictions:
        neutral = EmotionResponse(
            modality="none",
            top_emotion="neutral",
            confidence=0.0,
            predictions=[EmotionPrediction(label="neutral", score=1.0)],
            metadata={"warning": "No modality data provided"},
        )
        return MultimodalEmotionResponse(
            top_emotion="neutral",
            confidence=0.0,
            source_weights={},
            modality_results={"none": neutral},
            insights={"note": "Provide at least one input modality for fusion."},
        )

    fused = fuse_emotions(predictions)

    return MultimodalEmotionResponse(
        top_emotion=fused["top_emotion"],
        confidence=float(fused["confidence"]),
        source_weights=fused["weights"],
        modality_results=results,
        insights={
            "application_customer_service": "Track customer satisfaction trends from chats/calls.",
            "application_healthcare": "Monitor emotional patterns over time for mental well-being support.",
            "application_social_media": "Estimate public sentiment from user generated content.",
            "training_note": "For production, train each modality on large labeled emotion datasets.",
        },
    )
