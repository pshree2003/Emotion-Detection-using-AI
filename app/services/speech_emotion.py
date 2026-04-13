from __future__ import annotations

import tempfile
from typing import Dict, List


class SpeechEmotionAnalyzer:
    """Speech emotion inference based on acoustic features and heuristics."""

    def __init__(self) -> None:
        self.sample_rate = 22050

    def _heuristic_scores(self, energy: float, pitch: float) -> Dict[str, float]:
        # Simple interpretable fallback when a trained SER model is unavailable.
        joy = min(1.0, (energy * 0.6) + (pitch * 0.4))
        sadness = min(1.0, max(0.0, (1.0 - energy) * 0.7 + (1.0 - pitch) * 0.3))
        anger = min(1.0, max(0.0, energy * 0.8 + (1.0 - pitch) * 0.2))
        neutral = max(0.05, 1.0 - max(joy, sadness, anger) * 0.8)

        total = joy + sadness + anger + neutral
        return {
            "joy": round(joy / total, 4),
            "sadness": round(sadness / total, 4),
            "anger": round(anger / total, 4),
            "neutral": round(neutral / total, 4),
        }

    def analyze(self, audio_bytes: bytes) -> List[Dict[str, float]]:
        try:
            import librosa
            import numpy as np
            import soundfile as sf

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
                tmp.write(audio_bytes)
                tmp.flush()
                signal, sr = sf.read(tmp.name)
                if signal.ndim > 1:
                    signal = signal.mean(axis=1)
                if sr != self.sample_rate:
                    signal = librosa.resample(signal, orig_sr=sr, target_sr=self.sample_rate)

                rms = float(np.mean(librosa.feature.rms(y=signal)))
                pitches, magnitudes = librosa.piptrack(y=signal, sr=self.sample_rate)
                pitch_values = pitches[magnitudes > np.median(magnitudes)]
                mean_pitch_hz = float(np.mean(pitch_values)) if len(pitch_values) else 120.0

                # Normalize rough ranges to 0..1.
                norm_energy = min(1.0, max(0.0, rms * 10.0))
                norm_pitch = min(1.0, max(0.0, (mean_pitch_hz - 75.0) / 225.0))

                scores = self._heuristic_scores(norm_energy, norm_pitch)
                return sorted(
                    (
                        {"label": label, "score": score}
                        for label, score in scores.items()
                    ),
                    key=lambda x: x["score"],
                    reverse=True,
                )
        except Exception:
            # Conservative fallback when audio parsing fails.
            return [
                {"label": "neutral", "score": 0.55},
                {"label": "sadness", "score": 0.2},
                {"label": "joy", "score": 0.15},
                {"label": "anger", "score": 0.1},
            ]
