# Emotion Detection Using AI

This project implements a multimodal **Emotion Detection AI** system that analyzes:
- **Text** using NLP emotion classification
- **Speech** using audio feature-based ML heuristics
- **Facial Expressions** using computer vision emotion analysis

It supports practical use cases in:
- Customer service sentiment and satisfaction tracking
- Healthcare mental well-being monitoring
- Social media public sentiment analysis

## Tech Stack
- FastAPI backend for API and web serving
- Transformers (Hugging Face) for text emotion inference
- Librosa + SoundFile for speech signal processing
- DeepFace + OpenCV for facial expression analysis
- Weighted multimodal late-fusion for final emotion decision

## Project Structure

emotion_detection_ai/
- app/
  - schemas.py
  - services/
    - text_emotion.py
    - speech_emotion.py
    - facial_emotion.py
    - fusion.py
- templates/
  - index.html
- main.py
- train_text_model.py
- requirements.txt

## Setup

1. Create a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Run the app:

```powershell
uvicorn main:app --reload
```

4. Open the app:
- http://127.0.0.1:8000
- API docs: http://127.0.0.1:8000/docs

## API Endpoints
- `POST /analyze/text`
- `POST /analyze/speech`
- `POST /analyze/facial`
- `POST /analyze/multimodal`
- `GET /health`

## Notes on Model Training

The repository includes `train_text_model.py` to fine-tune a text emotion model on a large labeled dataset (`go_emotions`).

For production-grade performance:
- Train each modality on dedicated emotion datasets
- Evaluate class imbalance and bias
- Use domain-specific calibration (e.g., healthcare vs social media)
- Add robust metrics (F1, ROC-AUC, confusion matrix)

## Example Applications
- **Customer Service**: detect dissatisfaction early in support interactions
- **Healthcare**: monitor emotional trend changes in patient journals or voice diaries
- **Social Media**: identify mood shifts and public response at scale
