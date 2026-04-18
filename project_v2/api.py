"""
api.py
------
FastAPI inference server. Exposes one endpoint:

  POST /analyze
  Body:  { "text": "comment string here" }
  Returns: { "labels": [...], "probabilities": {...}, "is_toxic": bool }

Run locally:
  python api.py

Run via Docker:
  docker-compose up

The Chrome extension calls this at http://localhost:8000/analyze
CORS is enabled for all origins so the browser extension can reach it.
"""

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from inference import predict

app = FastAPI(
    title="Toxic Comment Classifier API",
    description="Multi-label toxicity detection using fine-tuned DistilBERT",
    version="1.0.0",
)

# Allow browser extension to call the API from any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)


class CommentRequest(BaseModel):
    text: str


class CommentResponse(BaseModel):
    labels: list[str]
    probabilities: dict[str, float]
    is_toxic: bool


@app.get("/")
def health():
    return {"status": "ok", "model": "DistilBERT", "labels": 6}


@app.post("/analyze", response_model=CommentResponse)
def analyze(req: CommentRequest):
    """
    Analyze a single comment for toxicity.
    Returns which of the 6 labels fired and all raw probabilities.
    """
    result = predict(req.text)
    return result


@app.post("/analyze_batch")
def analyze_batch(texts: list[str]):
    """
    Analyze multiple comments in one request.
    Used by the extension to batch-process all comments on a page.
    """
    return [predict(t) for t in texts]


if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)
