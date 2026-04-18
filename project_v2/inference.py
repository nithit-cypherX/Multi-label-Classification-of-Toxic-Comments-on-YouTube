"""
inference.py
------------
Loads the trained DistilBERT classifier and thresholds once at startup.
Exposes a single predict(text) function used by the API server.

Loading strategy:
  - Backbone: DistilBertModel.from_pretrained(BERT_PATH)  ← saved with save_pretrained()
  - Head:     torch.load(head_weights.pth)                ← saved as state_dict dict
  - Thresholds: pickle.load(bert_thresholds.pkl)          ← per-label optimal thresholds
"""

import pickle
import numpy as np
import torch
from transformers import DistilBertTokenizer, DistilBertModel

from config import BERT_PATH, THRESHOLDS_PATH, LABEL_COLS, MAX_SEQ_LEN, DEVICE
from model import DistilBERTClassifier
from preprocess import preprocess

# ── Load model once at module import time ──────────────────────────────────────
print(f"[inference] Loading DistilBERT from {BERT_PATH} ...")

_model = DistilBERTClassifier().to(DEVICE)
_model.bert = DistilBertModel.from_pretrained(BERT_PATH)

head = torch.load(f'{BERT_PATH}/head_weights.pth', map_location=DEVICE)
_model.pre_classifier.load_state_dict(head['pre_classifier'])
_model.classifier.load_state_dict(head['classifier'])
_model.eval()

_tokenizer = DistilBertTokenizer.from_pretrained(BERT_PATH)

with open(THRESHOLDS_PATH, 'rb') as f:
    _thresholds = pickle.load(f)   # shape: (6,) float array

# Override thresholds for better sensitivity on borderline comments.
# Original tuned thresholds were optimised for precision on the Jigsaw test set.
# Lower values catch more subtle toxicity at the cost of more false positives.
# Format: [toxic, severe_toxic, obscene, threat, insult, identity_hate]
import numpy as np
_thresholds = np.array([0.30, 0.007, 0.025, 0.014, 0.013, 0.018])

print(f"[inference] Model ready. Thresholds: { {l: round(float(t), 2) for l, t in zip(LABEL_COLS, _thresholds)} }")


# ── Public API ─────────────────────────────────────────────────────────────────

def predict(text: str) -> dict:
    """
    Run DistilBERT inference on a single comment string.

    Args:
        text: raw comment text (no preprocessing needed by the caller)

    Returns:
        {
          "labels":       ["toxic", "insult"],   # labels that fired above threshold
          "probabilities": {"toxic": 0.87, "insult": 0.72, ...},  # all 6 probs
          "is_toxic":     True   # True if ANY label fired
        }
    """
    if not text or not text.strip():
        return {
            "labels": [],
            "probabilities": {l: 0.0 for l in LABEL_COLS},
            "is_toxic": False,
        }

    # Preprocess in bert mode (light cleaning, preserve grammar)
    clean = preprocess(text, mode='bert')

    encoding = _tokenizer(
        clean,
        max_length=MAX_SEQ_LEN,
        padding='max_length',
        truncation=True,
        return_tensors='pt',
    )

    input_ids      = encoding['input_ids'].to(DEVICE)
    attention_mask = encoding['attention_mask'].to(DEVICE)

    with torch.no_grad():
        logits = _model(input_ids=input_ids, attention_mask=attention_mask)
        probs  = torch.sigmoid(logits).cpu().numpy()[0]   # shape: (6,)

    # Apply per-label tuned thresholds
    fired_labels = [
        label for label, prob, threshold in zip(LABEL_COLS, probs, _thresholds)
        if prob >= threshold
    ]

    return {
        "labels":        fired_labels,
        "probabilities": {l: round(float(p), 4) for l, p in zip(LABEL_COLS, probs)},
        "is_toxic":      len(fired_labels) > 0,
    }