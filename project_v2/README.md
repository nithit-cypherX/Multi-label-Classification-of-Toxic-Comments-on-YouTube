# Multi-label Classification of Toxic Comments
### ITCS352 Deep Learning — Course Project

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Repository Structure](#2-repository-structure)
3. [Dataset](#3-dataset)
4. [Model Architecture](#4-model-architecture)
5. [Results](#5-results)
6. [How to Run](#6-how-to-run)
7. [Design Decisions](#7-design-decisions)
8. [Limitations & Future Work](#8-limitations--future-work)

---

## 1. Project Overview

This project builds a **production-style deep learning pipeline** for multi-label toxic comment classification. The goal is to detect multiple types of toxicity simultaneously — a single comment can be `toxic`, `obscene`, and `identity_hate` all at once.

**Task type:** Multi-label classification (each sample can have 0 or more of 6 labels)

**Three models trained across three tiers:**

| Tier | Model | Features | Test Macro-F1 |
|------|-------|----------|--------------|
| Classical ML | Logistic Regression + TF-IDF | Bag of bigrams | 0.5553 |
| Deep Learning | Bi-Directional LSTM | Trained embeddings | 0.6109 |
| Transformer | DistilBERT (fine-tuned) | Pre-trained contextual | **0.6527** |

The project is structured as modular Python files (not notebooks), following the same style as the course lab exercises. Each stage — preprocessing, dataset loading, model definition, training utilities, and the main training loop — lives in its own file.

---

## 2. Repository Structure

```
project/
├── config.py          # Single source of truth — all hyperparameters and paths
├── preprocess.py      # Dual-mode text cleaning (classical + bert)
├── dataset.py         # Jigsaw data loading, vocabulary, PyTorch Datasets
├── model.py           # BiLSTMClassifier + DistilBERTClassifier (nn.Module)
├── dl_utils.py        # Metrics, train loops, EarlyStopping, plots
├── baseline.py        # LR + TF-IDF pipeline (Model 1)
├── trainer.py         # Main entry point — runs all 3 models end-to-end
├── requirements.txt
│
├── data/
│   └── train.csv      # Jigsaw dataset (download from Kaggle, not committed)
│
├── saved_models/      # Created on first run
│   ├── baseline_lr.pkl
│   ├── vocab.pkl
│   ├── lstm_best.pth
│   ├── lstm_checkpoint.pth
│   ├── distilbert/
│   ├── bert_checkpoint.pth
│   ├── bert_thresholds.pkl
│   └── preprocessed_cache.pkl
│
└── results/           # Created on first run
    ├── bilstm_learning_curves.png
    ├── distilbert_learning_curves.png
    ├── model_comparison.png
    ├── per_label_f1_heatmap.png
    └── all_results.pkl
```

---

## 3. Dataset

**Source:** [Jigsaw Toxic Comment Classification Challenge](https://www.kaggle.com/datasets/julian3833/jigsaw-toxic-comment-classification-challenge) (Kaggle)

**Size:** 159,571 comments from Wikipedia talk pages (after deduplication)

**Labels:** 6 binary labels per comment

| Label | Positive samples | Prevalence | Imbalance ratio |
|-------|-----------------|------------|----------------|
| toxic | 15,294 | 9.58% | 9.4x |
| severe_toxic | 1,595 | 1.00% | 99.0x |
| obscene | 8,449 | 5.29% | 17.9x |
| threat | 478 | 0.30% | 332.8x |
| insult | 7,877 | 4.94% | 19.3x |
| identity_hate | 1,405 | 0.88% | 112.6x |

**Split:** 70% train / 15% val / 15% test → 111,699 / 23,935 / 23,937

**Key challenge:** `threat` appears in only 0.30% of samples — a 332:1 imbalance ratio. This is addressed through `BCEWithLogitsLoss` with per-class `pos_weight` and per-label threshold tuning.

---

## 4. Model Architecture

### Model 1 — Logistic Regression + TF-IDF (Baseline)

The "fast and interpretable" benchmark. TF-IDF with unigrams + bigrams (15K features), lbfgs solver, balanced class weights, 5-fold grid search over `C ∈ {0.01, 0.1, 1.0, 10.0}`.

### Model 2 — Bi-Directional LSTM

```
Embedding(vocab=20K, dim=128, padding_idx=0)
    ↓
BiLSTM(hidden=256, layers=2, dropout=0.4, bidirectional=True)
    ↓
Mean pooling across time steps
    ↓
Dropout(0.4)
    ↓
Linear(512 → 6)
```

Why BiLSTM over standard LSTM: reads each sentence left-to-right **and** right-to-left. The reverse pass allows "not bad" to be understood as positive — the negation context flows backward through the sequence. Mean pooling is used instead of the last hidden state because comments vary in length; averaging across all timesteps captures the full sentence rather than just its end.

### Model 3 — DistilBERT (Fine-tuned)

```
DistilBERT backbone (distilbert-base-uncased, 66M params)
    ↓
[CLS] token representation
    ↓
Linear(768 → 768) + ReLU
    ↓
Dropout(0.2)
    ↓
Linear(768 → 6)
```

Two-phase fine-tuning strategy:
- **Epoch 1:** backbone frozen — only the classification head is trained. This warms up the new head before touching BERT's pretrained weights.
- **Epochs 2+:** backbone unfrozen — full fine-tuning with AdamW + linear warmup (10% of steps).

---

## 5. Results

### Final Test Scores

| Model | Macro-F1 | Micro-F1 | ROC-AUC | Hamming↓ | Subset Acc |
|-------|----------|----------|---------|----------|------------|
| Logistic Regression | 0.5553 | 0.6669 | 0.9713 | 0.0308 | 0.8702 |
| BiLSTM | 0.6109 | 0.7057 | **0.9798** | 0.0265 | 0.8881 |
| DistilBERT | **0.6527** | **0.7617** | 0.9752 | **0.0189** | **0.9128** |

### Per-Label F1 — DistilBERT (Best Model)

| Label | F1 |
|-------|----|
| toxic | 0.800 |
| severe_toxic | 0.455 |
| obscene | 0.819 |
| threat | 0.533 |
| insult | 0.748 |
| identity_hate | 0.561 |

**Notable finding:** BiLSTM achieves a slightly higher ROC-AUC (0.9798) than DistilBERT (0.9752) despite lower Macro-F1. This means BiLSTM's probability rankings are marginally better, but DistilBERT's tuned thresholds produce better binary decisions. The two metrics measure different things — ROC-AUC measures discrimination, Macro-F1 measures the quality of the final binary prediction.

### Learning Curves

BiLSTM shows a widening train/val loss gap from epoch 8 onward — a classic overfitting signal. This was addressed with `dropout=0.4` and `weight_decay=1e-5`. DistilBERT's val F1 plateaus around epoch 6 while train loss continues falling — standard BERT fine-tuning behavior on subsampled data.

---

## 6. How to Run

### Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Download Jigsaw dataset from Kaggle and place at:
# ../data/train.csv
```

### Train all models

```bash
python trainer.py
```

The first run will:
1. Load and split the Jigsaw CSV
2. Preprocess all text (slow — ~20min for 159K comments via NLTK). **Cached automatically after first run.**
3. Train Model 1 (LR grid search — ~10min on CPU)
4. Train Model 2 (BiLSTM — ~30min on CPU)
5. Train Model 3 (DistilBERT — ~3hrs on CPU with 30K subsample)
6. Tune thresholds, evaluate, and save all plots to `results/`

### Resume after interruption

Both BiLSTM and DistilBERT checkpoint after every epoch. If training is interrupted, just re-run `python trainer.py` — it will automatically resume from the last saved epoch.

### Skip already-trained models

In `config.py`, set the stage flags:

```python
RUN_BASELINE = False  # skip — loads saved baseline_lr.pkl
RUN_LSTM     = False  # skip — loads saved lstm_best.pth
RUN_BERT     = True   # run this one
```

### GPU training

If you have a GPU, set `BERT_TRAIN_SUBSAMPLE = None` in `config.py` to use the full 111K training set. On a T4 GPU this reduces training to ~5 minutes per epoch.

---

## 7. Design Decisions

### Dual-mode preprocessing

Classical models (LR, LSTM) need heavy preprocessing — lemmatization, stopword removal, slang expansion — to reduce vocabulary noise. DistilBERT needs the opposite: minimal cleaning to preserve the grammatical structure its attention heads depend on. We implement `preprocess(text, mode='classical'|'bert')` to handle both from a single function.

**Critical detail:** negation words (`not`, `no`, `never`, etc.) are explicitly excluded from stopword removal. Without this, "not racist" becomes "racist" — a catastrophic reversal of meaning.

### Imbalanced loss function

`BCEWithLogitsLoss` with `pos_weight = neg_count / pos_count` per class. For `threat` (332:1 imbalance), this means a false negative costs 332× more than a false positive. Without this, the model learns to predict "no threat" for everything and still achieves 99.7% accuracy — which is completely useless.

### Per-label threshold tuning

After training, we sweep thresholds from 0.20 to 0.65 in 0.05 steps on the validation set and select the threshold that maximizes F1 per label independently. This is particularly important for rare classes where the model outputs low probabilities even for true positives.

| Label | Tuned threshold |
|-------|----------------|
| toxic | 0.60 |
| severe_toxic | 0.55 |
| obscene | 0.50 |
| threat | 0.25 |
| insult | 0.55 |
| identity_hate | 0.25 |

`threat` and `identity_hate` have low thresholds (0.25) because the model is systematically under-confident on these rare classes.

### Preprocessing cache

NLTK lemmatization on 159K comments takes ~20 minutes. The preprocessed arrays are saved to `saved_models/preprocessed_cache.pkl` on first run and loaded instantly on all subsequent runs.

---

## 8. Limitations & Future Work

**Current limitations:**

- DistilBERT was trained on a 30K subsample due to CPU constraints. With full 111K training data on GPU, Macro-F1 of 0.72–0.75 is achievable based on published benchmarks for this dataset.
- `threat` (332:1 imbalance) and `severe_toxic` (99:1) remain the hardest labels — even with pos_weight and threshold tuning, ~60 positive test samples is genuinely difficult.
- No data augmentation was used in this version. Back-translation or paraphrase-based augmentation for minority classes could meaningfully help `threat` and `identity_hate`.

**Highest-impact improvements:**

1. GPU training on full dataset — single biggest lever available
2. HateBERT or ToxicBERT — BERT variants pretrained on hate speech, better initialization than general DistilBERT
3. Ensemble — combine LR (high precision) + DistilBERT (high recall) through a meta-learner

---

*ITCS352 Deep Learning — Project, Semester 2/2025*