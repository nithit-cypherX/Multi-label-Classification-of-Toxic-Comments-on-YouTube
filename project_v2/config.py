"""
config.py
---------
Single source of truth for all hyperparameters, paths, and constants.
Change values here — nowhere else.
"""

import torch

# ── Labels ────────────────────────────────────────────────────────────────────
LABEL_COLS = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
NUM_CLASSES = len(LABEL_COLS)

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_PATH       = './data/train.csv'       # Jigsaw train.csv from Kaggle
SAVE_DIR        = './saved_models'
RESULTS_DIR     = './results'
LOG_DIR         = './runs'

BASELINE_PATH   = f'{SAVE_DIR}/baseline_lr.pkl'
LSTM_PATH       = f'{SAVE_DIR}/lstm_best.pth'
BERT_PATH       = f'{SAVE_DIR}/distilbert'
THRESHOLDS_PATH = f'{SAVE_DIR}/bert_thresholds.pkl'
RESULTS_PATH    = f'{RESULTS_DIR}/all_results.pkl'

# ── Data split ─────────────────────────────────────────────────────────────────
TRAIN_SIZE = 0.70
VAL_SIZE   = 0.15
TEST_SIZE  = 0.15
RANDOM_SEED = 42

# ── Preprocessing ──────────────────────────────────────────────────────────────
MAX_VOCAB_SIZE  = 20000   # vocabulary size for LSTM embedding
MAX_SEQ_LEN     = 128     # max token length (LSTM padding & BERT truncation)

# ── Baseline (Logistic Regression + TF-IDF) ────────────────────────────────────
TFIDF_MAX_FEATURES = 15000
TFIDF_NGRAM_RANGE  = (1, 2)
LR_C_VALUES        = [0.01, 0.1, 1.0, 10.0]
LR_CV_FOLDS        = 5
LR_MAX_ITER        = 1000

# ── BiLSTM ─────────────────────────────────────────────────────────────────────
LSTM_EMBED_DIM    = 128
LSTM_HIDDEN_DIM   = 256
LSTM_NUM_LAYERS   = 2
LSTM_DROPOUT      = 0.4       # was 0.3 (underfit) → 0.5 (overfit) → 0.4 (balanced)
LSTM_WEIGHT_DECAY = 1e-5      # light L2, don't over-regularize
LSTM_BATCH_SIZE   = 64
LSTM_EPOCHS       = 15
LSTM_LR           = 1e-3
LSTM_PATIENCE     = 3

# Checkpoint paths (for resume support)
LSTM_CKPT_PATH   = f'{SAVE_DIR}/lstm_checkpoint.pth'
BERT_CKPT_PATH   = f'{SAVE_DIR}/bert_checkpoint.pth'

# ── DistilBERT ─────────────────────────────────────────────────────────────────
BERT_MODEL_NAME     = 'distilbert-base-uncased'
BERT_BATCH_SIZE     = 32
BERT_EPOCHS         = 10      # was 5 — model was still improving at epoch 5, never early-stopped
BERT_LR             = 2e-5    # slightly lower — reduce train/val loss gap (was 3e-5)
BERT_WEIGHT_DECAY   = 0.05    # stronger L2 (was 0.01) — train_loss 0.063 vs val 0.225 is a big gap
BERT_PATIENCE       = 3       # was 2 — give more room before stopping
BERT_MAX_POS_WEIGHT = 10.0
BERT_TRAIN_SUBSAMPLE = 30000  # was 20K — slightly more data, ~20min/epoch on CPU

# Threshold sweep range for per-label tuning
THRESHOLD_RANGE  = (0.20, 0.65, 0.05)  # (start, stop, step)

# ── Training stage flags ───────────────────────────────────────────────────────
# Set to False to skip a stage entirely on next run.
# Baseline auto-skips if saved file exists regardless of this flag.
RUN_BASELINE = False   # already done
RUN_LSTM     = False   # already done (Test Macro-F1: 0.61)
RUN_BERT     = True    # still in progress

# ── Device ─────────────────────────────────────────────────────────────────────
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'