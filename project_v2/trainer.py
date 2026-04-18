"""
trainer.py
----------
Main entry point. Runs the full pipeline:

  1. Load & split Jigsaw dataset
  2. Preprocess (dual mode: classical + bert)
  3. Train Model 1: Logistic Regression + TF-IDF  (baseline)
  4. Train Model 2: BiLSTM                         (with resume support)
  5. Train Model 3: DistilBERT                     (with resume support)
  6. Tune per-label thresholds (DistilBERT)
  7. Final evaluation on held-out test set
  8. Save all results + plots

Resume behaviour:
  - Baseline:    skips entirely if saved_models/baseline_lr.pkl exists
  - Preprocessing: skips if saved_models/preprocessed_cache.pkl exists
  - BiLSTM:      resumes from saved_models/lstm_checkpoint.pth if it exists
  - DistilBERT:  resumes from saved_models/bert_checkpoint.pth if it exists

Run:
    python trainer.py
"""

import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from config import *
from preprocess import preprocess_batch
from dataset import (
    load_jigsaw, build_vocab, encode_texts,
    ToxicDataset, ToxicBertDataset,
    save_vocab, load_vocab,
)
from model import BiLSTMClassifier, DistilBERTClassifier
from baseline import train_baseline, evaluate_baseline, load_baseline
from dl_utils import (
    train_one_epoch_lstm, evaluate_lstm,
    train_one_epoch_bert, evaluate_bert,
    compute_pos_weight, EarlyStopping,
    tune_thresholds, apply_thresholds,
    compute_all_metrics,
    plot_learning_curves, plot_per_label_f1, plot_model_comparison,
    macro_f1,
)

for d in [SAVE_DIR, RESULTS_DIR, LOG_DIR]:
    os.makedirs(d, exist_ok=True)

print(f"Device: {DEVICE}")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Load & split data
# ══════════════════════════════════════════════════════════════════════════════

df_train, df_val, df_test = load_jigsaw(DATA_PATH)

y_train = df_train[LABEL_COLS].values.astype(np.float32)
y_val   = df_val[LABEL_COLS].values.astype(np.float32)
y_test  = df_test[LABEL_COLS].values.astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — Preprocess (cached so re-runs skip NLTK entirely)
# ══════════════════════════════════════════════════════════════════════════════

_CACHE = f'{SAVE_DIR}/preprocessed_cache.pkl'

if os.path.exists(_CACHE):
    print("\nLoading preprocessed text from cache...")
    with open(_CACHE, 'rb') as f:
        cache = pickle.load(f)
    X_train_cls  = cache['X_train_cls']
    X_val_cls    = cache['X_val_cls']
    X_test_cls   = cache['X_test_cls']
    X_train_bert = cache['X_train_bert']
    X_val_bert   = cache['X_val_bert']
    X_test_bert  = cache['X_test_bert']
else:
    print("\nPreprocessing — classical mode (for LR + LSTM)...")
    X_train_cls = preprocess_batch(df_train['comment_text'].tolist(), mode='classical')
    X_val_cls   = preprocess_batch(df_val['comment_text'].tolist(),   mode='classical')
    X_test_cls  = preprocess_batch(df_test['comment_text'].tolist(),  mode='classical')

    print("Preprocessing — bert mode (for DistilBERT)...")
    X_train_bert = preprocess_batch(df_train['comment_text'].tolist(), mode='bert')
    X_val_bert   = preprocess_batch(df_val['comment_text'].tolist(),   mode='bert')
    X_test_bert  = preprocess_batch(df_test['comment_text'].tolist(),  mode='bert')

    with open(_CACHE, 'wb') as f:
        pickle.dump({
            'X_train_cls': X_train_cls, 'X_val_cls': X_val_cls, 'X_test_cls': X_test_cls,
            'X_train_bert': X_train_bert, 'X_val_bert': X_val_bert, 'X_test_bert': X_test_bert,
        }, f)
    print(f"Preprocessed text cached → {_CACHE}")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — Model 1: Logistic Regression (skip if already done)
# ══════════════════════════════════════════════════════════════════════════════

if not RUN_BASELINE and os.path.exists(BASELINE_PATH):
    print(f"\nSkipping baseline (RUN_BASELINE=False) — loading from {BASELINE_PATH}")
    lr_model, tfidf_vec = load_baseline()
    X_val_vec   = tfidf_vec.transform(X_val_cls)
    y_pred_lr   = lr_model.predict(X_val_vec)
    y_prob_lr   = lr_model.predict_proba(X_val_vec)
    lr_val_metrics = compute_all_metrics(y_val, y_pred_lr, y_prob_lr, model_name='Logistic Regression')
elif os.path.exists(BASELINE_PATH):
    print(f"\nBaseline already trained — loading from {BASELINE_PATH}")
    lr_model, tfidf_vec = load_baseline()
    X_val_vec   = tfidf_vec.transform(X_val_cls)
    y_pred_lr   = lr_model.predict(X_val_vec)
    y_prob_lr   = lr_model.predict_proba(X_val_vec)
    lr_val_metrics = compute_all_metrics(y_val, y_pred_lr, y_prob_lr, model_name='Logistic Regression')
else:
    lr_model, tfidf_vec, lr_val_metrics = train_baseline(
        X_train_cls, y_train, X_val_cls, y_val
    )


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — Model 2: BiLSTM (skip if RUN_LSTM=False)
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("MODEL 2: Bi-Directional LSTM")
print("="*60)

_VOCAB_PATH = f'{SAVE_DIR}/vocab.pkl'
if os.path.exists(_VOCAB_PATH):
    vocab = load_vocab(_VOCAB_PATH)
    print(f"Vocabulary loaded ({len(vocab):,} tokens)")
else:
    vocab = build_vocab(X_train_cls, max_vocab=MAX_VOCAB_SIZE)
    save_vocab(vocab, _VOCAB_PATH)

if not RUN_LSTM:
    print("Skipping LSTM training (RUN_LSTM=False) — loading best weights...")
    lstm_model = BiLSTMClassifier(vocab_size=len(vocab)).to(DEVICE)
    pos_weight   = compute_pos_weight(y_train, DEVICE)
    lstm_loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    lstm_model.load_state_dict(torch.load(LSTM_PATH, map_location=DEVICE))

    X_test_enc   = encode_texts(X_test_cls, vocab, MAX_SEQ_LEN)
    lstm_test_dl = DataLoader(ToxicDataset(X_test_enc, y_test), batch_size=LSTM_BATCH_SIZE)
    _, y_test_true, y_test_pred_lstm, y_test_prob_lstm = evaluate_lstm(
        lstm_test_dl, lstm_model, lstm_loss_fn, DEVICE
    )
    lstm_test_metrics = compute_all_metrics(
        y_test_true, y_test_pred_lstm, y_test_prob_lstm, model_name='BiLSTM'
    )
    print(f"BiLSTM Test Macro-F1: {lstm_test_metrics['Macro-F1']:.4f}")

else:
    X_train_enc = encode_texts(X_train_cls, vocab, MAX_SEQ_LEN)
    X_val_enc   = encode_texts(X_val_cls,   vocab, MAX_SEQ_LEN)
    X_test_enc  = encode_texts(X_test_cls,  vocab, MAX_SEQ_LEN)

    lstm_train_dl = DataLoader(ToxicDataset(X_train_enc, y_train), batch_size=LSTM_BATCH_SIZE, shuffle=True)
    lstm_val_dl   = DataLoader(ToxicDataset(X_val_enc,   y_val),   batch_size=LSTM_BATCH_SIZE)
    lstm_test_dl  = DataLoader(ToxicDataset(X_test_enc,  y_test),  batch_size=LSTM_BATCH_SIZE)

    lstm_model = BiLSTMClassifier(vocab_size=len(vocab)).to(DEVICE)
    print(f"BiLSTM parameters: {lstm_model.count_parameters():,}")

    pos_weight   = compute_pos_weight(y_train, DEVICE)
    lstm_loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    lstm_optim   = optim.Adam(lstm_model.parameters(), lr=LSTM_LR, weight_decay=LSTM_WEIGHT_DECAY)

    lstm_start_epoch  = 0
    best_lstm_f1      = 0.0
    lstm_train_losses = []
    lstm_val_losses   = []
    lstm_val_f1s      = []

    if os.path.exists(LSTM_CKPT_PATH):
        print(f"\nResuming BiLSTM from checkpoint: {LSTM_CKPT_PATH}")
        ckpt = torch.load(LSTM_CKPT_PATH, map_location=DEVICE)
        lstm_model.load_state_dict(ckpt['model_state'])
        lstm_optim.load_state_dict(ckpt['optimizer_state'])
        lstm_start_epoch  = ckpt['epoch'] + 1
        best_lstm_f1      = ckpt['best_f1']
        lstm_train_losses = ckpt['train_losses']
        lstm_val_losses   = ckpt['val_losses']
        lstm_val_f1s      = ckpt['val_f1s']
        print(f"Resuming from epoch {lstm_start_epoch} (best so far: {best_lstm_f1:.4f})")
    else:
        print(f"\nTraining from scratch — up to {LSTM_EPOCHS} epochs (patience={LSTM_PATIENCE})...")

    lstm_early_stop = EarlyStopping(patience=LSTM_PATIENCE, mode='max')
    for f1 in lstm_val_f1s:
        lstm_early_stop(f1)
    lstm_early_stop.stop = False

    for epoch in range(lstm_start_epoch, LSTM_EPOCHS):
        train_loss = train_one_epoch_lstm(lstm_train_dl, lstm_model, lstm_loss_fn, lstm_optim, DEVICE)
        val_loss, y_val_true, y_val_pred, y_val_prob = evaluate_lstm(lstm_val_dl, lstm_model, lstm_loss_fn, DEVICE)

        val_f1 = macro_f1(y_val_true, y_val_pred)
        lstm_train_losses.append(train_loss)
        lstm_val_losses.append(val_loss)
        lstm_val_f1s.append(val_f1)

        print(f"Epoch {epoch+1:02d}/{LSTM_EPOCHS}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  val_macro_f1={val_f1:.4f}")

        torch.save({
            'epoch': epoch, 'model_state': lstm_model.state_dict(),
            'optimizer_state': lstm_optim.state_dict(), 'best_f1': best_lstm_f1,
            'train_losses': lstm_train_losses, 'val_losses': lstm_val_losses, 'val_f1s': lstm_val_f1s,
        }, LSTM_CKPT_PATH)

        if val_f1 > best_lstm_f1:
            best_lstm_f1 = val_f1
            torch.save(lstm_model.state_dict(), LSTM_PATH)
            print(f"  ✓ Best model saved (Macro-F1={best_lstm_f1:.4f})")

        if lstm_early_stop(val_f1):
            print(f"  Early stopping triggered at epoch {epoch+1}")
            break

    plot_learning_curves(lstm_train_losses, lstm_val_losses, lstm_val_f1s, 'BiLSTM', RESULTS_DIR)

    lstm_model.load_state_dict(torch.load(LSTM_PATH, map_location=DEVICE))
    _, y_test_true, y_test_pred_lstm, y_test_prob_lstm = evaluate_lstm(
        lstm_test_dl, lstm_model, lstm_loss_fn, DEVICE
    )
    lstm_test_metrics = compute_all_metrics(
        y_test_true, y_test_pred_lstm, y_test_prob_lstm, model_name='BiLSTM'
    )
    print(f"\nBiLSTM Test Macro-F1: {lstm_test_metrics['Macro-F1']:.4f}")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — Model 3: DistilBERT (resume from checkpoint if available)
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("MODEL 3: DistilBERT")
print("="*60)

bert_train_ds = ToxicBertDataset(X_train_bert, y_train)
bert_val_ds   = ToxicBertDataset(X_val_bert,   y_val)
bert_test_ds  = ToxicBertDataset(X_test_bert,  y_test)

# Subsample training set for CPU — full 111K takes ~8hrs/epoch on CPU
# BERT_TRAIN_SUBSAMPLE = 20000 → ~35min/epoch, still strong results
if BERT_TRAIN_SUBSAMPLE and BERT_TRAIN_SUBSAMPLE < len(bert_train_ds):
    np.random.seed(RANDOM_SEED)
    sub_idx = np.random.choice(len(bert_train_ds), BERT_TRAIN_SUBSAMPLE, replace=False)
    bert_train_ds = torch.utils.data.Subset(bert_train_ds, sub_idx)
    # Recompute pos_weight on the subsample labels
    sub_labels = y_train[sub_idx]
    pos_weight  = compute_pos_weight(sub_labels, DEVICE)
    bert_loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    print(f"BERT training on {BERT_TRAIN_SUBSAMPLE:,} samples (subsampled for CPU)")

bert_train_dl = DataLoader(bert_train_ds, batch_size=BERT_BATCH_SIZE, shuffle=True)
bert_val_dl   = DataLoader(bert_val_ds,   batch_size=BERT_BATCH_SIZE)
bert_test_dl  = DataLoader(bert_test_ds,  batch_size=BERT_BATCH_SIZE)

bert_model = DistilBERTClassifier().to(DEVICE)
print(f"DistilBERT total parameters: {bert_model.count_parameters():,}")

bert_loss_fn   = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
bert_optim     = optim.AdamW(bert_model.parameters(), lr=BERT_LR, weight_decay=BERT_WEIGHT_DECAY)
total_steps    = len(bert_train_dl) * BERT_EPOCHS
warmup_steps   = int(0.1 * total_steps)
bert_scheduler = get_linear_schedule_with_warmup(bert_optim, warmup_steps, total_steps)

# ── Resume from checkpoint if exists ──────────────────────────────────────
bert_start_epoch  = 0
best_bert_f1      = 0.0
bert_train_losses = []
bert_val_losses   = []
bert_val_f1s      = []

if os.path.exists(BERT_CKPT_PATH):
    print(f"\nResuming DistilBERT from checkpoint: {BERT_CKPT_PATH}")
    ckpt = torch.load(BERT_CKPT_PATH, map_location=DEVICE)
    bert_model.load_state_dict(ckpt['model_state'])
    bert_optim.load_state_dict(ckpt['optimizer_state'])
    bert_scheduler.load_state_dict(ckpt['scheduler_state'])
    bert_start_epoch  = ckpt['epoch'] + 1
    best_bert_f1      = ckpt['best_f1']
    bert_train_losses = ckpt['train_losses']
    bert_val_losses   = ckpt['val_losses']
    bert_val_f1s      = ckpt['val_f1s']
    print(f"Resuming from epoch {bert_start_epoch} (best so far: {best_bert_f1:.4f})")
else:
    print(f"\nTraining for up to {BERT_EPOCHS} epochs (patience={BERT_PATIENCE})...")
    print("Epoch 1: backbone frozen. Epoch 2+: full fine-tuning.")

bert_early_stop = EarlyStopping(patience=BERT_PATIENCE, mode='max')
for f1 in bert_val_f1s:
    bert_early_stop(f1)
bert_early_stop.stop = False

for epoch in range(bert_start_epoch, BERT_EPOCHS):
    if epoch == 0:
        bert_model.freeze_backbone()
        print("\n  [Epoch 1] Backbone frozen — training head only")
    elif epoch == 1:
        bert_model.unfreeze_backbone()
        print("\n  [Epoch 2+] Backbone unfrozen — full fine-tuning")

    train_loss = train_one_epoch_bert(
        bert_train_dl, bert_model, bert_loss_fn, bert_optim, bert_scheduler, DEVICE
    )
    val_loss, y_val_true, y_val_pred, y_val_prob = evaluate_bert(
        bert_val_dl, bert_model, bert_loss_fn, DEVICE
    )

    val_f1 = macro_f1(y_val_true, y_val_pred)
    bert_train_losses.append(train_loss)
    bert_val_losses.append(val_loss)
    bert_val_f1s.append(val_f1)

    print(f"Epoch {epoch+1:02d}/{BERT_EPOCHS}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  val_macro_f1={val_f1:.4f}")

    # Save full checkpoint every epoch
    torch.save({
        'epoch':           epoch,
        'model_state':     bert_model.state_dict(),
        'optimizer_state': bert_optim.state_dict(),
        'scheduler_state': bert_scheduler.state_dict(),
        'best_f1':         best_bert_f1,
        'train_losses':    bert_train_losses,
        'val_losses':      bert_val_losses,
        'val_f1s':         bert_val_f1s,
    }, BERT_CKPT_PATH)

    if val_f1 > best_bert_f1:
        best_bert_f1 = val_f1
        bert_model.bert.save_pretrained(BERT_PATH)
        torch.save({
            'pre_classifier': bert_model.pre_classifier.state_dict(),
            'classifier':     bert_model.classifier.state_dict(),
        }, f'{BERT_PATH}/head_weights.pth')
        print(f"  ✓ Best model saved (Macro-F1={best_bert_f1:.4f})")

    if bert_early_stop(val_f1):
        print(f"  Early stopping triggered at epoch {epoch+1}")
        break

plot_learning_curves(bert_train_losses, bert_val_losses, bert_val_f1s, 'DistilBERT', RESULTS_DIR)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 6 — Threshold tuning
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("THRESHOLD TUNING (DistilBERT)")
print("="*60)

_, y_val_true, _, y_val_prob = evaluate_bert(bert_val_dl, bert_model, bert_loss_fn, DEVICE)
best_thresholds = tune_thresholds(y_val_true, y_val_prob)

with open(THRESHOLDS_PATH, 'wb') as f:
    pickle.dump(best_thresholds, f)
print(f"\nThresholds saved → {THRESHOLDS_PATH}")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 7 — Final test evaluation
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("FINAL TEST EVALUATION")
print("="*60)

lr_test_metrics = evaluate_baseline(lr_model, tfidf_vec, X_test_cls, y_test)

_, y_test_true, _, y_test_prob_bert = evaluate_bert(bert_test_dl, bert_model, bert_loss_fn, DEVICE)
y_test_pred_bert  = apply_thresholds(y_test_prob_bert, best_thresholds)
bert_test_metrics = compute_all_metrics(
    y_test_true, y_test_pred_bert, y_test_prob_bert, model_name='DistilBERT'
)

all_results = [lr_test_metrics, lstm_test_metrics, bert_test_metrics]


# ══════════════════════════════════════════════════════════════════════════════
# STEP 8 — Save + plot
# ══════════════════════════════════════════════════════════════════════════════

with open(RESULTS_PATH, 'wb') as f:
    pickle.dump(all_results, f)

plot_per_label_f1(all_results, RESULTS_DIR)
plot_model_comparison(all_results, RESULTS_DIR)

print("\n" + "="*60)
print("FINAL RESULTS SUMMARY")
print("="*60)
header = f"{'Model':<25} {'Macro-F1':>10} {'Micro-F1':>10} {'ROC-AUC':>10} {'Hamming↓':>10}"
print(header)
print("-" * len(header))
for r in all_results:
    print(f"{r['Model']:<25} {r['Macro-F1']:>10.4f} {r['Micro-F1']:>10.4f} "
          f"{r.get('ROC-AUC', float('nan')):>10.4f} {r['Hamming Loss']:>10.4f}")

print("\nTraining complete.")