"""
dl_utils.py
-----------
Shared utilities for training and evaluation.

  Metrics:       macro_f1, per_label_f1, roc_auc
  Training:      train_one_epoch_lstm, train_one_epoch_bert
  Evaluation:    evaluate_lstm, evaluate_bert
  Tuning:        tune_thresholds
  Plotting:      plot_learning_curves, plot_per_label_f1
  Misc:          EarlyStopping, compute_pos_weight
"""

import os
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import (
    f1_score, roc_auc_score, classification_report,
    hamming_loss, accuracy_score,
)

from config import LABEL_COLS, NUM_CLASSES, BERT_MAX_POS_WEIGHT, THRESHOLD_RANGE


# ── Metrics ────────────────────────────────────────────────────────────────────

def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Macro-averaged F1 across all labels. Primary metric for imbalanced multi-label."""
    return f1_score(y_true, y_pred, average='macro', zero_division=0)


def per_label_f1(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """F1 score for each individual label."""
    scores = f1_score(y_true, y_pred, average=None, zero_division=0)
    return dict(zip(LABEL_COLS, scores))


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None = None,
    model_name: str = '',
) -> dict:
    """
    Compute the full suite of evaluation metrics.

    Args:
        y_true:     (N, num_classes) ground truth binary labels
        y_pred:     (N, num_classes) binary predictions (after threshold)
        y_prob:     (N, num_classes) raw probabilities (for ROC-AUC)
        model_name: string label for display

    Returns:
        Dictionary of metric name → value.
    """
    metrics = {
        'Model':        model_name,
        'Macro-F1':     f1_score(y_true, y_pred, average='macro',    zero_division=0),
        'Micro-F1':     f1_score(y_true, y_pred, average='micro',    zero_division=0),
        'Weighted-F1':  f1_score(y_true, y_pred, average='weighted', zero_division=0),
        'Hamming Loss': hamming_loss(y_true, y_pred),
        'Subset Acc':   accuracy_score(y_true, y_pred),    # exact match
    }

    if y_prob is not None:
        try:
            metrics['ROC-AUC'] = roc_auc_score(y_true, y_prob, average='macro')
        except ValueError:
            metrics['ROC-AUC'] = float('nan')  # fails if a class has no positives in y_true

    per_label = per_label_f1(y_true, y_pred)
    for label, score in per_label.items():
        metrics[f'F1_{label}'] = score

    return metrics


# ── Positive weight for imbalanced BCE loss ────────────────────────────────────

def compute_pos_weight(labels: np.ndarray, device: str) -> torch.Tensor:
    """
    Compute per-class positive weight for BCEWithLogitsLoss.

    pos_weight = #negative_samples / #positive_samples per class
    Capped at BERT_MAX_POS_WEIGHT to prevent extreme values destabilising training.

    Args:
        labels: (N, num_classes) training label array
        device: torch device string

    Returns:
        Tensor of shape (num_classes,)
    """
    pos  = labels.sum(axis=0)
    neg  = len(labels) - pos
    weights = np.where(pos > 0, neg / pos, 1.0)
    weights = np.clip(weights, 1.0, BERT_MAX_POS_WEIGHT)
    return torch.tensor(weights, dtype=torch.float32).to(device)


# ── Early Stopping ─────────────────────────────────────────────────────────────

class EarlyStopping:
    """Stop training when validation metric stops improving."""

    def __init__(self, patience: int = 3, mode: str = 'max', delta: float = 1e-4):
        """
        Args:
            patience: epochs to wait without improvement before stopping
            mode:     'max' (higher is better, e.g. F1) or 'min' (lower is better, e.g. loss)
            delta:    minimum change to count as improvement
        """
        self.patience = patience
        self.mode     = mode
        self.delta    = delta
        self.best     = None
        self.counter  = 0
        self.stop     = False

    def __call__(self, value: float) -> bool:
        if self.best is None:
            self.best = value
            return False

        improved = (value > self.best + self.delta) if self.mode == 'max' \
                   else (value < self.best - self.delta)

        if improved:
            self.best    = value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True

        return self.stop


# ── LSTM Training & Evaluation ─────────────────────────────────────────────────

def train_one_epoch_lstm(
    dataloader,
    model: nn.Module,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
) -> float:
    """Train BiLSTM for one epoch. Returns average loss."""
    model.train()
    total_loss = 0.0

    for X, y in tqdm(dataloader, desc='  Train', leave=False):
        X, y = X.to(device), y.to(device)

        optimizer.zero_grad()
        logits = model(X)
        loss   = loss_fn(logits, y)
        loss.backward()

        # Gradient clipping prevents exploding gradients (common in RNNs)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(dataloader)


def evaluate_lstm(
    dataloader,
    model: nn.Module,
    loss_fn: nn.Module,
    device: str,
    threshold: float = 0.5,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Evaluate BiLSTM on a DataLoader.

    Returns:
        avg_loss, y_true (N,C), y_pred binary (N,C), y_prob (N,C)
    """
    model.eval()
    total_loss = 0.0
    all_probs, all_true = [], []

    with torch.no_grad():
        for X, y in tqdm(dataloader, desc='  Eval', leave=False):
            X, y = X.to(device), y.to(device)
            logits = model(X)
            loss   = loss_fn(logits, y)
            total_loss += loss.item()

            probs = torch.sigmoid(logits)
            all_probs.append(probs.cpu().numpy())
            all_true.append(y.cpu().numpy())

    y_prob = np.vstack(all_probs)
    y_true = np.vstack(all_true)
    y_pred = (y_prob >= threshold).astype(int)

    return total_loss / len(dataloader), y_true, y_pred, y_prob


# ── DistilBERT Training & Evaluation ──────────────────────────────────────────

def train_one_epoch_bert(
    dataloader,
    model: nn.Module,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: str,
) -> float:
    """Train DistilBERT for one epoch. Returns average loss."""
    model.train()
    total_loss = 0.0

    for batch in tqdm(dataloader, desc='  Train', leave=False):
        input_ids      = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels         = batch['labels'].to(device)

        optimizer.zero_grad()
        logits = model(input_ids=input_ids, attention_mask=attention_mask)
        loss   = loss_fn(logits, labels)
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def evaluate_bert(
    dataloader,
    model: nn.Module,
    loss_fn: nn.Module,
    device: str,
    threshold: float = 0.5,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Evaluate DistilBERT on a DataLoader.

    Returns:
        avg_loss, y_true (N,C), y_pred binary (N,C), y_prob (N,C)
    """
    model.eval()
    total_loss = 0.0
    all_probs, all_true = [], []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc='  Eval', leave=False):
            input_ids      = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels         = batch['labels'].to(device)

            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            loss   = loss_fn(logits, labels)
            total_loss += loss.item()

            probs = torch.sigmoid(logits)
            all_probs.append(probs.cpu().numpy())
            all_true.append(labels.cpu().numpy())

    y_prob = np.vstack(all_probs)
    y_true = np.vstack(all_true)
    y_pred = (y_prob >= threshold).astype(int)

    return total_loss / len(dataloader), y_true, y_pred, y_prob


# ── Per-Label Threshold Tuning ─────────────────────────────────────────────────

def tune_thresholds(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> np.ndarray:
    """
    Find the optimal classification threshold for each label independently
    by maximising F1-score on the validation set.

    Why per-label thresholds:
        DistilBERT fine-tuned on imbalanced data is often under-confident on
        minority classes (e.g., outputs 0.35 for a genuine 'threat').
        Lowering the threshold for rare classes catches these cases.

    Args:
        y_true: (N, num_classes) validation ground truth
        y_prob: (N, num_classes) validation probabilities

    Returns:
        thresholds: (num_classes,) optimal threshold per label
    """
    start, stop, step = THRESHOLD_RANGE
    thresholds = np.full(y_true.shape[1], 0.5)

    for i in range(y_true.shape[1]):
        best_t  = 0.5
        best_f1 = 0.0
        for t in np.arange(start, stop, step):
            preds = (y_prob[:, i] >= t).astype(int)
            f1    = f1_score(y_true[:, i], preds, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_t  = t
        thresholds[i] = best_t
        print(f"  {LABEL_COLS[i]:<20} threshold={best_t:.2f}  val-F1={best_f1:.4f}")

    return thresholds


def apply_thresholds(y_prob: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """Apply per-label thresholds to probability matrix."""
    return (y_prob >= thresholds).astype(int)


# ── Plotting ───────────────────────────────────────────────────────────────────

def plot_learning_curves(
    train_losses: list[float],
    val_losses: list[float],
    val_f1s: list[float],
    model_name: str,
    save_dir: str,
) -> None:
    """
    Plot training vs validation loss AND validation F1 over epochs.
    The divergence point (val_loss rising while train_loss still falling)
    is where overfitting begins — annotated automatically.
    """
    os.makedirs(save_dir, exist_ok=True)
    epochs = range(1, len(train_losses) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'{model_name} — Learning Curves', fontsize=14)

    # Loss curves
    ax1.plot(epochs, train_losses, label='Train Loss', marker='o')
    ax1.plot(epochs, val_losses,   label='Val Loss',   marker='s')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss over Epochs')
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Annotate divergence point
    if len(val_losses) > 1:
        best_epoch = int(np.argmin(val_losses))
        ax1.axvline(x=best_epoch + 1, color='red', linestyle='--', alpha=0.7,
                    label=f'Best epoch ({best_epoch+1})')
        ax1.legend()

    # F1 curve
    ax2.plot(epochs, val_f1s, label='Val Macro-F1', marker='D', color='green')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Macro-F1')
    ax2.set_title('Validation Macro-F1 over Epochs')
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, f'{model_name.lower().replace(" ", "_")}_learning_curves.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Learning curves saved → {path}")


def plot_per_label_f1(results: list[dict], save_dir: str) -> None:
    """
    Heatmap-style bar chart showing per-label F1 for each model.
    Rows = models, columns = labels.
    """
    os.makedirs(save_dir, exist_ok=True)

    model_names = [r['Model'] for r in results]
    label_cols  = [c for c in LABEL_COLS]
    data = np.array([[r.get(f'F1_{l}', 0) for l in label_cols] for r in results])

    fig, ax = plt.subplots(figsize=(12, len(model_names) * 1.5 + 2))
    im = ax.imshow(data, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')

    ax.set_xticks(range(len(label_cols)))
    ax.set_yticks(range(len(model_names)))
    ax.set_xticklabels(label_cols, rotation=30, ha='right')
    ax.set_yticklabels(model_names)
    ax.set_title('Per-Label F1 Score by Model', fontsize=14)

    for i in range(len(model_names)):
        for j in range(len(label_cols)):
            ax.text(j, i, f'{data[i, j]:.2f}', ha='center', va='center',
                    fontsize=9, color='black')

    plt.colorbar(im, ax=ax, label='F1 Score')
    plt.tight_layout()
    path = os.path.join(save_dir, 'per_label_f1_heatmap.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Per-label F1 heatmap saved → {path}")


def plot_model_comparison(results: list[dict], save_dir: str) -> None:
    """Bar chart comparing all models on the key metrics."""
    os.makedirs(save_dir, exist_ok=True)

    metrics     = ['Macro-F1', 'Micro-F1', 'Weighted-F1', 'ROC-AUC', 'Hamming Loss', 'Subset Acc']
    model_names = [r['Model'] for r in results]
    colors      = ['#4C72B0', '#DD8452', '#55A868'][:len(results)]

    x = np.arange(len(metrics))
    width = 0.8 / len(results)

    fig, ax = plt.subplots(figsize=(14, 6))
    for i, (result, color) in enumerate(zip(results, colors)):
        vals = [result.get(m, 0) for m in metrics]
        bars = ax.bar(x + i * width, vals, width, label=result['Model'], color=color, alpha=0.85)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=8)

    ax.set_xticks(x + width * (len(results) - 1) / 2)
    ax.set_xticklabels(metrics, rotation=20, ha='right')
    ax.set_ylabel('Score')
    ax.set_title('Model Comparison — All Metrics', fontsize=14)
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, 'model_comparison.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Model comparison chart saved → {path}")
