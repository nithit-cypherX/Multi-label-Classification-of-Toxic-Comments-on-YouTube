"""
dataset.py
----------
Handles everything data-related:
  - load_jigsaw()          → loads + splits Jigsaw CSV into train/val/test DataFrames
  - ToxicDataset           → PyTorch Dataset for BiLSTM (integer-encoded tokens)
  - ToxicBertDataset       → PyTorch Dataset for DistilBERT (HuggingFace tokenizer)
  - build_vocab()          → builds vocabulary from training text
  - encode_texts()         → converts text → padded integer sequences
"""

import pickle
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from collections import Counter
from transformers import DistilBertTokenizer

from config import (
    LABEL_COLS, RANDOM_SEED, TRAIN_SIZE, VAL_SIZE,
    MAX_VOCAB_SIZE, MAX_SEQ_LEN, BERT_MODEL_NAME,
)


# ── Data Loading & Splitting ───────────────────────────────────────────────────

def load_jigsaw(data_path: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load Jigsaw CSV and split into train / val / test DataFrames.

    Expected CSV columns: id, comment_text, toxic, severe_toxic, obscene,
                          threat, insult, identity_hate

    Returns:
        df_train, df_val, df_test
    """
    df = pd.read_csv(data_path)

    # Basic sanity checks
    assert 'comment_text' in df.columns, "CSV must have a 'comment_text' column"
    assert all(c in df.columns for c in LABEL_COLS), f"CSV missing label columns: {LABEL_COLS}"

    # Drop duplicates and nulls
    df = df.drop_duplicates(subset='comment_text')
    df = df.dropna(subset=['comment_text'] + LABEL_COLS)
    df = df.reset_index(drop=True)

    print(f"Total samples after cleaning: {len(df):,}")
    _print_class_distribution(df)

    # Stratified-ish split: shuffle then slice
    np.random.seed(RANDOM_SEED)
    idx = np.random.permutation(len(df))
    n_train = int(len(df) * TRAIN_SIZE)
    n_val   = int(len(df) * VAL_SIZE)

    train_idx = idx[:n_train]
    val_idx   = idx[n_train:n_train + n_val]
    test_idx  = idx[n_train + n_val:]

    df_train = df.iloc[train_idx].reset_index(drop=True)
    df_val   = df.iloc[val_idx].reset_index(drop=True)
    df_test  = df.iloc[test_idx].reset_index(drop=True)

    print(f"\nSplit → Train: {len(df_train):,} | Val: {len(df_val):,} | Test: {len(df_test):,}")
    return df_train, df_val, df_test


def _print_class_distribution(df: pd.DataFrame) -> None:
    print("\nClass distribution:")
    print(f"{'Label':<20} {'Positive':>10} {'%':>8} {'Imbalance ratio':>18}")
    print("-" * 60)
    for col in LABEL_COLS:
        pos   = df[col].sum()
        total = len(df)
        neg   = total - pos
        ratio = neg / pos if pos > 0 else float('inf')
        print(f"{col:<20} {pos:>10,} {pos/total*100:>7.2f}% {ratio:>17.1f}x")


# ── Vocabulary ─────────────────────────────────────────────────────────────────

def build_vocab(texts: list[str], max_vocab: int = MAX_VOCAB_SIZE) -> dict[str, int]:
    """
    Build token → index vocabulary from training texts.

    Reserves index 0 for <PAD> and index 1 for <UNK>.

    Args:
        texts:      list of preprocessed strings (classical mode)
        max_vocab:  maximum vocabulary size

    Returns:
        vocab dict mapping token string → integer index
    """
    counter = Counter()
    for text in texts:
        counter.update(text.split())

    vocab = {'<PAD>': 0, '<UNK>': 1}
    for token, _ in counter.most_common(max_vocab - 2):
        vocab[token] = len(vocab)

    print(f"Vocabulary size: {len(vocab):,} tokens")
    return vocab


def encode_texts(
    texts: list[str],
    vocab: dict[str, int],
    max_len: int = MAX_SEQ_LEN,
) -> np.ndarray:
    """
    Convert preprocessed text strings to padded integer sequences.

    Args:
        texts:   list of preprocessed strings
        vocab:   token → index mapping (from build_vocab)
        max_len: sequence length (truncate / pad to this)

    Returns:
        numpy array of shape (N, max_len)
    """
    unk_idx = vocab['<UNK>']
    pad_idx = vocab['<PAD>']
    encoded = []

    for text in texts:
        tokens = text.split()[:max_len]
        ids = [vocab.get(t, unk_idx) for t in tokens]
        # Pad to max_len
        ids += [pad_idx] * (max_len - len(ids))
        encoded.append(ids)

    return np.array(encoded, dtype=np.int64)


def save_vocab(vocab: dict, path: str) -> None:
    with open(path, 'wb') as f:
        pickle.dump(vocab, f)


def load_vocab(path: str) -> dict:
    with open(path, 'rb') as f:
        return pickle.load(f)


# ── PyTorch Datasets ───────────────────────────────────────────────────────────

class ToxicDataset(Dataset):
    """
    PyTorch Dataset for BiLSTM.
    Accepts integer-encoded sequences (output of encode_texts).
    """

    def __init__(self, encoded: np.ndarray, labels: np.ndarray):
        """
        Args:
            encoded: (N, max_len) int64 array from encode_texts()
            labels:  (N, num_classes) float32 array of binary labels
        """
        self.X = torch.tensor(encoded, dtype=torch.long)
        self.y = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class ToxicBertDataset(Dataset):
    """
    PyTorch Dataset for DistilBERT.
    Tokenizes text on-the-fly using HuggingFace tokenizer.
    """

    def __init__(self, texts: list[str], labels: np.ndarray, max_len: int = MAX_SEQ_LEN):
        """
        Args:
            texts:   list of lightly-preprocessed strings (bert mode)
            labels:  (N, num_classes) float32 array of binary labels
            max_len: max token length for DistilBERT
        """
        self.texts  = texts
        self.labels = torch.tensor(labels, dtype=torch.float32)
        self.max_len = max_len
        self.tokenizer = DistilBertTokenizer.from_pretrained(BERT_MODEL_NAME)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )
        return {
            'input_ids':      encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels':         self.labels[idx],
        }
