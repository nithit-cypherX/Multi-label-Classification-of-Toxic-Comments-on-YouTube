"""
model.py
--------
Model architectures for the DL project.

  BiLSTMClassifier   → Model 2 (Deep Learning)
  DistilBERTClassifier → Model 3 (Transformer)
"""

import torch
import torch.nn as nn
from transformers import DistilBertModel

from config import (
    NUM_CLASSES, MAX_VOCAB_SIZE,
    LSTM_EMBED_DIM, LSTM_HIDDEN_DIM, LSTM_NUM_LAYERS, LSTM_DROPOUT,
    BERT_MODEL_NAME,
)


# ── Model 2: Bi-Directional LSTM ───────────────────────────────────────────────

class BiLSTMClassifier(nn.Module):
    """
    Bi-Directional LSTM for multi-label toxic comment classification.

    Architecture:
        Embedding → BiLSTM (stacked) → Dropout → Mean pool → Linear head

    Why BiLSTM:
        Reads each sentence left-to-right AND right-to-left.
        This lets it understand that "not bad" ≠ "bad"
        (the negation context flows back from the right pass).

    Why mean pooling over last hidden state:
        The last hidden state only sees the end of the sequence.
        Averaging across all timesteps captures the full sentence,
        which is better for long comments.
    """

    def __init__(
        self,
        vocab_size: int = MAX_VOCAB_SIZE,
        embed_dim: int = LSTM_EMBED_DIM,
        hidden_dim: int = LSTM_HIDDEN_DIM,
        num_layers: int = LSTM_NUM_LAYERS,
        dropout: float = LSTM_DROPOUT,
        num_classes: int = NUM_CLASSES,
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            padding_idx=0,   # <PAD> token index — zero-contribution to gradient
        )

        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,      # key: forward + backward pass
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.dropout = nn.Dropout(dropout)

        # hidden_dim * 2 because BiLSTM concatenates forward + backward hidden states
        self.classifier = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len) integer token indices

        Returns:
            logits: (batch, num_classes) — raw scores, apply sigmoid for probabilities
        """
        # (batch, seq_len) → (batch, seq_len, embed_dim)
        embedded = self.dropout(self.embedding(x))

        # lstm_out: (batch, seq_len, hidden_dim * 2)
        lstm_out, _ = self.lstm(embedded)

        # Mean pooling across time steps — captures full sentence context
        pooled = lstm_out.mean(dim=1)   # (batch, hidden_dim * 2)
        pooled = self.dropout(pooled)

        return self.classifier(pooled)  # (batch, num_classes)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ── Model 3: DistilBERT Classifier ────────────────────────────────────────────

class DistilBERTClassifier(nn.Module):
    """
    Fine-tuned DistilBERT for multi-label toxic comment classification.

    Architecture:
        DistilBERT backbone → [CLS] token → Dropout → Linear head

    Why DistilBERT:
        - 40% smaller than BERT, 60% faster, retains 97% of performance
        - Pre-trained on 6B+ tokens — already understands negation, sarcasm, context
        - Fine-tuning only needs a small dataset because the backbone is frozen-then-thawed

    Why [CLS] token for classification:
        DistilBERT is trained to encode sentence-level meaning in the [CLS]
        (class) token's representation. It's the standard pooling strategy.
    """

    def __init__(
        self,
        model_name: str = BERT_MODEL_NAME,
        num_classes: int = NUM_CLASSES,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.bert = DistilBertModel.from_pretrained(model_name)
        hidden_size = self.bert.config.hidden_size   # 768 for distilbert-base

        self.pre_classifier = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            input_ids:      (batch, seq_len) token indices from DistilBertTokenizer
            attention_mask: (batch, seq_len) 1 for real tokens, 0 for padding

        Returns:
            logits: (batch, num_classes) — raw scores, apply sigmoid for probabilities
        """
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        # [CLS] representation: first token of last hidden state
        cls_output = outputs.last_hidden_state[:, 0, :]   # (batch, 768)

        # Small pre-classifier helps adapt BERT's general representations
        # to our specific task before the final linear layer
        x = self.pre_classifier(cls_output)
        x = nn.ReLU()(x)
        x = self.dropout(x)

        return self.classifier(x)   # (batch, num_classes)

    def freeze_backbone(self) -> None:
        """Freeze BERT weights — only train the classifier head. Use for first epoch."""
        for param in self.bert.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self) -> None:
        """Unfreeze all weights for full fine-tuning."""
        for param in self.bert.parameters():
            param.requires_grad = True

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
