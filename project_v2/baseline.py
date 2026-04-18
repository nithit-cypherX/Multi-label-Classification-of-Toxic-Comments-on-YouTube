"""
baseline.py
-----------
Model 1: Logistic Regression + TF-IDF (classical ML baseline).

This is the "dumb but fast" benchmark. It converts text to TF-IDF vectors
and trains one binary LR classifier per label (OneVsRest strategy).

Why it's the baseline:
  - No GPU needed, trains in seconds
  - Interpretable: you can inspect which words drive each decision
  - Any DL model that can't beat this isn't learning anything useful

Why LR beats Naive Bayes here:
  - NB independence assumption is fundamentally wrong for toxic text
    ("not racist" and "racist" look identical to NB)
  - LR with balanced class weights handles imbalance better via loss scaling
  - LR supports bigrams natively through TF-IDF, capturing "hate speech"
    as a two-word feature rather than two independent words
"""

import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from config import (
    LABEL_COLS, TFIDF_MAX_FEATURES, TFIDF_NGRAM_RANGE,
    LR_C_VALUES, LR_CV_FOLDS, LR_MAX_ITER,
    BASELINE_PATH,
)
from dl_utils import compute_all_metrics


def train_baseline(
    X_train: list[str],
    y_train: np.ndarray,
    X_val: list[str],
    y_val: np.ndarray,
) -> tuple[OneVsRestClassifier, TfidfVectorizer, dict]:
    """
    Train the TF-IDF + Logistic Regression baseline.

    Grid-searches over C and penalty on the original (non-augmented) training set.
    Final model is trained on the full training set.

    Args:
        X_train:  preprocessed training texts (classical mode)
        y_train:  (N, num_classes) training labels
        X_val:    preprocessed validation texts
        y_val:    (N, num_classes) validation labels

    Returns:
        model:      fitted OneVsRestClassifier
        vectorizer: fitted TfidfVectorizer
        metrics:    evaluation dict from compute_all_metrics()
    """
    print("\n" + "="*60)
    print("MODEL 1: Logistic Regression + TF-IDF (Baseline)")
    print("="*60)

    # ── Vectorise ──────────────────────────────────────────────────────────────
    print("\nFitting TF-IDF vectorizer...")
    vectorizer = TfidfVectorizer(
        max_features=TFIDF_MAX_FEATURES,
        ngram_range=TFIDF_NGRAM_RANGE,   # unigrams + bigrams
        min_df=2,                          # ignore tokens in <2 docs (typos)
        max_df=0.95,                       # ignore tokens in >95% of docs (too common)
        sublinear_tf=True,                 # log scaling: tf → 1 + log(tf)
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_val_vec   = vectorizer.transform(X_val)

    print(f"TF-IDF matrix: {X_train_vec.shape[0]:,} samples × {X_train_vec.shape[1]:,} features")

    # ── Grid Search ────────────────────────────────────────────────────────────
    # Solver: lbfgs is the best choice here.
    #   - SAGA + L1 on 111K × 15K features hits convergence problems on CPU
    #   - lbfgs with L2 converges reliably, runs in minutes not hours
    #   - At 111K samples L1 vs L2 sparsity makes <1% difference in Macro-F1
    print(f"\nRunning {LR_CV_FOLDS}-fold grid search over C={LR_C_VALUES}...")

    base_lr = LogisticRegression(
        solver='lbfgs',
        class_weight='balanced',
        max_iter=LR_MAX_ITER,
        random_state=42,
    )

    clf = OneVsRestClassifier(base_lr)

    param_grid = {'estimator__C': LR_C_VALUES}

    grid_search = GridSearchCV(
        clf,
        param_grid,
        cv=LR_CV_FOLDS,
        scoring='f1_macro',
        n_jobs=-1,
        verbose=1,
    )
    grid_search.fit(X_train_vec, y_train)

    best_params = grid_search.best_params_
    best_cv_f1  = grid_search.best_score_
    print(f"\nBest params:  {best_params}")
    print(f"Best CV Macro-F1: {best_cv_f1:.4f}")

    # ── Retrain on full training set with best C ───────────────────────────────
    print("\nRetraining on full training set with best params...")
    best_lr = LogisticRegression(
        solver='lbfgs',
        class_weight='balanced',
        max_iter=LR_MAX_ITER,
        random_state=42,
        C=best_params['estimator__C'],
    )
    model = OneVsRestClassifier(best_lr)
    model.fit(X_train_vec, y_train)

    # ── Evaluate on validation set ─────────────────────────────────────────────
    y_pred = model.predict(X_val_vec)
    y_prob = model.predict_proba(X_val_vec)

    metrics = compute_all_metrics(y_val, y_pred, y_prob, model_name='Logistic Regression')
    _print_metrics(metrics)

    # ── Save ───────────────────────────────────────────────────────────────────
    save_baseline(model, vectorizer)

    return model, vectorizer, metrics


def evaluate_baseline(
    model: OneVsRestClassifier,
    vectorizer: TfidfVectorizer,
    X_test: list[str],
    y_test: np.ndarray,
) -> dict:
    """Final evaluation of the baseline on the held-out test set."""
    print("\nEvaluating baseline on test set...")
    X_test_vec = vectorizer.transform(X_test)
    y_pred = model.predict(X_test_vec)
    y_prob = model.predict_proba(X_test_vec)
    metrics = compute_all_metrics(y_test, y_pred, y_prob, model_name='Logistic Regression')
    _print_metrics(metrics)
    return metrics


def save_baseline(model: OneVsRestClassifier, vectorizer: TfidfVectorizer) -> None:
    import os
    os.makedirs(BASELINE_PATH.rsplit('/', 1)[0], exist_ok=True)
    with open(BASELINE_PATH, 'wb') as f:
        pickle.dump({'model': model, 'vectorizer': vectorizer}, f)
    print(f"Baseline saved → {BASELINE_PATH}")


def load_baseline(path: str = BASELINE_PATH) -> tuple[OneVsRestClassifier, TfidfVectorizer]:
    with open(path, 'rb') as f:
        obj = pickle.load(f)
    return obj['model'], obj['vectorizer']


def _print_metrics(metrics: dict) -> None:
    print(f"\n  Macro-F1:     {metrics['Macro-F1']:.4f}")
    print(f"  Micro-F1:     {metrics['Micro-F1']:.4f}")
    print(f"  Hamming Loss: {metrics['Hamming Loss']:.4f}")
    print(f"  Subset Acc:   {metrics['Subset Acc']:.4f}")
    if 'ROC-AUC' in metrics:
        print(f"  ROC-AUC:      {metrics['ROC-AUC']:.4f}")
    print("\n  Per-label F1:")
    for label in LABEL_COLS:
        print(f"    {label:<20} {metrics.get(f'F1_{label}', 0):.4f}")