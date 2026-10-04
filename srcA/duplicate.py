from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


@dataclass(frozen=True)
class DuplicateMetrics:
    """Evaluation metrics for near-duplicate detection."""
    threshold: float
    precision: float
    recall: float
    f1: float
    accuracy: float


def cosine_similarity_pairs(embeddings: np.ndarray, index_a: np.ndarray, index_b: np.ndarray) -> np.ndarray:
    """Compute cosine similarity for corresponding embedding pairs."""
    left = embeddings[index_a]
    right = embeddings[index_b]
    return np.sum(left * right, axis=1)


def predict_duplicates(similarities: np.ndarray, threshold: float) -> np.ndarray:
    """Convert pairwise similarities into duplicate predictions."""
    return similarities >= threshold


def compute_duplicate_metrics(y_true: np.ndarray, y_pred: np.ndarray, threshold: float) -> DuplicateMetrics:
    """Compute standard binary classification metrics."""
    return DuplicateMetrics(
        threshold=threshold,
        precision=precision_score(y_true, y_pred, zero_division=0),
        recall=recall_score(y_true, y_pred, zero_division=0),
        f1=f1_score(y_true, y_pred, zero_division=0),
        accuracy=accuracy_score(y_true, y_pred),
    )


def evaluate_threshold(similarities: np.ndarray, y_true: np.ndarray, threshold: float) -> DuplicateMetrics:
    """Evaluate one similarity threshold."""
    predictions = predict_duplicates(similarities, threshold)
    return compute_duplicate_metrics(y_true, predictions, threshold)


def search_best_threshold(similarities: np.ndarray, y_true: np.ndarray, thresholds: np.ndarray | None = None) -> DuplicateMetrics:
    """Search for the threshold with the best F1 score on development data."""
    if thresholds is None:
        thresholds = np.linspace(0.50, 0.99, 100)

    best_metrics = None

    for threshold in thresholds:
        metrics = evaluate_threshold(similarities, y_true, float(threshold))

        if best_metrics is None or metrics.f1 > best_metrics.f1:
            best_metrics = metrics

    return best_metrics


def build_pair_similarity_table(pairs: pd.DataFrame, creatives: pd.DataFrame, embeddings: np.ndarray) -> pd.DataFrame:
    """Attach fused cosine similarity to a pair dataframe."""
    creative_to_index = {creative_id: index for index, creative_id in enumerate(creatives["creative_id"])}

    index_a = pairs["creative_id_a"].map(creative_to_index)
    index_b = pairs["creative_id_b"].map(creative_to_index)

    if index_a.isna().any():
        raise ValueError("Some creative_id_a values are missing from creatives.parquet.")

    if index_b.isna().any():
        raise ValueError("Some creative_id_b values are missing from creatives.parquet.")

    similarities = cosine_similarity_pairs(
        embeddings,
        index_a.to_numpy(dtype=int),
        index_b.to_numpy(dtype=int),
    )

    result = pairs.copy()
    result["similarity"] = similarities

    return result


def evaluate_duplicate_pairs(pairs: pd.DataFrame, creatives: pd.DataFrame, embeddings: np.ndarray, threshold: float) -> DuplicateMetrics:
    """Evaluate a frozen duplicate threshold on labeled pairs."""
    scored_pairs = build_pair_similarity_table(pairs, creatives, embeddings)
    y_true = scored_pairs["is_duplicate"].astype(bool).to_numpy()
    y_pred = predict_duplicates(scored_pairs["similarity"].to_numpy(), threshold)
    return compute_duplicate_metrics(y_true, y_pred, threshold)


def score_duplicate_pairs(pairs: pd.DataFrame, creatives: pd.DataFrame, embeddings: np.ndarray) -> pd.DataFrame:
    """Return pair-level similarity scores for analysis."""
    return build_pair_similarity_table(pairs, creatives, embeddings)