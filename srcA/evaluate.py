from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score


def build_multilabel_matrix(labels: list[list[str]], persona_names: list[str]) -> np.ndarray:
    matrix = np.zeros((len(labels), len(persona_names)), dtype=int)
    persona_to_index = {p: i for i, p in enumerate(persona_names)}
    for i, row in enumerate(labels):
        for persona in row:
            if persona in persona_to_index:
                matrix[i, persona_to_index[persona]] = 1
    return matrix


def evaluate_predictions(y_true: list[list[str]], y_pred: list[list[str]], persona_names: list[str]) -> dict[str, float]:
    true_matrix = build_multilabel_matrix(y_true, persona_names)
    pred_matrix = build_multilabel_matrix(y_pred, persona_names)
    return {
        "macro_f1": f1_score(true_matrix, pred_matrix, average="macro", zero_division=0),
        "micro_f1": f1_score(true_matrix, pred_matrix, average="micro", zero_division=0),
        "weighted_f1": f1_score(true_matrix, pred_matrix, average="weighted", zero_division=0),
        "hamming_loss": float(np.mean(true_matrix != pred_matrix)),
        "subset_accuracy": float(np.mean(np.all(true_matrix == pred_matrix, axis=1))),
    }


def predictions_from_probabilities(probabilities: np.ndarray, persona_names: list[str], threshold: float) -> list[list[str]]:
    return [[persona_names[i] for i, score in enumerate(row) if score >= threshold] for row in probabilities]


def evaluate_selective_prediction(probabilities: np.ndarray, y_true: list[list[str]], persona_names: list[str], threshold: float) -> dict[str, float]:
    top_indices = np.argmax(probabilities, axis=1)
    top_scores = probabilities[np.arange(len(probabilities)), top_indices]
    accepted = top_scores >= threshold
    correct = [persona_names[top_indices[i]] in y_true[i] for i in range(len(y_true)) if accepted[i]]
    return {"threshold": threshold, "coverage": float(np.mean(accepted)), "precision": float(np.mean(correct)) if correct else 0.0, "accepted": int(accepted.sum()), "abstained": int((~accepted).sum())}


def evaluate_coverage_precision(probabilities: np.ndarray, y_true: list[list[str]], persona_names: list[str], thresholds: np.ndarray | None = None) -> pd.DataFrame:
    thresholds = thresholds if thresholds is not None else np.arange(0.20, 0.91, 0.05)
    return pd.DataFrame([evaluate_selective_prediction(probabilities, y_true, persona_names, float(t)) for t in thresholds])


def evaluate_per_persona(y_true: list[list[str]], y_pred: list[list[str]], persona_names: list[str]) -> pd.DataFrame:
    true_matrix = build_multilabel_matrix(y_true, persona_names)
    pred_matrix = build_multilabel_matrix(y_pred, persona_names)

    rows = []

    for index, persona in enumerate(persona_names):
        rows.append({
            "persona": persona,
            "precision": precision_score(true_matrix[:, index], pred_matrix[:, index], zero_division=0),
            "recall": recall_score(true_matrix[:, index], pred_matrix[:, index], zero_division=0),
            "f1": f1_score(true_matrix[:, index], pred_matrix[:, index], zero_division=0),
            "support": int(true_matrix[:, index].sum()),
        })

    return pd.DataFrame(rows).sort_values("f1", ascending=False).reset_index(drop=True)