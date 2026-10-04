
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from sklearn.neighbors import NearestNeighbors


@dataclass
class KNNPersonaModel:
    """kNN retrieval model for multi-label persona prediction."""
    model: NearestNeighbors
    labels: np.ndarray
    persona_names: list[str]
    n_neighbors: int
    weighting: str = "linear"


def train_knn_model(embeddings: np.ndarray, labels: list[list[str]], n_neighbors: int = 7, weighting: str = "linear") -> KNNPersonaModel:
    """Fit cosine kNN retrieval on labeled creative embeddings."""
    if n_neighbors < 1:
        raise ValueError("n_neighbors must be at least 1.")
    if weighting not in {"linear", "squared"}:
        raise ValueError("weighting must be 'linear' or 'squared'.")
    model = NearestNeighbors(n_neighbors=min(n_neighbors, len(embeddings)), metric="cosine")
    model.fit(embeddings)
    persona_names = sorted({persona for row in labels for persona in row})
    return KNNPersonaModel(model=model, labels=np.array(labels, dtype=object), persona_names=persona_names, n_neighbors=n_neighbors, weighting=weighting)


def predict_personas(model: KNNPersonaModel, embeddings: np.ndarray, threshold: float = 0.35) -> tuple[list[list[str]], np.ndarray]:
    """Predict multiple personas using similarity-weighted neighbor voting."""
    distances, indices = model.model.kneighbors(embeddings)
    similarities = np.maximum(1.0 - distances, 0.0)

    if model.weighting == "linear":
        weights = similarities
    elif model.weighting == "squared":
        weights = similarities ** 2
    else:
        raise ValueError(f"Unsupported weighting: {model.weighting}")

    probabilities = np.zeros((len(embeddings), len(model.persona_names)), dtype=float)
    persona_to_index = {persona: index for index, persona in enumerate(model.persona_names)}

    for row_index, neighbor_indices in enumerate(indices):
        row_weights = weights[row_index]
        weight_sum = row_weights.sum()
        if weight_sum <= 0:
            continue

        for neighbor_position, neighbor_index in enumerate(neighbor_indices):
            contribution = row_weights[neighbor_position] / weight_sum
            for persona in model.labels[neighbor_index]:
                probabilities[row_index, persona_to_index[persona]] += contribution

    predictions = [[model.persona_names[index] for index, score in enumerate(row) if score >= threshold] for row in probabilities]
    return predictions, probabilities


def predict_top_personas(model: KNNPersonaModel, embedding: np.ndarray, top_k: int = 3) -> list[dict]:
    """Return top persona predictions with similarity-weighted evidence."""
    _, probabilities = predict_personas(model, embedding.reshape(1, -1), threshold=0.0)
    scores = probabilities[0]
    order = np.argsort(scores)[::-1][:top_k]
    return [{"name": model.persona_names[index], "score": float(scores[index])} for index in order]
