from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer


@dataclass
class PersonaClassifier:
    """One-vs-rest multi-label classifier for persona prediction."""
    model: OneVsRestClassifier
    binarizer: MultiLabelBinarizer
    persona_names: list[str]


def train_classifier(embeddings: np.ndarray, labels: list[list[str]], seed: int = 42, C: float = 1.0, class_weight: str | None = "balanced") -> PersonaClassifier:
    """Train one-vs-rest logistic regression for persona prediction."""
    binarizer = MultiLabelBinarizer()
    y = binarizer.fit_transform(labels)
    base_model = LogisticRegression(max_iter=3000, random_state=seed, C=C, class_weight=class_weight)
    model = OneVsRestClassifier(base_model)
    model.fit(embeddings, y)
    return PersonaClassifier(model=model, binarizer=binarizer, persona_names=list(binarizer.classes_))


def predict_personas(model: PersonaClassifier, embeddings: np.ndarray, threshold: float = 0.5) -> tuple[list[list[str]], np.ndarray]:
    """Predict multiple personas and return per-persona probabilities."""
    probabilities = model.model.predict_proba(embeddings)
    predictions = [[model.persona_names[index] for index, score in enumerate(row) if score >= threshold] for row in probabilities]
    return predictions, probabilities


def predict_top_personas(model: PersonaClassifier, embedding: np.ndarray, top_k: int = 3) -> list[dict]:
    """Return top persona predictions for one creative."""
    probabilities = model.model.predict_proba(embedding.reshape(1, -1))[0]
    order = np.argsort(probabilities)[::-1][:top_k]
    return [{"name": model.persona_names[index], "score": float(probabilities[index])} for index in order]