# srcA/abstention.py

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AbstentionConfig:
    suggest_threshold: float = 0.60
    caution_threshold: float = 0.45
    min_margin: float = 0.10
    min_personas: int = 1
    max_personas: int = 3


@dataclass(frozen=True)
class AbstentionResult:
    decision: str
    personas: list[dict[str, float | str]]
    confidence: float
    margin: float


def _rank_personas(persona_names: list[str], probabilities: np.ndarray, config: AbstentionConfig) -> list[dict[str, float | str]]:
    order = np.argsort(probabilities)[::-1]
    ranked = [{"name": persona_names[index], "score": float(probabilities[index])} for index in order]
    return ranked[: config.max_personas]


def decide_abstention(
    persona_names: list[str],
    probabilities: np.ndarray,
    config: AbstentionConfig | None = None,
) -> AbstentionResult:
    config = config or AbstentionConfig()

    probabilities = np.asarray(probabilities, dtype=float).reshape(-1)

    if len(persona_names) != len(probabilities):
        raise ValueError("persona_names and probabilities must have the same length.")

    if len(probabilities) == 0:
        raise ValueError("probabilities cannot be empty.")

    if not np.all(np.isfinite(probabilities)):
        raise ValueError("probabilities must contain only finite values.")

    order = np.argsort(probabilities)[::-1]
    top_index = order[0]
    top_score = float(probabilities[top_index])
    second_score = float(probabilities[order[1]]) if len(order) > 1 else 0.0
    margin = top_score - second_score

    ranked = _rank_personas(persona_names, probabilities, config)

    if top_score >= config.suggest_threshold and margin >= config.min_margin:
        decision = "suggest"
        selected = [item for item in ranked if float(item["score"]) >= config.suggest_threshold]
    elif top_score >= config.caution_threshold:
        decision = "suggest_with_caution"
        selected = [ranked[0]]
    else:
        decision = "abstain"
        selected = []

    if decision != "abstain" and len(selected) < config.min_personas:
        selected = [ranked[0]]

    return AbstentionResult(
        decision=decision,
        personas=selected,
        confidence=top_score,
        margin=margin,
    )


def apply_abstention(
    persona_names: list[str],
    probability_matrix: np.ndarray,
    config: AbstentionConfig | None = None,
) -> list[AbstentionResult]:
    probability_matrix = np.asarray(probability_matrix, dtype=float)

    if probability_matrix.ndim == 1:
        probability_matrix = probability_matrix.reshape(1, -1)

    if probability_matrix.ndim != 2:
        raise ValueError("probability_matrix must be a 1D or 2D array.")

    return [decide_abstention(persona_names, row, config) for row in probability_matrix]


def summarize_abstention(results: list[AbstentionResult]) -> dict[str, float | int]:
    total = len(results)

    if total == 0:
        return {"total": 0, "suggest": 0, "caution": 0, "abstain": 0, "coverage": 0.0}

    suggest = sum(result.decision == "suggest" for result in results)
    caution = sum(result.decision == "suggest_with_caution" for result in results)
    abstain = sum(result.decision == "abstain" for result in results)
    coverage = (suggest + caution) / total

    return {"total": total, "suggest": suggest, "caution": caution, "abstain": abstain, "coverage": coverage}