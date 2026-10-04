# srcA/pipeline.py

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .abstention import AbstentionConfig, AbstentionResult, decide_abstention
from .evidence import EvidenceItem, retrieve_persona_evidence
from .rationale import generate_rationale
from .schema import PersonaSuggestion, validate_suggestion


@dataclass
class PartAResources:
    persona_names: list[str]
    reference_embeddings: np.ndarray
    reference_creative_ids: list[str]
    creative_personas: dict[str, list[str]]
    duplicate_threshold: float = 0.79


def run_persona_prediction(
    creative_id: str,
    embedding: np.ndarray,
    probability_matrix: np.ndarray,
    resources: PartAResources,
    abstention_config: AbstentionConfig | None = None,
    evidence_top_k: int = 3,
    evidence_candidate_k: int = 20,
    mock_llm: bool | None = None,
) -> PersonaSuggestion:
    probabilities = np.asarray(probability_matrix, dtype=float).reshape(-1)

    if len(probabilities) != len(resources.persona_names):
        raise ValueError("Probability count does not match persona count.")

    abstention: AbstentionResult = decide_abstention(
        persona_names=resources.persona_names,
        probabilities=probabilities,
        config=abstention_config,
    )

    if abstention.decision == "suggest":
        confidence = "high"
        decision_reason = (
            "The top persona score and separation from the next-best persona "
            "meet the configured suggestion thresholds."
        )
    elif abstention.decision == "suggest_with_caution":
        confidence = "medium"
        decision_reason = (
            "The model has enough evidence for a tentative suggestion, "
            "but the confidence is below the strong-suggestion threshold."
        )
    else:
        confidence = "low"
        decision_reason = (
            "The available model evidence is below the minimum threshold "
            "required for a reliable persona suggestion."
        )

    all_evidence: list[EvidenceItem] = []

    for persona in abstention.personas:
        persona_name = str(persona["name"])

        persona_evidence = retrieve_persona_evidence(
            query_embedding=embedding,
            reference_embeddings=resources.reference_embeddings,
            reference_creative_ids=resources.reference_creative_ids,
            creative_personas=resources.creative_personas,
            persona=persona_name,
            top_k=evidence_top_k,
            candidate_k=evidence_candidate_k,
            duplicate_threshold=resources.duplicate_threshold,
            exclude_creative_id=creative_id,
        )

        all_evidence.extend(persona_evidence)

    unique_evidence = {
        item.creative_id: item
        for item in all_evidence
    }

    evidence = sorted(
        unique_evidence.values(),
        key=lambda item: item.similarity,
        reverse=True,
    )

    if abstention.personas:
        top_persona = abstention.personas[0]

        rationale, rationale_source = generate_rationale(
            persona=str(top_persona["name"]),
            score=float(top_persona["score"]),
            evidence=evidence,
            mock_llm=mock_llm,
        )
    else:
        rationale = (
            "The available model evidence was not strong enough "
            "to make a reliable persona suggestion."
        )
        rationale_source = "template"

    result = {
        "creative_id": creative_id,
        "decision": abstention.decision,
        "decision_reason": decision_reason,
        "personas": abstention.personas,
        "evidence": [
            {
                "creative_id": item.creative_id,
                "similarity": round(item.similarity, 4),
                "near_duplicate": item.near_duplicate,
                "confirmed_personas": item.confirmed_personas,
            }
            for item in evidence
        ],
        "rationale": rationale,
        "rationale_source": rationale_source,
        "confidence": confidence,
    }

    return validate_suggestion(result)