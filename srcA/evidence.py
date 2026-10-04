from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class EvidenceItem:
    creative_id: str
    similarity: float
    near_duplicate: bool
    confirmed_personas: list[str]


def cosine_similarity(query_embedding: np.ndarray, embeddings: np.ndarray) -> np.ndarray:
    query = np.asarray(query_embedding, dtype=float).reshape(1, -1)
    matrix = np.asarray(embeddings, dtype=float)

    query_norm = np.linalg.norm(query, axis=1, keepdims=True)
    matrix_norm = np.linalg.norm(matrix, axis=1, keepdims=True)

    if np.any(query_norm == 0) or np.any(matrix_norm == 0):
        raise ValueError("Embeddings must have non-zero norm.")

    query = query / query_norm
    matrix = matrix / matrix_norm
    return (matrix @ query.T).reshape(-1)


def retrieve_evidence(

    query_embedding: np.ndarray,
    reference_embeddings: np.ndarray,
    reference_creative_ids: list[str],
    creative_personas: dict[str, list[str]] | None = None,
    top_k: int = 5,
    duplicate_threshold: float = 0.79,
    exclude_creative_id: str | None = None,
) -> list[EvidenceItem]:
    if len(reference_embeddings) != len(reference_creative_ids):
        raise ValueError("Embeddings and creative IDs must have the same length.")
    if creative_personas is None:
       creative_personas = {}
    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    similarities = cosine_similarity(query_embedding, reference_embeddings)
    order = np.argsort(similarities)[::-1]

    evidence: list[EvidenceItem] = []

    for index in order:
        creative_id = str(reference_creative_ids[index])

        if exclude_creative_id is not None and creative_id == exclude_creative_id:
            continue

        similarity = float(similarities[index])
        labels = creative_personas.get(creative_id, [])

        evidence.append(EvidenceItem(creative_id=creative_id, similarity=similarity, near_duplicate=similarity >= duplicate_threshold, confirmed_personas=list(labels)))

        if len(evidence) >= top_k:
            break

    return evidence


def filter_evidence_by_persona(evidence: list[EvidenceItem], creative_personas: dict[str, list[str]], persona: str) -> list[EvidenceItem]:
    return [item for item in evidence if persona in creative_personas.get(item.creative_id, [])]


def retrieve_persona_evidence(
    query_embedding: np.ndarray,
    reference_embeddings: np.ndarray,
    reference_creative_ids: list[str],
    creative_personas: dict[str, list[str]],
    persona: str,
    top_k: int = 3,
    candidate_k: int = 20,
    duplicate_threshold: float = 0.79,
    exclude_creative_id: str | None = None,
) -> list[EvidenceItem]:
    candidates = retrieve_evidence(
        query_embedding=query_embedding,
        reference_embeddings=reference_embeddings,
        reference_creative_ids=reference_creative_ids,
        creative_personas=creative_personas,
        top_k=candidate_k,
        duplicate_threshold=duplicate_threshold,
        exclude_creative_id=exclude_creative_id,
    )
    persona_candidates = filter_evidence_by_persona(candidates, creative_personas, persona)
    return persona_candidates[:top_k]


def evidence_to_dict(evidence: list[EvidenceItem]) -> list[dict[str, object]]:
    return [
        {
            "creative_id": item.creative_id,
            "similarity": round(item.similarity, 4),
            "near_duplicate": item.near_duplicate,
            "confirmed_personas": item.confirmed_personas,
        }
        for item in evidence
    ]