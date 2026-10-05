import numpy as np
import pytest

from srcA.evidence import (
    cosine_similarity,
    retrieve_evidence,
    retrieve_persona_evidence,
)


def test_cosine_similarity():
    query = np.array([1.0, 0.0])
    embeddings = np.array([[1.0, 0.0], [0.0, 1.0]])

    result = cosine_similarity(query, embeddings)

    assert result[0] == pytest.approx(1.0)
    assert result[1] == pytest.approx(0.0)


def test_retrieve_evidence_returns_ranked_results():
    query = np.array([1.0, 0.0])
    embeddings = np.array([
        [1.0, 0.0],
        [0.8, 0.2],
        [0.0, 1.0],
    ])
    ids = ["a", "b", "c"]

    result = retrieve_evidence(query, embeddings, ids, top_k=2)

    assert [item.creative_id for item in result] == ["a", "b"]
    assert result[0].similarity > result[1].similarity


def test_duplicate_flag():
    query = np.array([1.0, 0.0])
    embeddings = np.array([[1.0, 0.0], [0.0, 1.0]])
    ids = ["duplicate", "other"]

    result = retrieve_evidence(
        query,
        embeddings,
        ids,
        top_k=2,
        duplicate_threshold=0.88,
    )

    assert result[0].near_duplicate is True
    assert result[1].near_duplicate is False


def test_exclude_creative():
    query = np.array([1.0, 0.0])
    embeddings = np.array([[1.0, 0.0], [0.8, 0.2]])
    ids = ["current", "other"]

    result = retrieve_evidence(
        query,
        embeddings,
        ids,
        top_k=2,
        exclude_creative_id="current",
    )

    assert len(result) == 1
    assert result[0].creative_id == "other"


def test_persona_evidence_filtering():
    query = np.array([1.0, 0.0])
    embeddings = np.array([
        [1.0, 0.0],
        [0.9, 0.1],
        [0.0, 1.0],
    ])
    ids = ["a", "b", "c"]

    personas = {
        "a": ["Tech Enthusiast"],
        "b": ["Traveler & Leisure"],
        "c": ["Tech Enthusiast"],
    }

    result = retrieve_persona_evidence(
        query,
        embeddings,
        ids,
        personas,
        "Tech Enthusiast",
        top_k=2,
        candidate_k=3,
    )

    assert [item.creative_id for item in result] == ["a", "c"]