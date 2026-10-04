# tests/test_abstention.py

import numpy as np
import pytest

from srcA.abstention import AbstentionConfig, decide_abstention, summarize_abstention


PERSONAS = ["Health & Wellness Seeker", "Tech Enthusiast", "Traveler & Leisure"]


def test_strong_prediction_returns_suggest():
    result = decide_abstention(PERSONAS, np.array([0.10, 0.20, 0.80]))
    assert result.decision == "suggest"
    assert result.personas[0]["name"] == "Traveler & Leisure"


def test_borderline_prediction_returns_caution():
    result = decide_abstention(PERSONAS, np.array([0.20, 0.30, 0.48]))
    assert result.decision == "suggest_with_caution"
    assert len(result.personas) == 1


def test_weak_prediction_abstains():
    result = decide_abstention(PERSONAS, np.array([0.31, 0.28, 0.30]))
    assert result.decision == "abstain"
    assert result.personas == []


def test_strong_score_with_small_margin_returns_caution():
    config = AbstentionConfig(suggest_threshold=0.60, caution_threshold=0.45, min_margin=0.10)
    result = decide_abstention(PERSONAS, np.array([0.20, 0.59, 0.61]), config)
    assert result.decision == "suggest_with_caution"


def test_invalid_probability_shape_raises():
    with pytest.raises(ValueError):
        decide_abstention(PERSONAS, np.array([0.5, 0.5]))


def test_invalid_probability_values_raise():
    with pytest.raises(ValueError):
        decide_abstention(PERSONAS, np.array([0.2, np.nan, 0.8]))


def test_abstention_summary():
    results = [
        decide_abstention(PERSONAS, np.array([0.10, 0.10, 0.80])),
        decide_abstention(PERSONAS, np.array([0.20, 0.30, 0.48])),
        decide_abstention(PERSONAS, np.array([0.31, 0.28, 0.30])),
    ]

    summary = summarize_abstention(results)

    assert summary["total"] == 3
    assert summary["suggest"] == 1
    assert summary["caution"] == 1
    assert summary["abstain"] == 1
    assert summary["coverage"] == pytest.approx(2 / 3)