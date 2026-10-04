import pytest
from pydantic import ValidationError

from srcA.schema import (
    validate_suggestion,
    suggestion_to_dict,
    suggestion_to_json,
)


def valid_payload():
    return {
        "creative_id": "creative_001",
        "decision": "suggest",
        "decision_reason": "Strong evidence from confirmed persona examples.",
        "personas": [
    {
        "name": "Traveler & Leisure",
        "score": 0.91,
    },
    {
        "name": "Tech Enthusiast",
        "score": 0.82,
    },
],
        "evidence": [
            {
                "creative_id": "creative_010",
                "similarity": 0.91,
                "near_duplicate": True,
                "confirmed_personas": ["Traveler & Leisure"],
            }
        ],
        "rationale": "The prediction is supported by a near-duplicate confirmed as Traveler & Leisure.",
        "rationale_source": "template",
        "confidence": "high",
    }


def test_valid_suggestion():
    result = validate_suggestion(valid_payload())

    assert result.creative_id == "creative_001"
    assert result.decision == "suggest"
    assert len(result.personas) == 2


def test_score_must_be_between_zero_and_one():
    payload = valid_payload()
    payload["personas"][0]["score"] = 1.5

    with pytest.raises(ValidationError):
        validate_suggestion(payload)


def test_invalid_decision_rejected():
    payload = valid_payload()
    payload["decision"] = "maybe"

    with pytest.raises(ValidationError):
        validate_suggestion(payload)


def test_duplicate_personas_rejected():
    payload = valid_payload()
    payload["personas"] = [
        {"name": "Tech Enthusiast", "score": 0.82},
        {"name": "Tech Enthusiast", "score": 0.64},
    ]

    with pytest.raises(ValidationError):
        validate_suggestion(payload)


def test_empty_rationale_rejected():
    payload = valid_payload()
    payload["rationale"] = "   "

    with pytest.raises(ValidationError):
        validate_suggestion(payload)


def test_extra_fields_rejected():
    payload = valid_payload()
    payload["unexpected"] = "bad"

    with pytest.raises(ValidationError):
        validate_suggestion(payload)


def test_serialization():
    suggestion = validate_suggestion(valid_payload())

    data = suggestion_to_dict(suggestion)
    json_text = suggestion_to_json(suggestion)

    assert data["creative_id"] == "creative_001"
    assert '"creative_id":"creative_001"' in json_text.replace(" ", "")