from __future__ import annotations

from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, field_validator

Decision = Literal["suggest", "suggest_with_caution", "abstain"]
RationaleSource = Literal["llm", "template"]
Confidence = Literal["low", "medium", "high"]


class PersonaPrediction(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(min_length=1)
    score: float = Field(ge=0.0, le=1.0)


class Evidence(BaseModel):
    model_config = ConfigDict(extra="forbid")
    creative_id: str = Field(min_length=1)
    similarity: float = Field(ge=-1.0, le=1.0)
    near_duplicate: bool
    confirmed_personas: list[str]


class PersonaSuggestion(BaseModel):
    model_config = ConfigDict(extra="forbid")
    creative_id: str = Field(min_length=1)
    decision: Decision
    decision_reason: str = Field(min_length=1)
    personas: list[PersonaPrediction]
    evidence: list[Evidence]
    rationale: str = Field(min_length=1)
    rationale_source: RationaleSource
    confidence: Confidence

    @field_validator("personas")
    @classmethod
    def validate_personas(cls, value: list[PersonaPrediction]) -> list[PersonaPrediction]:
        names = [persona.name for persona in value]
        if len(names) != len(set(names)):
            raise ValueError("Duplicate personas are not allowed.")
        return value

    @field_validator("rationale")
    @classmethod
    def validate_rationale(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Rationale cannot be empty.")
        return value

    @field_validator("decision_reason")
    @classmethod
    def validate_decision_reason(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Decision reason cannot be empty.")
        return value


class LLMRationale(BaseModel):
    model_config = ConfigDict(extra="forbid")
    rationale: str = Field(min_length=1)

    @field_validator("rationale")
    @classmethod
    def validate_rationale(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Rationale cannot be empty.")
        return value


def validate_suggestion(data: dict) -> PersonaSuggestion:
    return PersonaSuggestion.model_validate(data)


def suggestion_to_dict(suggestion: PersonaSuggestion) -> dict:
    return suggestion.model_dump()


def suggestion_to_json(suggestion: PersonaSuggestion) -> str:
    return suggestion.model_dump_json(indent=2)