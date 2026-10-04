from __future__ import annotations

import json
import os
import re

from dotenv import load_dotenv
from openai import OpenAI

from .evidence import EvidenceItem
from .schema import LLMRationale

load_dotenv()


def _fallback_rationale(persona: str, score: float, evidence: list[EvidenceItem]) -> str:
    if not evidence:
        return f"The model suggested {persona}, but the available supporting evidence is limited."
    strongest = evidence[0]
    labels = ", ".join(str(label) for label in strongest.confirmed_personas) or "no confirmed persona label"
    return f"The model suggested {persona} with score {score:.2f}; the strongest retrieved example is {strongest.creative_id} with similarity {strongest.similarity:.2f} and confirmed label(s): {labels}."


def _build_prompt(persona: str, score: float, evidence: list[EvidenceItem]) -> str:
    evidence_text = json.dumps(
        [
            {
                "creative_id": item.creative_id,
                "similarity": round(item.similarity, 4),
                "near_duplicate": item.near_duplicate,
                "confirmed_personas": item.confirmed_personas,
            }
            for item in evidence
        ],
        indent=2,
    )

    return f"""You are explaining an ML model's persona suggestion for an advertisement.

Predicted persona: {persona}
Model score: {score:.4f}

Retrieved evidence:
{evidence_text}

Write a concise 1-2 sentence rationale.

STRICT RULES:
- Use only the supplied evidence.
- Only mention creative IDs present in the supplied evidence.
- Only mention persona labels present in the supplied evidence or the predicted persona.
- Do not invent information about the advertisement, audience, product, demographics, or creative content.
- Do not call a similarity "high" or "strong" unless the supplied numeric similarity supports that description.
- Do not infer semantic themes that are not present in the evidence.
- Return JSON only.

Schema:
{{"rationale": "..."}}"""


def _call_llm(prompt: str) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")

    client = OpenAI(api_key=api_key)
    response = client.responses.create(
        model=os.getenv("RATIONALE_MODEL", "gpt-4o-mini"),
        input=prompt,
    )
    return response.output_text


def _parse_rationale(response: str, evidence: list[EvidenceItem]) -> str:
    text = response.strip()

    # Handle models returning JSON inside a Markdown code fence.
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)

    data = json.loads(text)
    validated = LLMRationale.model_validate(data)
    rationale = validated.rationale.strip()

    allowed_ids = {item.creative_id for item in evidence}
    mentioned_ids = set(re.findall(r"\bc_\d+\b", rationale))

    unsupported_ids = mentioned_ids - allowed_ids

    if unsupported_ids:
        raise ValueError(
            f"Rationale references unsupported creative IDs: "
            f"{sorted(unsupported_ids)}"
        )

    return rationale


def generate_rationale(
    persona: str,
    score: float,
    evidence: list[EvidenceItem],
    mock_llm: bool | None = None,
) -> tuple[str, str]:
    if mock_llm is None:
        mock_llm = os.getenv("MOCK_LLM", "0") == "1"

    if mock_llm:
        return _fallback_rationale(persona, score, evidence), "template"

    prompt = _build_prompt(persona, score, evidence)

    try:
        response = _call_llm(prompt)
        return _parse_rationale(response, evidence), "llm"
    except Exception as exc:
       print(f"[LLM fallback] {type(exc).__name__}: {exc}")
       return _fallback_rationale(persona, score, evidence), "template"