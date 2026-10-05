# srcA/cli.py

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import typer

from .classifier import PersonaClassifier, predict_top_personas
from .config import (
    ARTIFACTS_DIR,
    CONFIRMED_PERSONAS_PATH,
    CREATIVES_PATH,
    DUPLICATE_THRESHOLD,
    FUSED_EMBEDDINGS_PATH,
    IMAGE_TEXT_MODEL,
)
from .data import load_confirmed_personas, load_creatives

from .pipeline import PartAResources, run_persona_prediction

app = typer.Typer(help="MemoLogs Part A persona suggestion CLI.")


def _load_classifier() -> PersonaClassifier:
    path = ARTIFACTS_DIR / "persona_classifier.pkl"

    if not path.exists():
        raise FileNotFoundError(f"Persona classifier not found: {path}")

    import joblib

    return joblib.load(path)


def _load_resources() -> PartAResources:
    creatives = load_creatives(CREATIVES_PATH)
    confirmed = load_confirmed_personas(CONFIRMED_PERSONAS_PATH)
    embeddings = np.load(FUSED_EMBEDDINGS_PATH)

    creative_personas = {
        str(row.creative_id): [p.strip() for p in str(row.personas).split("|") if p.strip()]
        for row in confirmed.itertuples()
    }

    return PartAResources(
        persona_names=sorted({persona for labels in creative_personas.values() for persona in labels}),
        reference_embeddings=embeddings,
        reference_creative_ids=creatives["creative_id"].astype(str).tolist(),
        creative_personas=creative_personas,
        duplicate_threshold=DUPLICATE_THRESHOLD,
    )


def _load_creative_embedding(creative_id: str) -> np.ndarray:
    creatives = load_creatives(CREATIVES_PATH)
    matches = creatives[creatives["creative_id"].astype(str) == creative_id]

    if matches.empty:
        raise ValueError(f"Creative not found: {creative_id}")

    index = matches.index[0]
    embeddings = np.load(FUSED_EMBEDDINGS_PATH)

    return embeddings[index]


@app.command()
def suggest(
    creative_id: str = typer.Option(..., "--creative-id", "-i"),
    mock_llm: bool = typer.Option(False, "--mock-llm"),
) -> None:
    resources = _load_resources()
    classifier = _load_classifier()
    embedding = _load_creative_embedding(creative_id)

    probabilities = classifier.model.predict_proba(embedding.reshape(1, -1))[0]

    result = run_persona_prediction(
        creative_id=creative_id,
        embedding=embedding,
        probability_matrix=probabilities,
        resources=resources,
        mock_llm=mock_llm,
    )

    typer.echo(json.dumps(result.model_dump(), indent=2))


if __name__ == "__main__":
    app()