from pathlib import Path
import sys
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from srcA.config import ARTIFACTS_DIR, SEED
from srcA.data import load_confirmed_personas, load_creatives
from srcA.embeddings import load_embeddings
from srcA.split import split_by_cluster, validate_no_cluster_leakage
from srcA.classifier import train_classifier, predict_personas
from srcA.evaluate import evaluate_coverage_precision
from srcA.abstention import AbstentionConfig, apply_abstention


def parse_personas(value):
    return [p.strip() for p in str(value).split("|") if p.strip()]


def main():
    creatives = load_creatives()
    confirmed = load_confirmed_personas()
    _, _, embeddings, metadata = load_embeddings()

    labeled = confirmed[["creative_id", "personas"]].copy()
    labeled = labeled.merge(creatives[["creative_id"]], on="creative_id", how="inner", validate="one_to_one")
    labeled["persona_list"] = labeled["personas"].apply(parse_personas)

    clusters = pd.read_parquet(ARTIFACTS_DIR / "duplicate_clusters.parquet")
    split = split_by_cluster(labeled, clusters, seed=SEED)
    validate_no_cluster_leakage(split.train, split.dev, split.test)

    index = {cid: i for i, cid in enumerate(metadata["creative_id"])}
    train_idx = [index[cid] for cid in split.train["creative_id"]]
    test_idx = [index[cid] for cid in split.test["creative_id"]]

    X_train, X_test = embeddings[train_idx], embeddings[test_idx]
    y_train, y_test = split.train["persona_list"].tolist(), split.test["persona_list"].tolist()

    persona_names = sorted({p for row in y_train for p in row})

    model = train_classifier(X_train, y_train, seed=SEED, C=30.0, class_weight="balanced")
    _, probabilities = predict_personas(model, X_test, threshold=0.0)

    curve = evaluate_coverage_precision(probabilities, y_test, persona_names)
    print("\nCoverage-Precision Curve:")
    print(curve.to_string(index=False))

    curve.to_csv(ARTIFACTS_DIR / "coverage_precision_curve.csv", index=False)

    config = AbstentionConfig(suggest_threshold=0.60, caution_threshold=0.45, min_margin=0.10)
    results = apply_abstention(persona_names, probabilities, config)

    accepted = [r for r in results if r.decision != "abstain"]
    correct = sum(any(p["name"] in true for p in r.personas) for r, true in zip(results, y_test) if r.decision != "abstain")

    print("\nCurrent Abstention Policy:")
    print(f"Suggest:  {sum(r.decision == 'suggest' for r in results)}")
    print(f"Caution:  {sum(r.decision == 'suggest_with_caution' for r in results)}")
    print(f"Abstain:  {sum(r.decision == 'abstain' for r in results)}")
    print(f"Coverage:  {len(accepted) / len(results):.4f}")
    print(f"Creative precision: {correct / len(accepted):.4f}" if accepted else "Creative precision: 0.0000")

    print("\nSaved:", ARTIFACTS_DIR / "coverage_precision_curve.csv")


if __name__ == "__main__":
    main()