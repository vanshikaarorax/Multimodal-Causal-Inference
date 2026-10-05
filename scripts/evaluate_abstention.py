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
    # ------------------------------------------------------------
    # 1. Load labelled data + embeddings
    # ------------------------------------------------------------
    creatives = load_creatives()
    confirmed = load_confirmed_personas()
    _, _, embeddings, metadata = load_embeddings()

    labeled = confirmed[["creative_id", "personas"]].copy()
    labeled = labeled.merge(
        creatives[["creative_id"]],
        on="creative_id",
        how="inner",
        validate="one_to_one",
    )
    labeled["persona_list"] = labeled["personas"].apply(parse_personas)

    # ------------------------------------------------------------
    # 2. Leakage-safe train/test split
    # ------------------------------------------------------------
    clusters = pd.read_parquet(
        ARTIFACTS_DIR / "duplicate_clusters.parquet"
    )

    split = split_by_cluster(
        labeled,
        clusters,
        seed=SEED,
    )

    validate_no_cluster_leakage(
        split.train,
        split.dev,
        split.test,
    )

    # ------------------------------------------------------------
    # 3. Build train/test matrices
    # ------------------------------------------------------------
    index = {
        cid: i
        for i, cid in enumerate(metadata["creative_id"])
    }

    train_idx = [
        index[cid]
        for cid in split.train["creative_id"]
    ]

    test_idx = [
        index[cid]
        for cid in split.test["creative_id"]
    ]

    X_train = embeddings[train_idx]
    X_test = embeddings[test_idx]

    y_train = split.train["persona_list"].tolist()
    y_test = split.test["persona_list"].tolist()

    persona_names = sorted({
        p
        for row in y_train
        for p in row
    })

    # ------------------------------------------------------------
    # 4. Train final classifier
    # ------------------------------------------------------------
    model = train_classifier(
        X_train,
        y_train,
        seed=SEED,
        C=30.0,
        class_weight="balanced",
    )

    _, probabilities = predict_personas(
        model,
        X_test,
        threshold=0.0,
    )

    # ------------------------------------------------------------
    # 5. Score-threshold coverage / precision curve
    # ------------------------------------------------------------
    curve = evaluate_coverage_precision(
        probabilities,
        y_test,
        persona_names,
    )

    print()
    print("=" * 70)
    print("A1 ABSTENTION EVALUATION")
    print("=" * 70)

    print()
    print("Evaluation set")
    print("-" * 70)
    print(f"Test creatives : {len(y_test)}")
    print("Split          : leakage-safe cluster split")

    print()
    print("=" * 70)
    print("1. SCORE-THRESHOLD COVERAGE-PRECISION TRADE-OFF")
    print("=" * 70)

    print(curve.to_string(index=False))

    curve.to_csv(
        ARTIFACTS_DIR / "coverage_precision_curve.csv",
        index=False,
    )

    # ------------------------------------------------------------
    # 6. Current production abstention policy
    # ------------------------------------------------------------
    config = AbstentionConfig(
        suggest_threshold=0.60,
        caution_threshold=0.45,
        min_margin=0.10,
    )

    results = apply_abstention(
        persona_names,
        probabilities,
        config,
    )

    accepted = [
        r
        for r in results
        if r.decision != "abstain"
    ]

    correct = sum(
        any(
            p["name"] in true
            for p in r.personas
        )
        for r, true in zip(results, y_test)
        if r.decision != "abstain"
    )

    total = len(results)
    coverage = len(accepted) / total if total else 0.0
    precision = (
        correct / len(accepted)
        if accepted
        else 0.0
    )

    suggest_count = sum(
        r.decision == "suggest"
        for r in results
    )

    caution_count = sum(
        r.decision == "suggest_with_caution"
        for r in results
    )

    abstain_count = sum(
        r.decision == "abstain"
        for r in results
    )

    # ------------------------------------------------------------
    # 7. Print current policy results
    # ------------------------------------------------------------
    print()
    print("=" * 70)
    print("2. CURRENT ABSTENTION POLICY")
    print("=" * 70)

    print(f"Suggest                  : {suggest_count}")
    print(f"Suggest with caution     : {caution_count}")
    print(f"Abstain                  : {abstain_count}")
    print(f"Coverage                 : {coverage:.4f} ({coverage:.2%})")
    print(f"Creative precision       : {precision:.4f} ({precision:.2%})")

    # ------------------------------------------------------------
    # 8. Save actual policy evaluation
    # ------------------------------------------------------------
    policy_summary = pd.DataFrame([
        {
            "suggest_threshold": config.suggest_threshold,
            "caution_threshold": config.caution_threshold,
            "min_margin": config.min_margin,
            "suggest": suggest_count,
            "suggest_with_caution": caution_count,
            "abstain": abstain_count,
            "coverage": coverage,
            "precision": precision,
            "total_test_creatives": total,
        }
    ])

    policy_summary.to_csv(
        ARTIFACTS_DIR / "abstention_policy_evaluation.csv",
        index=False,
    )

    # ------------------------------------------------------------
    # 9. Final summary
    # ------------------------------------------------------------
    print()
    print("=" * 70)
    print("3. SAVED ARTIFACTS")
    print("=" * 70)

    print(ARTIFACTS_DIR / "coverage_precision_curve.csv")
    print(ARTIFACTS_DIR / "abstention_policy_evaluation.csv")

    print()
    print("DONE")


if __name__ == "__main__":
    main()