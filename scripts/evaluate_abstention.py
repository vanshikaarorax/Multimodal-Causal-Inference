# scripts/evaluate_abstention.py

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
from srcA.abstention import AbstentionConfig, apply_abstention


def parse_personas(value):
    return [p.strip() for p in str(value).split("|") if p.strip()]


def evaluate_policy(results, y_true):
    accepted = [r for r in results if r.decision != "abstain"]

    if not accepted:
        return 0.0, 0.0, 0, len(results)

    correct = sum(
        any(p["name"] in true for p in r.personas)
        for r, true in zip(results, y_true)
        if r.decision != "abstain"
    )

    coverage = len(accepted) / len(results)
    precision = correct / len(accepted)

    return coverage, precision, len(accepted), len(results) - len(accepted)


def main():
    # ---------------------------------------------------------
    # 1. Load data
    # ---------------------------------------------------------
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

    # ---------------------------------------------------------
    # 2. Leakage-safe cluster split
    # ---------------------------------------------------------
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

    # ---------------------------------------------------------
    # 3. Build matrices
    # ---------------------------------------------------------
    index = {
        cid: i
        for i, cid in enumerate(metadata["creative_id"])
    }

    train_idx = [index[cid] for cid in split.train["creative_id"]]
    dev_idx = [index[cid] for cid in split.dev["creative_id"]]
    test_idx = [index[cid] for cid in split.test["creative_id"]]

    X_train = embeddings[train_idx]
    X_dev = embeddings[dev_idx]
    X_test = embeddings[test_idx]

    y_train = split.train["persona_list"].tolist()
    y_dev = split.dev["persona_list"].tolist()
    y_test = split.test["persona_list"].tolist()

    persona_names = sorted({
        p for row in y_train for p in row
    })

    # ---------------------------------------------------------
    # 4. Train selected LR model
    # ---------------------------------------------------------
    model = train_classifier(
        X_train,
        y_train,
        seed=SEED,
        C=30.0,
        class_weight="balanced",
    )

    _, dev_probabilities = predict_personas(
        model,
        X_dev,
        threshold=0.0,
    )

    _, test_probabilities = predict_personas(
        model,
        X_test,
        threshold=0.0,
    )

    # ---------------------------------------------------------
    # 5. Select abstention policy on DEV
    # ---------------------------------------------------------
    candidates = []

    for suggest_threshold in [0.70, 0.75, 0.80, 0.85]:
        for caution_threshold in [0.55, 0.60, 0.65]:
            for min_margin in [0.00, 0.05, 0.10, 0.15]:

                config = AbstentionConfig(
                    suggest_threshold=suggest_threshold,
                    caution_threshold=caution_threshold,
                    min_margin=min_margin,
                )

                results = apply_abstention(
                    persona_names,
                    dev_probabilities,
                    config,
                )

                coverage, precision, accepted, abstained = (
                    evaluate_policy(results, y_dev)
                )

                candidates.append({
                    "suggest_threshold": suggest_threshold,
                    "caution_threshold": caution_threshold,
                    "min_margin": min_margin,
                    "coverage": coverage,
                    "precision": precision,
                    "accepted": accepted,
                    "abstained": abstained,
                })

    dev_results = pd.DataFrame(candidates)

    # Avoid selecting a trivial policy with almost no coverage.
    eligible = dev_results[
        dev_results["coverage"] >= 0.50
    ].copy()

    best = eligible.sort_values(
        ["precision", "coverage"],
        ascending=[False, False],
    ).iloc[0]

    final_config = AbstentionConfig(
        suggest_threshold=float(best["suggest_threshold"]),
        caution_threshold=float(best["caution_threshold"]),
        min_margin=float(best["min_margin"]),
    )

    # ---------------------------------------------------------
    # 6. Evaluate locked policy on TEST exactly once
    # ---------------------------------------------------------
    test_results = apply_abstention(
        persona_names,
        test_probabilities,
        final_config,
    )

    coverage, precision, accepted, abstained = evaluate_policy(
        test_results,
        y_test,
    )

    suggest_count = sum(
        r.decision == "suggest"
        for r in test_results
    )

    caution_count = sum(
        r.decision == "suggest_with_caution"
        for r in test_results
    )

    abstain_count = sum(
        r.decision == "abstain"
        for r in test_results
    )

    # ---------------------------------------------------------
    # 7. Print results
    # ---------------------------------------------------------
    print("\n" + "=" * 70)
    print("A1 ABSTENTION EVALUATION")
    print("=" * 70)

    print(f"\nTrain creatives : {len(y_train)}")
    print(f"Dev creatives   : {len(y_dev)}")
    print(f"Test creatives  : {len(y_test)}")

    print("\n" + "=" * 70)
    print("DEV-SELECTED ABSTENTION POLICY")
    print("=" * 70)

    print(f"Suggest threshold : {final_config.suggest_threshold:.2f}")
    print(f"Caution threshold : {final_config.caution_threshold:.2f}")
    print(f"Minimum margin    : {final_config.min_margin:.2f}")

    print(f"\nDEV coverage  : {best['coverage']:.4f}")
    print(f"DEV precision : {best['precision']:.4f}")

    print("\n" + "=" * 70)
    print("FINAL HELD-OUT TEST")
    print("=" * 70)

    print(f"Suggest              : {suggest_count}")
    print(f"Suggest with caution : {caution_count}")
    print(f"Abstain              : {abstain_count}")
    print(f"Coverage             : {coverage:.4f} ({coverage:.2%})")
    print(f"Creative precision   : {precision:.4f} ({precision:.2%})")

    # ---------------------------------------------------------
    # 8. Save artifacts
    # ---------------------------------------------------------
    dev_results.to_csv(
        ARTIFACTS_DIR / "abstention_dev_search.csv",
        index=False,
    )

    policy_summary = pd.DataFrame([{
        "suggest_threshold": final_config.suggest_threshold,
        "caution_threshold": final_config.caution_threshold,
        "min_margin": final_config.min_margin,
        "dev_coverage": float(best["coverage"]),
        "dev_precision": float(best["precision"]),
        "test_suggest": suggest_count,
        "test_suggest_with_caution": caution_count,
        "test_abstain": abstain_count,
        "test_coverage": coverage,
        "test_precision": precision,
        "total_test_creatives": len(y_test),
    }])

    policy_summary.to_csv(
        ARTIFACTS_DIR / "abstention_policy_evaluation.csv",
        index=False,
    )

    print("\nSaved:")
    print(ARTIFACTS_DIR / "abstention_dev_search.csv")
    print(ARTIFACTS_DIR / "abstention_policy_evaluation.csv")
    print("\nDONE")


if __name__ == "__main__":
    main()