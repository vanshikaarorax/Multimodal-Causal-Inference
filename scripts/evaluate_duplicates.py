from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from srcA.config import DUPLICATE_THRESHOLD
from srcA.data import load_creatives, load_duplicate_pairs
from srcA.embeddings import load_embeddings
from srcA.duplicate import score_duplicate_pairs


def main():
    creatives = load_creatives()
    pairs = load_duplicate_pairs()
    _, _, fused_embeddings, _ = load_embeddings()

    scored_pairs = score_duplicate_pairs(
        pairs,
        creatives,
        fused_embeddings,
    )

    y_true = scored_pairs["is_duplicate"].astype(int)

    # Freeze the operating threshold from config.
    y_pred = (
        scored_pairs["similarity"] >= DUPLICATE_THRESHOLD
    ).astype(int)

    duplicate_scores = scored_pairs.loc[
        y_true == 1, "similarity"
    ]

    non_duplicate_scores = scored_pairs.loc[
        y_true == 0, "similarity"
    ]

    print(f"Total pairs: {len(scored_pairs)}")
    print(f"Duplicate pairs: {len(duplicate_scores)}")
    print(f"Non-duplicate pairs: {len(non_duplicate_scores)}")

    print("\nSimilarity statistics:")
    print(
        pd.DataFrame({
            "duplicate": duplicate_scores.describe(),
            "non_duplicate": non_duplicate_scores.describe(),
        })
    )

    # ------------------------------------------------------------------
    # Held-out evaluation
    # ------------------------------------------------------------------
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    ).ravel()

    print("\nHeld-out duplicate detection evaluation:")
    print(f"Threshold: {DUPLICATE_THRESHOLD:.2f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1:        {f1:.4f}")
    print(f"Accuracy:  {accuracy:.4f}")

    print("\nConfusion matrix:")
    print(f"TN={tn}, FP={fp}, FN={fn}, TP={tp}")

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------
    plt.figure(figsize=(10, 6))

    plt.hist(
        non_duplicate_scores,
        bins=30,
        alpha=0.6,
        label="Non-duplicate",
    )

    plt.hist(
        duplicate_scores,
        bins=30,
        alpha=0.6,
        label="Duplicate",
    )

    plt.axvline(
        DUPLICATE_THRESHOLD,
        linestyle="--",
        label=f"Threshold = {DUPLICATE_THRESHOLD:.2f}",
    )

    plt.xlabel("Cosine similarity")
    plt.ylabel("Number of pairs")
    plt.title(
        "Fused CLIP Similarity: Duplicate vs Non-Duplicate Pairs"
    )
    plt.legend()
    plt.tight_layout()

    output_path = (
        PROJECT_ROOT
        / "artifacts"
        / "duplicate_similarity_distribution.png"
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"\nSaved visualization: {output_path}")


if __name__ == "__main__":
    main()