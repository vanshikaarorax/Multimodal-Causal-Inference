from __future__ import annotations

from pathlib import Path
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
import numpy as np
import pandas as pd

from srcA.config import DUPLICATE_THRESHOLD
from srcA.embeddings import load_embeddings
from srcA.clustering import build_duplicate_edges



CREATIVES_PATH = PROJECT_ROOT / "creatives.parquet"
OUTPUT_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "duplicate_dev_candidates.csv"
)


def main() -> None:
    print("=" * 70)
    print("BUILD DUPLICATE CANDIDATES")
    print("=" * 70)

    print(
        f"\nUsing DUPLICATE_THRESHOLD = "
        f"{DUPLICATE_THRESHOLD}"
    )

    # ------------------------------------------------------------
    # 1. Load creatives
    # ------------------------------------------------------------
    creatives = pd.read_parquet(
        CREATIVES_PATH
    )

    print(
        f"Creatives: {len(creatives)}"
    )

    # ------------------------------------------------------------
    # 2. Load the SAME fused embeddings already produced
    # ------------------------------------------------------------
    _, _, fused_embeddings, metadata = load_embeddings()

    # Align creatives to embedding metadata.
    creatives = (
        creatives
        .set_index("creative_id")
        .loc[metadata["creative_id"]]
        .reset_index()
    )

    if len(creatives) != len(fused_embeddings):
        raise ValueError(
            "Creative count and embedding count do not match."
        )

    # ------------------------------------------------------------
    # 3. USE YOUR EXISTING CLUSTERING FUNCTION
    # ------------------------------------------------------------
    duplicate_edges = build_duplicate_edges(
        creatives=creatives,
        embeddings=fused_embeddings,
        threshold=DUPLICATE_THRESHOLD,
    )

    print(
        f"\nDuplicate edges found: "
        f"{len(duplicate_edges):,}"
    )

    # ------------------------------------------------------------
    # 4. Label the edges returned by clustering.py
    # ------------------------------------------------------------
    duplicate_edges["is_duplicate"] = 1

    duplicate_edges["label_notes"] = (
        "Threshold-based duplicate edge from "
        f"build_duplicate_edges "
        f"(similarity >= {DUPLICATE_THRESHOLD:.2f})."
    )

    # ------------------------------------------------------------
    # 5. Save
    # ------------------------------------------------------------
    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    duplicate_edges.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print(
        f"\nSaved to: {OUTPUT_PATH}"
    )

    print(
        "\nColumns:",
        duplicate_edges.columns.tolist(),
    )

    print(
        "\nLabel counts:"
    )

    print(
        duplicate_edges["is_duplicate"]
        .value_counts()
    )


if __name__ == "__main__":
    main()