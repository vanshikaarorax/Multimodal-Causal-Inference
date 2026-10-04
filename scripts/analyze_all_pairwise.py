from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from srcA.embeddings import load_embeddings


def main():
    _, _, fused_embeddings, _ = load_embeddings()

    similarities = fused_embeddings @ fused_embeddings.T
    n = len(fused_embeddings)

    upper_i, upper_j = np.triu_indices(n, k=1)
    pair_similarities = similarities[upper_i, upper_j]

    print(f"Creatives: {n}")
    print(f"Unique pairs: {len(pair_similarities):,}")
    print("\nSimilarity statistics:")
    print(pd.Series(pair_similarities).describe(percentiles=[
        0.90, 0.95, 0.97, 0.98, 0.99, 0.995, 0.999
    ]))

    for threshold in [0.70, 0.75, 0.78, 0.79, 0.80, 0.82, 0.85, 0.90]:
        count = np.sum(pair_similarities >= threshold)
        print(f"Similarity >= {threshold:.2f}: {count:,} pairs")


if __name__ == "__main__":
    main()