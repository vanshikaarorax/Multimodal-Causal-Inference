from pathlib import Path
import sys
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from srcA.config import ARTIFACTS_DIR


def main():
    path = ARTIFACTS_DIR / "duplicate_clusters.parquet"
    clusters = pd.read_parquet(path)

    sizes = clusters.groupby("cluster_id")["creative_id"].size()

    print("\nCluster size distribution:")
    print(sizes.value_counts().sort_index().to_string())

    print("\nLargest clusters:")
    print(sizes.sort_values(ascending=False).head(20).to_string())

    print(f"\nTotal clusters: {len(sizes):,}")
    print(f"Multi-creative clusters: {(sizes > 1).sum():,}")
    print(f"Largest cluster: {sizes.max()}")


if __name__ == "__main__":
    main()