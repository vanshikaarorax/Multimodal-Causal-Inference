from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from srcA.config import ARTIFACTS_DIR, DUPLICATE_THRESHOLD
from srcA.data import load_creatives
from srcA.embeddings import load_embeddings
from srcA.clustering import build_duplicate_edges, build_duplicate_clusters, summarize_clusters, save_clusters


def main():
    creatives = load_creatives()
    _, _, fused_embeddings, _ = load_embeddings()

    print(f"Creatives: {len(creatives)}")
    print(f"Duplicate threshold: {DUPLICATE_THRESHOLD}")

    edges = build_duplicate_edges(
        creatives,
        fused_embeddings,
        threshold=DUPLICATE_THRESHOLD,
    )

    print(f"Duplicate edges: {len(edges):,}")

    clusters = build_duplicate_clusters(
        creatives,
        fused_embeddings,
        threshold=DUPLICATE_THRESHOLD,
    )

    summary = summarize_clusters(clusters)

    print(f"Total clusters: {summary.total_clusters:,}")
    print(f"Multi-creative clusters: {summary.multi_creative_clusters:,}")
    print(f"Largest cluster: {summary.largest_cluster_size}")

    output_path = ARTIFACTS_DIR / "duplicate_clusters.parquet"
    save_clusters(clusters, output_path)

    print(f"Saved clusters: {output_path}")


if __name__ == "__main__":
    main()