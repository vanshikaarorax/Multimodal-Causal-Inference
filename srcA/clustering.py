from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage

from srcA.config import DUPLICATE_THRESHOLD


@dataclass(frozen=True)
class ClusterSummary:
    total_creatives: int
    total_clusters: int
    multi_creative_clusters: int
    largest_cluster_size: int


def compute_pairwise_distances(embeddings: np.ndarray) -> np.ndarray:
    """Compute condensed cosine-distance matrix for clustering."""
    embeddings = np.asarray(embeddings, dtype=np.float32)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True).clip(min=1e-12)

    similarity_matrix = embeddings @ embeddings.T
    similarity_matrix = np.clip(similarity_matrix, -1.0, 1.0)

    index_a, index_b = np.triu_indices(len(embeddings), k=1)
    distances = 1.0 - similarity_matrix[index_a, index_b]

    return distances


def build_duplicate_edges(
    creatives: pd.DataFrame,
    embeddings: np.ndarray,
    threshold: float = DUPLICATE_THRESHOLD,
) -> pd.DataFrame:
    """Build pairwise near-duplicate edges above the similarity threshold."""
    embeddings = np.asarray(embeddings, dtype=np.float32)
    embeddings = embeddings / np.linalg.norm(
        embeddings,
        axis=1,
        keepdims=True,
    ).clip(min=1e-12)

    similarity_matrix = embeddings @ embeddings.T
    similarity_matrix = np.clip(similarity_matrix, -1.0, 1.0)

    index_a, index_b = np.triu_indices(len(embeddings), k=1)
    similarities = similarity_matrix[index_a, index_b]

    duplicate_mask = similarities >= threshold
    creative_ids = creatives["creative_id"].to_numpy()

    return pd.DataFrame({
        "creative_id_a": creative_ids[index_a[duplicate_mask]],
        "creative_id_b": creative_ids[index_b[duplicate_mask]],
        "similarity": similarities[duplicate_mask],
    })


def build_duplicate_clusters(
    creatives: pd.DataFrame,
    embeddings: np.ndarray,
    threshold: float = DUPLICATE_THRESHOLD,
) -> pd.DataFrame:
    """Build order-independent duplicate clusters using complete linkage."""
    distances = compute_pairwise_distances(embeddings)

    linkage_matrix = linkage(
        distances,
        method="complete",
    )

    max_distance = 1.0 - threshold

    cluster_labels = fcluster(
        linkage_matrix,
        t=max_distance,
        criterion="distance",
    )

    clusters = pd.DataFrame({
        "creative_id": creatives["creative_id"].to_numpy(),
        "cluster_id": cluster_labels - 1,
    })

    cluster_sizes = clusters.groupby("cluster_id")["creative_id"].transform("size")
    clusters["cluster_size"] = cluster_sizes

    return clusters


def summarize_clusters(clusters: pd.DataFrame) -> ClusterSummary:
    """Return summary statistics for duplicate clusters."""
    cluster_sizes = clusters.groupby("cluster_id")["creative_id"].size()

    return ClusterSummary(
        total_creatives=len(clusters),
        total_clusters=len(cluster_sizes),
        multi_creative_clusters=int((cluster_sizes > 1).sum()),
        largest_cluster_size=int(cluster_sizes.max()),
    )


def save_clusters(clusters: pd.DataFrame, output_path) -> None:
    """Save duplicate cluster assignments."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    clusters.to_parquet(output_path, index=False)