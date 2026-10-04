from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from srcA.config import SEED


@dataclass(frozen=True)
class SplitResult:
    """Train/dev/test split produced at the duplicate-cluster level."""
    train: pd.DataFrame
    dev: pd.DataFrame
    test: pd.DataFrame


def load_cluster_assignments(cluster_path) -> pd.DataFrame:
    """Load duplicate cluster assignments."""
    clusters = pd.read_parquet(cluster_path)

    required_columns = {"creative_id", "cluster_id"}
    missing = required_columns - set(clusters.columns)

    if missing:
        raise ValueError(f"Missing cluster columns: {sorted(missing)}")

    if clusters["creative_id"].duplicated().any():
        raise ValueError("A creative appears in multiple clusters.")

    return clusters[["creative_id", "cluster_id"]]


def split_by_cluster(
    labeled_data: pd.DataFrame,
    clusters: pd.DataFrame,
    dev_size: float = 0.15,
    test_size: float = 0.15,
    seed: int = SEED,
) -> SplitResult:
    """Split labeled creatives into train/dev/test without duplicate-cluster leakage."""
    if dev_size <= 0 or test_size <= 0 or dev_size + test_size >= 1:
        raise ValueError("dev_size and test_size must be positive and sum to less than 1.")

    data = labeled_data.merge(
        clusters,
        on="creative_id",
        how="left",
        validate="one_to_one",
    )

    if data["cluster_id"].isna().any():
        missing_ids = data.loc[data["cluster_id"].isna(), "creative_id"].tolist()
        raise ValueError(f"Missing cluster assignments for creatives: {missing_ids[:10]}")

    cluster_ids = data["cluster_id"].drop_duplicates().to_numpy()

    train_clusters, remaining_clusters = train_test_split(
        cluster_ids,
        test_size=dev_size + test_size,
        random_state=seed,
    )

    relative_test_size = test_size / (dev_size + test_size)

    dev_clusters, test_clusters = train_test_split(
        remaining_clusters,
        test_size=relative_test_size,
        random_state=seed,
    )

    train_cluster_set = set(train_clusters)
    dev_cluster_set = set(dev_clusters)
    test_cluster_set = set(test_clusters)

    if train_cluster_set & dev_cluster_set:
        raise RuntimeError("Duplicate cluster leakage detected between train and dev.")

    if train_cluster_set & test_cluster_set:
        raise RuntimeError("Duplicate cluster leakage detected between train and test.")

    if dev_cluster_set & test_cluster_set:
        raise RuntimeError("Duplicate cluster leakage detected between dev and test.")

    train = data[data["cluster_id"].isin(train_cluster_set)].copy()
    dev = data[data["cluster_id"].isin(dev_cluster_set)].copy()
    test = data[data["cluster_id"].isin(test_cluster_set)].copy()

    return SplitResult(train=train, dev=dev, test=test)


def validate_no_cluster_leakage(
    train: pd.DataFrame,
    dev: pd.DataFrame,
    test: pd.DataFrame,
) -> None:
    """Verify that no duplicate cluster appears across any split."""
    train_clusters = set(train["cluster_id"])
    dev_clusters = set(dev["cluster_id"])
    test_clusters = set(test["cluster_id"])

    train_dev_overlap = train_clusters & dev_clusters
    train_test_overlap = train_clusters & test_clusters
    dev_test_overlap = dev_clusters & test_clusters

    if train_dev_overlap:
        raise AssertionError(
            f"Cluster leakage detected between train/dev: {len(train_dev_overlap)} clusters."
        )

    if train_test_overlap:
        raise AssertionError(
            f"Cluster leakage detected between train/test: {len(train_test_overlap)} clusters."
        )

    if dev_test_overlap:
        raise AssertionError(
            f"Cluster leakage detected between dev/test: {len(dev_test_overlap)} clusters."
        )


def summarize_split(split: SplitResult) -> pd.DataFrame:
    """Return basic train/dev/test split statistics."""
    return pd.DataFrame({
        "split": ["train", "dev", "test"],
        "creatives": [
            len(split.train),
            len(split.dev),
            len(split.test),
        ],
        "clusters": [
            split.train["cluster_id"].nunique(),
            split.dev["cluster_id"].nunique(),
            split.test["cluster_id"].nunique(),
        ],
    })