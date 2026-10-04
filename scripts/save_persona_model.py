from __future__ import annotations
from pathlib import Path
import joblib
import numpy as np
import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from srcA.classifier import train_classifier
from srcA.config import ARTIFACTS_DIR, FUSED_EMBEDDINGS_PATH, SEED
from srcA.data import load_confirmed_personas, load_creatives
from srcA.split import load_cluster_assignments, split_by_cluster


def main():
    creatives = load_creatives()
    confirmed = load_confirmed_personas()
    embeddings = np.load(FUSED_EMBEDDINGS_PATH)

    clusters = load_cluster_assignments(
        ARTIFACTS_DIR / "duplicate_clusters.parquet"
    )

    split = split_by_cluster(
        creatives,
        clusters,
        seed=SEED,
    )

    train_ids = set(split.train["creative_id"].astype(str))

    labels_by_id = {
        str(row.creative_id): [
            persona.strip()
            for persona in str(row.personas).split("|")
            if persona.strip()
        ]
        for row in confirmed.itertuples()
    }

    train_mask = creatives["creative_id"].astype(str).isin(train_ids)

    train_creatives = creatives.loc[train_mask]

    labeled_mask = train_creatives["creative_id"].astype(str).isin(labels_by_id)

    train_creatives = train_creatives[labeled_mask]

    train_indices = train_creatives.index.to_numpy()
    train_embeddings = embeddings[train_indices]

    train_labels = [
        labels_by_id[str(creative_id)]
        for creative_id in train_creatives["creative_id"]
    ]

    if len(train_embeddings) != len(train_labels):
        raise ValueError(
            f"Embedding/label mismatch: "
            f"{len(train_embeddings)} embeddings vs "
            f"{len(train_labels)} labels."
        )

    model = train_classifier(
        train_embeddings,
        train_labels,
        seed=SEED,
        C=30.0,
        class_weight="balanced",
    )

    output_path = ARTIFACTS_DIR / "persona_classifier.pkl"
    joblib.dump(model, output_path)

    print(f"Training creatives: {len(train_embeddings)}")
    print("Model: One-vs-Rest Logistic Regression")
    print("C: 30.0")
    print("Class weight: balanced")
    print(f"Saved model: {output_path}")


if __name__ == "__main__":
    main()