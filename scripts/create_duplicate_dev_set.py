from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from srcA.data import load_creatives, load_duplicate_pairs
from srcA.embeddings import load_embeddings


SEED = 42
DEV_PAIRS = 100
CANDIDATE_POOL = 2000


def build_pair_candidates(creatives, embeddings, eval_pairs):
    rng = np.random.default_rng(SEED)
    creative_ids = creatives["creative_id"].to_numpy()
    n = len(creative_ids)

    eval_pairs_set = {
        tuple(sorted((row.creative_id_a, row.creative_id_b)))
        for row in eval_pairs.itertuples()
    }

    pair_indices = rng.integers(0, n, size=(CANDIDATE_POOL * 3, 2))
    pair_indices = pair_indices[pair_indices[:, 0] != pair_indices[:, 1]]

    pairs = []
    seen = set()

    for index_a, index_b in pair_indices:
        creative_a = creative_ids[index_a]
        creative_b = creative_ids[index_b]
        pair_key = tuple(sorted((creative_a, creative_b)))

        if pair_key in seen or pair_key in eval_pairs_set:
            continue

        similarity = float(np.dot(embeddings[index_a], embeddings[index_b]))

        pairs.append({
            "creative_id_a": creative_a,
            "creative_id_b": creative_b,
            "similarity": similarity,
        })
        seen.add(pair_key)

    return pd.DataFrame(pairs)


def sample_by_similarity(candidates):
    candidates = candidates.sort_values("similarity").reset_index(drop=True)
    candidates["similarity_bin"] = pd.qcut(candidates["similarity"], q=10, duplicates="drop")

    sampled = (
        candidates.groupby("similarity_bin", observed=True, group_keys=False)
        .apply(lambda group: group.sample(min(len(group), DEV_PAIRS // 10), random_state=SEED))
        .reset_index(drop=True)
    )

    return sampled.sample(frac=1, random_state=SEED).reset_index(drop=True)


def main():
    creatives = load_creatives()
    eval_pairs = load_duplicate_pairs()
    _, _, fused_embeddings, _ = load_embeddings()

    candidates = build_pair_candidates(creatives, fused_embeddings, eval_pairs)
    dev_pairs = sample_by_similarity(candidates).head(DEV_PAIRS)

    dev_pairs["is_duplicate"] = ""
    dev_pairs["label_notes"] = ""

    output_path = PROJECT_ROOT / "artifacts" / "duplicate_dev_candidates.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dev_pairs.to_csv(output_path, index=False)

    print(f"Generated {len(dev_pairs)} development candidates.")
    print(f"Saved: {output_path}")
    print("\nLabel each row manually:")
    print("is_duplicate = 1 for near-duplicate")
    print("is_duplicate = 0 for non-duplicate")
    print("Use label_notes for a short reason if useful.")


if __name__ == "__main__":
    main()