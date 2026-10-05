from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from srcA.data import load_creatives
from srcA.embeddings import load_embeddings
from srcA.clustering import build_all_pairwise_similarities





OUTPUT_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "similarity_dev_set.csv"
)


def main():
    print("Loading creatives...")
    creatives = load_creatives()

    print("Loading embeddings...")
    _, _, fused_embeddings, metadata = load_embeddings()

    print(f"Creatives: {len(creatives):,}")
    print(f"Fused embeddings: {fused_embeddings.shape}")

    # Make sure creatives and embeddings correspond one-to-one.
    if len(creatives) != len(fused_embeddings):
        raise ValueError(
            f"Creative count ({len(creatives)}) does not match "
            f"embedding count ({len(fused_embeddings)})."
        )

    print("\nBuilding all pairwise similarities...")

    pairs = build_all_pairwise_similarities(
        creatives=creatives,
        embeddings=fused_embeddings,
    )

    expected_pairs = (
        len(creatives) * (len(creatives) - 1) // 2
    )

    print(f"Expected pairs: {expected_pairs:,}")
    print(f"Generated pairs: {len(pairs):,}")

    if len(pairs) != expected_pairs:
        raise ValueError(
            f"Expected {expected_pairs:,} pairs, "
            f"but generated {len(pairs):,}."
        )

    if pairs["similarity"].isna().any():
        raise ValueError(
            "Found missing similarity values."
        )

    print("\nSimilarity statistics:")
    print(
        f"Min:    {pairs['similarity'].min():.6f}"
    )
    print(
        f"Mean:   {pairs['similarity'].mean():.6f}"
    )
    print(
        f"Median: {pairs['similarity'].median():.6f}"
    )
    print(
        f"Max:    {pairs['similarity'].max():.6f}"
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("\nSaving all pairwise similarities...")
    print(f"Output: {OUTPUT_PATH}")

    pairs.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print("\n✓ Similarity dev set created successfully.")
    print(f"✓ Total pairs: {len(pairs):,}")
    print(f"✓ Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()