from pathlib import Path
import pandas as pd


# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

CREATIVES_PATH = PROJECT_ROOT / "creatives.parquet"
SUSPICIOUS_PATH = (
    PROJECT_ROOT / "artifacts/a1_suspicious_predictions.csv"
)


# ------------------------------------------------------------
# 1. Load data
# ------------------------------------------------------------
creatives = pd.read_parquet(CREATIVES_PATH)
suspicious = pd.read_csv(SUSPICIOUS_PATH)


# ------------------------------------------------------------
# 2. Keep only the columns we need from the creative dataset
# ------------------------------------------------------------
creative_info = creatives[
    [
        "creative_id",
        "ad_size",
        "ad_text",
        "image_path",
        "launched_at",
    ]
].copy()

creative_info["creative_id"] = creative_info["creative_id"].astype(str)
suspicious["creative_id"] = suspicious["creative_id"].astype(str)


# ------------------------------------------------------------
# 3. Merge model diagnostics with actual creative information
# ------------------------------------------------------------
review = suspicious.merge(
    creative_info,
    on="creative_id",
    how="left",
)


# ------------------------------------------------------------
# 4. Sort by highest confidence + weakest evidence
# ------------------------------------------------------------
review = review.sort_values(
    ["top_score", "nearest_confirmed_similarity"],
    ascending=[False, True],
).reset_index(drop=True)


# ------------------------------------------------------------
# 5. Print human-review table
# ------------------------------------------------------------
print("=" * 100)
print("A1 SUSPICIOUS ADS FOR HUMAN REVIEW")
print("=" * 100)

print(f"Total suspicious ads: {len(review)}")

print()


# ------------------------------------------------------------
# 6. Print top 20 candidates with actual ad text
# ------------------------------------------------------------
for i, row in review.head(20).iterrows():

    print("=" * 100)
    print(f"REVIEW #{i + 1}")
    print("=" * 100)

    print(f"Creative ID                 : {row['creative_id']}")
    print(f"Predicted persona           : {row['top_persona']}")
    print(f"Model score                 : {row['top_score']:.4f}")
    print(f"Second-best score           : {row['second_score']:.4f}")
    print(f"Margin                      : {row['margin']:.4f}")
    print(
        "Nearest confirmed similarity: "
        f"{row['nearest_confirmed_similarity']:.4f}"
    )
    print(
        f"Nearest confirmed creative  : "
        f"{row['nearest_confirmed_id']}"
    )
    print(f"Decision                    : {row['decision']}")
    print(f"Ad size                     : {row['ad_size']}")
    print(f"Image path                  : {row['image_path']}")
    print(f"Launched at                 : {row['launched_at']}")

    print()
    print("AD TEXT")
    print("-" * 100)

    ad_text = row["ad_text"]

    if pd.isna(ad_text) or not str(ad_text).strip():
        print("[NO AD TEXT]")
    else:
        print(str(ad_text).strip())

    print()


# ------------------------------------------------------------
# 7. Save complete review file
# ------------------------------------------------------------
output_path = (
    PROJECT_ROOT / "artifacts/a1_suspicious_for_review.csv"
)

review.to_csv(output_path, index=False)

print("=" * 100)
print("Saved:")
print(output_path)
print("=" * 100)