from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from srcA.config import FUSED_EMBEDDINGS_PATH
from srcA.abstention import AbstentionConfig, decide_abstention
from srcA.cli import _load_classifier


# ------------------------------------------------------------
# 1. Load dataset
# ------------------------------------------------------------
creatives = pd.read_parquet(PROJECT_ROOT / "creatives.parquet")
confirmed = pd.read_csv(PROJECT_ROOT / "confirmed_personas.csv")

confirmed_ids = set(confirmed["creative_id"].astype(str))

unlabelled_mask = ~creatives["creative_id"].astype(str).isin(confirmed_ids)
unlabelled = creatives.loc[unlabelled_mask].reset_index(drop=True)


# ------------------------------------------------------------
# 2. Load classifier using the SAME project loader as CLI
# ------------------------------------------------------------
classifier = _load_classifier()


# ------------------------------------------------------------
# 3. Load fused embeddings
# ------------------------------------------------------------
embeddings = np.load(FUSED_EMBEDDINGS_PATH)

creative_to_idx = {
    str(cid): i
    for i, cid in enumerate(creatives["creative_id"].astype(str))
}

unlabelled_indices = [
    creative_to_idx[str(cid)]
    for cid in unlabelled["creative_id"]
]

X_unlabelled = embeddings[unlabelled_indices]


# ------------------------------------------------------------
# 4. Predict with existing classifier
# ------------------------------------------------------------
probabilities = classifier.model.predict_proba(X_unlabelled)
persona_names = classifier.persona_names


# ------------------------------------------------------------
# 5. Existing abstention configuration
# ------------------------------------------------------------
config = AbstentionConfig()


# ------------------------------------------------------------
# 6. Run current abstention policy
# ------------------------------------------------------------
rows = []

for i, probs in enumerate(probabilities):
    order = np.argsort(probs)[::-1]

    top_idx = order[0]
    second_idx = order[1]

    top_score = float(probs[top_idx])
    second_score = float(probs[second_idx])
    margin = top_score - second_score

    result = decide_abstention(
        persona_names,
        probs,
        config=config,
    )

    rows.append({
        "creative_id": str(unlabelled.iloc[i]["creative_id"]),
        "top_persona": persona_names[top_idx],
        "top_score": top_score,
        "second_score": second_score,
        "margin": margin,
        "decision": result.decision,
    })

results = pd.DataFrame(rows)


# ------------------------------------------------------------
# 7. Decision distribution
# ------------------------------------------------------------
print()
print("=" * 70)
print("A1 — UNLABELLED POOL DIAGNOSTIC")
print("=" * 70)

print(f"Total creatives : {len(creatives)}")
print(f"Confirmed       : {len(confirmed_ids)}")
print(f"Unlabelled      : {len(unlabelled)}")

print()
print("=" * 70)
print("CURRENT DECISION DISTRIBUTION")
print("=" * 70)

counts = results["decision"].value_counts()

for decision in ["suggest", "suggest_with_caution", "abstain"]:
    count = int(counts.get(decision, 0))
    percentage = 100.0 * count / len(results)
    print(f"{decision:24s}{count:5d} ({percentage:6.2f}%)")


# ------------------------------------------------------------
# 8. Top-score distribution
# ------------------------------------------------------------
print()
print("=" * 70)
print("TOP SCORE DISTRIBUTION")
print("=" * 70)

print(
    results["top_score"]
    .quantile([0, .25, .50, .75, .90, .95, .99, 1.00])
    .to_string()
)


# ------------------------------------------------------------
# 9. Margin distribution
# ------------------------------------------------------------
print()
print("=" * 70)
print("MARGIN DISTRIBUTION")
print("=" * 70)

print(
    results["margin"]
    .quantile([0, .25, .50, .75, .90, .95, .99, 1.00])
    .to_string()
)


# ------------------------------------------------------------
# 10. Nearest confirmed creative
# ------------------------------------------------------------
confirmed_indices = [
    creative_to_idx[str(cid)]
    for cid in confirmed_ids
    if str(cid) in creative_to_idx
]

X_confirmed = embeddings[confirmed_indices]

confirmed_creative_ids = [
    str(creatives.iloc[idx]["creative_id"])
    for idx in confirmed_indices
]

# Embeddings are normalized, so dot product = cosine similarity.
similarities = X_unlabelled @ X_confirmed.T

nearest_indices = similarities.argmax(axis=1)
nearest_similarities = similarities.max(axis=1)

results["nearest_confirmed_similarity"] = nearest_similarities

results["nearest_confirmed_id"] = [
    confirmed_creative_ids[idx]
    for idx in nearest_indices
]


# ------------------------------------------------------------
# 11. Similarity distribution
# ------------------------------------------------------------
print()
print("=" * 70)
print("NEAREST CONFIRMED-CREATIVE SIMILARITY")
print("=" * 70)

print(
    results["nearest_confirmed_similarity"]
    .quantile([0, .25, .50, .75, .90, .95, .99, 1.00])
    .to_string()
)


# ------------------------------------------------------------
# 12. High-score + weak-evidence cases
# ------------------------------------------------------------
suspicious = results[
    (results["top_score"] >= 0.80)
    & (results["nearest_confirmed_similarity"] < 0.70)
].copy()

suspicious = suspicious.sort_values(
    ["top_score", "nearest_confirmed_similarity"],
    ascending=[False, True],
)

print()
print("=" * 70)
print("HIGH SCORE + WEAK CONFIRMED EVIDENCE")
print("=" * 70)

print(f"Count: {len(suspicious)}")

if not suspicious.empty:
    print()
    print(
        suspicious[
            [
                "creative_id",
                "top_persona",
                "top_score",
                "margin",
                "nearest_confirmed_similarity",
                "nearest_confirmed_id",
                "decision",
            ]
        ].head(20).to_string(index=False)
    )


# ------------------------------------------------------------
# 13. Very high-score + weak-evidence cases
# ------------------------------------------------------------
very_suspicious = results[
    (results["top_score"] >= 0.90)
    & (results["nearest_confirmed_similarity"] < 0.70)
].copy()

very_suspicious = very_suspicious.sort_values(
    ["top_score", "nearest_confirmed_similarity"],
    ascending=[False, True],
)

print()
print("=" * 70)
print("VERY HIGH SCORE + WEAK CONFIRMED EVIDENCE")
print("=" * 70)

print(f"Count: {len(very_suspicious)}")


# ------------------------------------------------------------
# 14. Add original creative metadata
# ------------------------------------------------------------
#
# We keep the model diagnostics above unchanged and attach
# every original column from creatives.parquet.
#
# This lets us inspect the actual creative information for
# suspicious predictions without guessing the dataset schema.
#
creative_metadata = unlabelled.copy()

creative_metadata["creative_id"] = (
    creative_metadata["creative_id"].astype(str)
)

results["creative_id"] = results["creative_id"].astype(str)

results_with_metadata = results.merge(
    creative_metadata,
    on="creative_id",
    how="left",
    suffixes=("", "_creative"),
)

suspicious_with_metadata = results_with_metadata[
    (results_with_metadata["top_score"] >= 0.80)
    & (results_with_metadata["nearest_confirmed_similarity"] < 0.70)
].copy()

suspicious_with_metadata = suspicious_with_metadata.sort_values(
    ["top_score", "nearest_confirmed_similarity"],
    ascending=[False, True],
)

very_suspicious_with_metadata = results_with_metadata[
    (results_with_metadata["top_score"] >= 0.90)
    & (results_with_metadata["nearest_confirmed_similarity"] < 0.70)
].copy()

very_suspicious_with_metadata = very_suspicious_with_metadata.sort_values(
    ["top_score", "nearest_confirmed_similarity"],
    ascending=[False, True],
)


# ------------------------------------------------------------
# 15. Save diagnostics
# ------------------------------------------------------------
results.to_csv(
    PROJECT_ROOT / "artifacts/a1_unlabelled_abstention_diagnostic.csv",
    index=False,
)

suspicious_with_metadata.to_csv(
    PROJECT_ROOT / "artifacts/a1_suspicious_predictions.csv",
    index=False,
)

very_suspicious_with_metadata.to_csv(
    PROJECT_ROOT / "artifacts/a1_very_suspicious_predictions.csv",
    index=False,
)

print()
print("=" * 70)
print("SAVED")
print("=" * 70)

print("artifacts/a1_unlabelled_abstention_diagnostic.csv")
print("artifacts/a1_suspicious_predictions.csv")
print("artifacts/a1_very_suspicious_predictions.csv")

print()
print("DONE")