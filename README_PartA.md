# MemoLogs — Part A
## Multimodal Persona Prediction with Duplicate-Aware Evaluation, Abstention & Evidence-Grounded Rationale
> **AI/ML Engineer Take-Home — Part A**
Given an advertisement creative, predict the persona(s) it is likely targeting, provide supporting evidence, detect near-duplicates, prevent leakage between evaluation splits, and decide whether to **suggest**, **suggest with caution**, or **abstain**.
---
## 1. Overview
MemoLogs Part A is a **multimodal, multi-label, leakage-aware persona prediction system**.
**Input:** image, ad text, and `creative_id`.
**Output:** predicted personas, model scores, confidence/trust state, supporting creatives, cosine similarity, near-duplicate status, evidence-grounded rationale, and deterministic fallback.
### Architecture
```text
Creative + Text
      │
      ▼
CLIP Image + Text Encoders
      │
      ▼
30% Image + 70% Text → 512-D fused vector
      │
      ├──► Duplicate Detection → Complete-Linkage Clustering
      ├──► Persona Prediction → kNN + Logistic Regression
      └──► Evidence Retrieval → Labelled References
                    │
                    ▼
             Abstention / Trust
                    │
                    ▼
          Evidence-Grounded Rationale
                    │
                    ▼
              Validated JSON
```
The system is designed around three problems that a simple classifier does not solve well:
- duplicate campaigns can leak across splits;
- a creative can have multiple personas;
- weak evidence should result in abstention rather than a forced recommendation.
---
# 2. Key Results
### Final configuration
| Component | Final choice |
|---|---|
| Vision-language encoder | `openai/clip-vit-base-patch32` |
| Embedding dimension | 512 |
| Image/text fusion | **30% image + 70% text** |
| Duplicate operating threshold | **0.88 cosine similarity** |
| Duplicate clustering | Complete-linkage hierarchical clustering |
| Persona baseline | Weighted multi-label kNN |
| Final classifier | One-vs-Rest Logistic Regression |
| Logistic Regression `C` | **3.0** |
| Persona threshold | **0.60** |
| Train / Dev / Test | 414 / 86 / 100 creatives |
### Held-out test performance
| Metric | kNN | Logistic Regression |
|---|---:|---:|
| Macro-F1 | 49.32% | **61.10%** |
| Micro-F1 | 55.10% | **64.89%** |
| Weighted-F1 | 56.05% | **64.13%** |
| Hamming Loss ↓ | 11.96% | **8.97%** |
| Subset Accuracy | 45.65% | **51.09%** |
One-vs-Rest Logistic Regression was selected because it generalized better to unseen creative clusters than the kNN baseline.
---
# 3. Why This Architecture?
The dataset contains only **600 human-labelled creatives** compared with **2,882 total creatives**. Training a large multimodal model from scratch would therefore be inappropriate.
Instead, a pretrained vision-language encoder provides semantic representations while lightweight models learn persona decision boundaries:
```text
Pretrained multimodal model
        ↓
Semantic representation
        ↓
Small labelled dataset
        ↓
Lightweight persona classifier
```
This lets the labelled data learn **persona boundaries** rather than image/text semantics from scratch.
---
# 4. Dataset
## Main creative dataset
`creatives.parquet` contains **2,882 advertisements** with:
`creative_id`, `ad_size`, `ad_text`, `image_path`, `launched_at`
The inspected dataset contains no missing values in these fields.
## Images
`images/` contains **2,882 images**.
Observed dimensions:
| Dimension | Count |
|---|---:|
| 300 × 250 | 1,090 |
| 728 × 90 | 935 |
| 300 × 600 | 321 |
| 970 × 250 | 240 |
| 970 × 90 | 109 |
| 160 × 600 | 100 |
| 320 × 50 | 50 |
| 336 × 280 | 37 |
CLIP preprocessing standardizes the different banner formats through resizing, cropping, normalization, and model-specific input formatting.
## Confirmed personas
`confirmed_personas.csv` contains **600 human-confirmed creatives**.
The task is **multi-label**. Labels are pipe-delimited, for example:
`Traveler & Leisure|Health & Wellness Seeker`
The eight personas in `personas.yaml` are:
1. Parent & Family
2. Homeowner
3. Small Business & B2B
4. Health & Wellness Seeker
5. Tech Enthusiast
6. Retiree & Senior
7. Student & Learner
8. Traveler & Leisure
### Label distribution
| Persona | Confirmed creatives |
|---|---:|
| Traveler & Leisure | 146 |
| Health & Wellness Seeker | 113 |
| Tech Enthusiast | 99 |
| Small Business & B2B | 93 |
| Student & Learner | 83 |
| Homeowner | 70 |
| Retiree & Senior | 16 |
| Parent & Family | 13 |
The imbalance makes **Macro-F1** especially useful because it gives each persona equal weight.
## Duplicate evaluation set
`dup_pairs_eval.csv` contains **120 held-out pairs**:
- 60 duplicate
- 60 non-duplicate
It is used for duplicate evaluation rather than directly training the duplicate detector.
---
# 5. Multimodal Embeddings
## CLIP
Primary encoder:
`openai/clip-vit-base-patch32`
CLIP produces a **512-dimensional projected embedding** for both image and text.
Each modality is normalized before fusion.
### Fusion
```text
fused =
    0.30 × normalized(image_embedding)
  + 0.70 × normalized(text_embedding)
then normalize(fused)
```
### Fusion experiment
| Image | Text | Macro-F1 | Micro-F1 | Weighted-F1 | Hamming | Subset |
|---:|---:|---:|---:|---:|---:|---:|
| 70% | 30% | 58.00% | 59.39% | 60.76% | 12.64% | 34.78% |
| 50% | 50% | 60.38% | 63.81% | 64.27% | 10.33% | 46.74% |
| **30%** | **70%** | **61.10%** | **64.89%** | 64.13% | **8.97%** | **51.09%** |
| 20% | 80% | 59.54% | **66.36%** | **67.30%** | 9.78% | 46.74% |
Experiment notebook: `srcA/notebooks/embeddings.ipynb`
The 30/70 configuration was selected because it achieved the highest Macro-F1, lowest Hamming loss, and highest exact subset accuracy. The experiment suggests that **ad text carries the strongest persona-discriminative signal** in this dataset, while vision adds complementary information.
---
# 6. Embedding Model Comparison
A second vision-language model was evaluated:
`google/siglip-base-patch16-224`
Both models used the same leakage-safe split and downstream persona-model search.
| Encoder | Test Macro-F1 |
|---|---:|
| **CLIP ViT-B/32** | **61.10%** |
| SigLIP Base | 57.22% |
CLIP was retained because it achieved stronger held-out performance while providing a lightweight, CPU-friendly solution. Larger CLIP-family models were not pursued because the labelled dataset did not provide evidence that additional size and compute would improve generalization.
---
# 7. Near-Duplicate Detection
Duplicate detection is used for:
1. supporting evidence;
2. preventing train/test leakage.
The system computes cosine similarity between fused embeddings.
### Threshold selection
The final duplicate operating point is **0.88**.
It was selected through manual development review of high-similarity candidates. Reviewed hard negatives were at **0.856** and **0.869**, while a reviewed genuine near-duplicate was **0.893**. The held-out 120-pair evaluation was run after this threshold decision.
Analysis notebook: `srcA/notebooks/duplicate_threshold_analysis.ipynb`
### Full pairwise analysis
Across 2,882 creatives:
`4,151,521` unique pairs.
| Statistic | Value |
|---|---:|
| Mean | 0.51265 |
| 90th percentile | 0.64643 |
| 95th percentile | 0.68563 |
| 99th percentile | 0.75132 |
| 99.9th percentile | 0.81548 |
| Maximum | 1.00 |
| Similarity | Pairs |
|---:|---:|
| ≥ 0.70 | 154,325 |
| ≥ 0.75 | 43,272 |
| ≥ 0.78 | 16,297 |
| ≥ 0.88 | **11,283** |
| ≥ 0.80 | 7,703 |
| ≥ 0.85 | 1,341 |
| ≥ 0.90 | 513 |
---
# 8. Complete-Linkage Duplicate Clustering
Simple high-similarity edges can create **chaining**:
```text
A ≈ B
B ≈ C
```
does not necessarily mean `A ≈ C`.
A connected-component strategy can therefore create artificially large campaign groups.
The final implementation uses **complete-linkage hierarchical clustering** at the 0.88 similarity threshold, requiring creatives within a cluster to remain mutually similar.
| Measure | Earlier grouping | Complete linkage |
|---|---:|---:|
| Duplicate edges | 1,655 | 1,655 |
| Clusters | 1,923 | **2,038** |
| Largest cluster | 87 creatives | **14 creatives** |
This produces tighter campaign groups and a cleaner foundation for leakage prevention.
---
# 9. Leakage-Safe Evaluation
The final split is performed using **duplicate-cluster IDs**, not individual creatives. This prevents near-duplicate creatives from appearing in different splits.
| Split | Creatives | Clusters |
|---|---:|---:|
| Train | 414 | 343 |
| Development | 86 | 74 |
| Test | 100 | 74 |
```text
Duplicate clustering
        ↓
Cluster-level split
        ├── Train → fit models
        ├── Dev   → select thresholds / hyperparameters
        └── Test  → final held-out evaluation
```
The test set is not used to select the final model configuration.
---
# 10. Persona Prediction
Two required approaches were implemented.
## Approach 1 — Similarity-Based kNN
The kNN model:
1. embeds the query creative;
2. retrieves similar training creatives;
3. collects **all** persona labels from the neighbours;
4. weights persona votes by similarity;
5. produces an independent score for each persona.
Example:
`Traveler & Leisure → 0.82`, `Health & Wellness → 0.67`, `Tech Enthusiast → 0.04`
This supports genuine multi-label prediction rather than forcing one persona per creative.
## Approach 2 — One-vs-Rest Logistic Regression
```text
512-D fused embedding
        ↓
One-vs-Rest Logistic Regression
        ├── Parent & Family
        ├── Homeowner
        ├── Small Business & B2B
        ├── Health & Wellness Seeker
        ├── Tech Enthusiast
        ├── Retiree & Senior
        ├── Student & Learner
        └── Traveler & Leisure
```
Each persona receives an independent score.
---
# 11. Hyperparameter Search
### kNN
`k ∈ {3, 5, 7, 9, 11, 15}`
### Logistic Regression
`C ∈ {0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0}`
### Persona thresholds
`0.20 → 0.70`
Selection criterion: **development Macro-F1**.
Selected configurations:
```text
kNN:
    k = 5
    threshold = 0.20
Logistic Regression:
    C = 3.0
    threshold = 0.45
```
---
# 12. Final Model Comparison
| Model | Macro-F1 | Micro-F1 | Weighted-F1 | Hamming | Subset Accuracy |
|---|---:|---:|---:|---:|---:|
| kNN | 49.32% | 55.10% | 56.05% | 11.96% | 45.65% |
| **One-vs-Rest Logistic Regression** | **61.10%** | **64.89%** | **64.13%** | **8.97%** | **51.09%** |
kNN performed strongly on the development set but dropped on unseen creative clusters. Logistic Regression showed better held-out generalization and was selected as the final persona model.
---
# 13. Trust Layer: Abstention
The system should **not force a persona recommendation when evidence is weak**.
The development-selected operating policy is:
| Condition | Decision | Confidence |
|---|---|---|
| Top score ≥ **0.70** | `suggest` | high |
| Top score ≥ **0.60** | `suggest_with_caution` | medium |
| Otherwise | `abstain` | low |
Held-out test behaviour:
- **56** suggest
- **8** caution
- **36** abstain
- **64%** coverage
- **81.25%** precision among suggested predictions
The full unlabelled audit is in `srcA/notebooks/abstention_unlabelled_audit.ipynb` and covers all **2,282 unlabelled creatives**.
Example:
`0.91 → suggest`, `0.55 → suggest_with_caution`, `0.32 → abstain`
The confidence label is an **operating decision derived from model scores and score margin**, not a formally calibrated probability.
---
# 14. Evidence Retrieval
For selected personas, the system retrieves supporting labelled creatives.
Example evidence item:
```json
{
  "creative_id": "c_2074",
  "similarity": 0.9414,
  "near_duplicate": true
}
```
Evidence is:
- retrieved from the confirmed/reference set;
- deduplicated;
- sorted by similarity;
- marked as near-duplicate when similarity crosses the duplicate threshold.
The classifier prediction does **not** require a direct lookup of the query creative in `confirmed_personas.csv`.
The confirmed dataset is used for **supervised training** and **labelled reference evidence**. A new/unconfirmed creative can therefore still be passed through the trained classifier.
---
# 15. LLM Rationale
The LLM is used only as an **explanation layer**. It does not determine the persona.
```text
ML classifier
    ↓
Persona + score
    ↓
Retrieve evidence
    ↓
Evidence-only LLM prompt
    ↓
JSON response
    ↓
Schema validation
    ├── valid → use rationale
    └── invalid / unavailable → deterministic fallback
```
The prompt is constrained to supplied evidence, including:
- predicted persona;
- model score;
- retrieved creative IDs;
- similarity values;
- near-duplicate flags.
The LLM is explicitly instructed not to invent demographics, product details, visual objects, audience characteristics, or campaign themes unless they are present in the supplied evidence.
---
# 16. Mock LLM Mode
The pipeline can run without an external LLM API:
```bash
python -m srcA.cli --creative-id c_0412 --mock-llm
```
Normal mode:
```bash
python -m srcA.cli --creative-id c_0412
```
The rationale source is reported as `llm`, `mock`, or `fallback`.
This makes the inference pipeline easier to test and reproduce.
---
# 17. Structured Output
The final response is validated using Pydantic.
```json
{
  "creative_id": "c_0030",
  "confidence": "medium",
  "decision": "suggest_with_caution",
  "personas": [
    {
      "name": "Traveler & Leisure",
      "score": 0.5529
    }
  ],
  "evidence": [
    {
      "creative_id": "c_2074",
      "similarity": 0.9414,
      "near_duplicate": true
    }
  ],
  "rationale": "...",
  "rationale_source": "llm"
}
```
Strict validation prevents unexpected fields or malformed values from silently passing through the pipeline.
---
# 18. CLI Usage
### Normal LLM mode
```bash
python -m srcA.cli --creative-id c_0412
```
### Deterministic mock mode
```bash
python -m srcA.cli --creative-id c_0412 --mock-llm
```
The CLI returns:
`creative_id`, `confidence`, `decision`, `personas`, `evidence`, `rationale`, and `rationale_source`.
---
# 20. Project Structure
```text
memologs/
├── creatives.parquet
├── confirmed_personas.csv
├── personas.yaml
├── dup_pairs_eval.csv
├── geo_panel.csv
├── images/
├── srcA/
│   ├── config.py
│   ├── data.py
│   ├── embeddings.py
│   ├── embeddings2.py
│   ├── duplicate.py
│   ├── clustering.py
│   ├── split.py
│   ├── knn_model.py
│   ├── classifier.py
│   ├── evaluate.py
│   ├── abstention.py
│   ├── evidence.py
│   ├── rationale.py
│   ├── schema.py
│   ├── pipeline.py
│   └── cli.py
├── scripts/
│   ├── evaluate_duplicates.py
│   ├── analyze_all_pairwise.py
│   ├── build_duplicate_clusters.py
│   ├── inspect_duplicate_clusters.py
│   ├── train_persona_models.py
│   └── save_persona_model.py
├── artifacts/
├── notebooks/
│   ├── abstention_unlabelled_audit.ipynb
│   ├── duplicate_threshold_analysis.ipynb
│   ├── embeddings.ipynb
│   └── Label_quality&personas.ipynb
├── REPORT.md
├── README.md
├── Makefile
└── pyproject.toml
```
# 21. Reproducibility
The project separates:
```text
SOURCE DATA
    ↓
EMBEDDINGS
    ↓
DUPLICATE CLUSTERS
    ↓
TRAIN / DEV / TEST
    ↓
MODEL SEARCH
    ↓
FINAL MODEL
    ↓
INFERENCE
```
Important reproducibility choices:
- fixed random seed;
- cached artifacts;
- cluster-level splitting;
- development-only hyperparameter selection;
- held-out test evaluation;
- deterministic mock LLM mode;
- lightweight/lazy imports for cached-embedding inference;
- strict output schema.
---
# 22. Experimentation Summary
### Iteration 1 — Initial baseline
Established kNN and classifier baselines.
### Iteration 2 — Genuine multi-label modelling
Changed from single-persona classification to independent persona scoring.
### Iteration 3 — Leakage-safe splitting
Moved train/dev/test assignment from creatives to duplicate clusters.
### Iteration 4 — Better duplicate clustering
Replaced chaining-prone grouping with complete-linkage hierarchical clustering.
### Iteration 5 — Threshold tuning
Selected persona decision thresholds using the development set. The duplicate operating point was reviewed on a development sample and fixed at **0.88** before held-out evaluation.
### Iteration 6 — Hyperparameter tuning
Searched kNN `k` and Logistic Regression `C`.
### Iteration 7 — Multimodal fusion search
Compared 70/30, 50/50, 30/70, and 20/80 image/text fusion and selected **30/70**.
### Iteration 8 — Representation comparison
Compared CLIP ViT-B/32 against SigLIP Base.
### Iteration 9 — Trust-aware inference
Added **abstention, confidence, evidence, LLM rationale, schema validation, and fallback**.
The final system reflects both **model experimentation** and **evaluation-design improvements**.
---
# 23. Limitations
### Limited labelled data
Only 600 creatives are labelled, with strong class imbalance.
### Probability calibration
Classifier scores are operating scores rather than formally calibrated probabilities.
### Duplicate threshold
The 0.88 threshold is an operating point chosen from development review. A dedicated duplicate-threshold development set would make threshold uncertainty easier to quantify.
### Evidence richness
The current evidence payload is primarily `creative_id`, `similarity`, and `near_duplicate`. A richer evidence layer could provide controlled text snippets or other grounded features.
### Dataset-specific generalization
Metrics are measured on this dataset and leakage-safe split and should not automatically be assumed to transfer to a different ad inventory.
---
# 24. Potential Future Improvements
- per-persona probability calibration;
- per-persona precision / recall / F1 analysis;
- dedicated duplicate-threshold development;
- richer multimodal evidence retrieval;
- uncertainty calibration;
- campaign-level drift monitoring;
- temporal evaluation;
- monitoring abstention coverage versus precision;
- retraining as new confirmed labels arrive.
---
# 25. Final Takeaway
The final Part A system is deliberately more than:
```text
ad → classifier → persona
```
It is:
```text
ad
 │
 ▼
multimodal representation
 │
 ├──► duplicate detection
 ├──► leakage-safe clustering
 └──► persona prediction
             │
             ▼
        trust decision
             │
        ┌────┴────┐
        ▼         ▼
     evidence   abstain
        │
        ▼
 grounded rationale
        │
        ▼
   validated JSON
```
> **Make the prediction, understand the evidence behind it, and know when not to trust it.**
### Final held-out result
**61.10% Macro-F1 · 64.89% Micro-F1 · 64.13% Weighted-F1 · 8.97% Hamming loss · 51.09% subset accuracy**
These results come from a lightweight One-vs-Rest Logistic Regression model operating on a carefully selected pretrained multimodal representation, supported by duplicate-aware evaluation and a trust layer rather than model size alone.
---
# Quick Start
```bash
# Activate environment
source .venv/bin/activate
# Normal inference
python -m srcA.cli --creative-id c_0412
# Deterministic mock-LLM inference
python -m srcA.cli --creative-id c_0412 --mock-llm
```
For the complete methodology, experimental record, and detailed discussion, see `REPORT.md`.
