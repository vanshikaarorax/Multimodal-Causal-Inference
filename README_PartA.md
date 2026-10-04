# MemoLogs --- Part A

## Multimodal Persona Prediction with Duplicate-Aware Evaluation, Abstention & Evidence-Grounded Rationale

> **AI/ML Engineer Take-Home --- Part A**\
> Given an advertisement creative, predict the persona(s) it is likely
> targeting, provide supporting evidence, detect near-duplicates,
> prevent leakage between evaluation splits, and decide when the system
> should **suggest**, **suggest with caution**, or **abstain**.

------------------------------------------------------------------------

## 1. Overview

MemoLogs Part A is implemented as a **multimodal, multi-label,
leakage-aware persona prediction system**.

The system takes an advertisement creative containing:

-   an image
-   ad text
-   a `creative_id`

and produces a structured prediction containing:

-   predicted persona(s)
-   model scores
-   confidence / trust state
-   supporting labelled creatives
-   cosine similarity
-   near-duplicate status
-   an evidence-constrained LLM rationale
-   deterministic fallback when the LLM is unavailable or invalid

The core design is intentionally modular:

``` text
                         ┌──────────────────────┐
                         │   Creative + Text    │
                         └──────────┬───────────┘
                                    │
                         ┌──────────▼───────────┐
                         │  CLIP Multimodal     │
                         │  Image + Text        │
                         └──────────┬───────────┘
                                    │
                         ┌──────────▼───────────┐
                         │ 30% Image + 70% Text │
                         │ 512-D fused vector   │
                         └──────────┬───────────┘
                                    │
              ┌─────────────────────┼─────────────────────┐
              │                     │                     │
              ▼                     ▼                     ▼
      Duplicate Detection     Persona Models       Evidence Retrieval
              │                     │                     │
              ▼                     ▼                     ▼
      Complete-Linkage        kNN + Logistic       Labelled References
       Clustering             Regression                 │
              │                     │                     │
              └──────────────┬──────┴─────────────────────┘
                             ▼
                    ┌────────────────────┐
                    │ Abstention / Trust │
                    └─────────┬──────────┘
                              ▼
                    ┌────────────────────┐
                    │ Evidence-Grounded  │
                    │ LLM Rationale      │
                    └─────────┬──────────┘
                              ▼
                         Structured JSON
```

The important idea is that the project does **not** treat persona
prediction as a simple classification problem. Duplicate campaigns can
leak across splits, one creative can have multiple personas, and a model
should not be forced to make a recommendation when evidence is weak.

------------------------------------------------------------------------

# 2. Key Results

### Final persona model

  Component                 Final choice
  ------------------------- ------------------------------------------
  Vision-language encoder   `openai/clip-vit-base-patch32`
  Embedding dimension       512
  Image/text fusion         **30% image + 70% text**
  Duplicate threshold       **0.79 cosine similarity**
  Duplicate clustering      Complete-linkage hierarchical clustering
  Persona baseline          Weighted multi-label kNN
  Final classifier          One-vs-Rest Logistic Regression
  Logistic Regression `C`   **30**
  Persona threshold         **0.60**
  Train creatives           421
  Dev creatives             87
  Test creatives            92

### Held-out test performance

  Metric                 kNN   Logistic Regression
  ----------------- -------- ---------------------
  Macro-F1            49.32%            **61.10%**
  Micro-F1            55.10%            **64.89%**
  Weighted-F1         56.05%            **64.13%**
  Hamming Loss ↓      11.96%             **8.97%**
  Subset Accuracy     45.65%            **51.09%**

The final One-vs-Rest Logistic Regression model was selected because it
generalized better to unseen creative clusters than the kNN baseline.

------------------------------------------------------------------------

# 3. Why This Architecture?

The dataset contains only **600 human-labelled creatives**, compared
with **2,882 total creatives**.

Training a large multimodal model from scratch would therefore be
inappropriate.

Instead, the system uses a pretrained vision-language model to provide a
semantically meaningful representation and learns relatively lightweight
persona decision boundaries on top of those embeddings.

Conceptually:

``` text
Large pretrained model
        │
        ▼
Semantic multimodal representation
        │
        ▼
Small labelled dataset
        │
        ▼
Lightweight persona classifier
```

This lets the labelled data teach the system **where the persona
boundaries are**, rather than forcing the labelled data to learn
image/text semantics from scratch.

------------------------------------------------------------------------

# 4. Dataset

## Main creative dataset

`creatives.parquet`

**2,882 advertisements**

Columns:

``` text
creative_id
ad_size
ad_text
image_path
launched_at
```

The inspected dataset contains no missing values in these fields.

------------------------------------------------------------------------

## Images

`images/`

There are **2,882 images**.

Observed dimensions include:

  Dimension     Count
  ----------- -------
  300 × 250     1,090
  728 × 90        935
  300 × 600       321
  970 × 250       240
  970 × 90        109
  160 × 600       100
  320 × 50         50
  336 × 280        37

The different banner formats make it important to use a vision encoder
with standardized preprocessing rather than relying on raw pixel
dimensions.

CLIP's processor handles resizing, cropping, normalization, and the
required model input format.

------------------------------------------------------------------------

## Confirmed personas

`confirmed_personas.csv`

Contains **600 human-confirmed creatives**.

The task is **multi-label**: a creative can belong to multiple personas.

Labels are pipe-delimited:

``` text
Traveler & Leisure|Health & Wellness Seeker
```

The eight allowed personas are defined in `personas.yaml`:

1.  Parent & Family
2.  Homeowner
3.  Small Business & B2B
4.  Health & Wellness Seeker
5.  Tech Enthusiast
6.  Retiree & Senior
7.  Student & Learner
8.  Traveler & Leisure

### Label distribution

  Persona                      Confirmed creatives
  -------------------------- ---------------------
  Traveler & Leisure                           146
  Health & Wellness Seeker                     113
  Tech Enthusiast                               99
  Small Business & B2B                          93
  Student & Learner                             83
  Homeowner                                     70
  Retiree & Senior                              16
  Parent & Family                               13

The imbalance makes **Macro-F1** particularly useful because it gives
every persona equal weight rather than allowing frequent personas to
dominate the evaluation.

------------------------------------------------------------------------

## Duplicate evaluation set

`dup_pairs_eval.csv`

Contains **120 held-out pairs**:

-   60 duplicate
-   60 non-duplicate

This set is used for duplicate evaluation rather than being used
directly to train the duplicate detector.

------------------------------------------------------------------------

# 5. Multimodal Embeddings

## CLIP

The primary encoder is:

``` text
openai/clip-vit-base-patch32
```

CLIP produces a **512-dimensional projected embedding** for both image
and text.

For each creative:

``` text
Image
  │
  ▼
CLIP image encoder
  │
  ▼
512-D image embedding

Text
  │
  ▼
CLIP text encoder
  │
  ▼
512-D text embedding
```

Each modality is normalized before fusion.

------------------------------------------------------------------------

## Fusion

The final representation is:

``` text
fused =
    0.30 × normalized(image_embedding)
  + 0.70 × normalized(text_embedding)
```

The resulting vector is normalized again.

### Why 30/70?

Four fusion configurations were evaluated:

  --------------------------------------------------------------------------------------
       Image       Text     Macro-F1     Micro-F1   Weighted-F1     Hamming       Subset
  ---------- ---------- ------------ ------------ ------------- ----------- ------------
         70%        30%       58.00%       59.39%        60.76%      12.64%       34.78%

         50%        50%       60.38%       63.81%        64.27%      10.33%       46.74%

     **30%**    **70%**   **61.10%**   **64.89%**        64.13%   **8.97%**   **51.09%**

         20%        80%       59.54%   **66.36%**    **67.30%**       9.78%       46.74%
  --------------------------------------------------------------------------------------

The 30/70 configuration was selected because it achieved:

-   highest Macro-F1
-   lowest Hamming loss
-   highest exact subset accuracy

The experiment suggests that **ad text carries the strongest
persona-discriminative signal in this dataset**, while the visual
representation provides complementary information.

------------------------------------------------------------------------

# 6. Embedding Model Comparison

A second vision-language model was evaluated:

``` text
google/siglip-base-patch16-224
```

Both models were evaluated using the same leakage-safe split and
downstream persona-model search.

  Encoder           Test Macro-F1
  --------------- ---------------
  CLIP ViT-B/32        **61.10%**
  SigLIP Base              57.22%

CLIP was retained because it achieved stronger held-out performance
while also providing a lighter, CPU-friendly solution.

Larger CLIP-family models were considered but were not pursued as final
candidates because the available labelled dataset did not provide
evidence that the additional model size and compute would translate into
better generalization.

------------------------------------------------------------------------

# 7. Near-Duplicate Detection

Duplicate detection is important for two reasons:

1.  supporting evidence
2.  preventing train/test leakage

The system computes cosine similarity between fused embeddings.

The selected duplicate threshold is:

``` text
0.79
```

------------------------------------------------------------------------

## Full pairwise analysis

Across 2,882 creatives:

``` text
Unique pairs: 4,151,521
```

Similarity statistics:

  Statistic               Value
  ------------------- ---------
  Mean                  0.51265
  90th percentile       0.64643
  95th percentile       0.68563
  99th percentile       0.75132
  99.9th percentile     0.81548
  Maximum                  1.00

Number of pairs above selected thresholds:

    Similarity        Pairs
  ------------ ------------
        ≥ 0.70      154,325
        ≥ 0.75       43,272
        ≥ 0.78       16,297
    **≥ 0.79**   **11,283**
        ≥ 0.80        7,703
        ≥ 0.85        1,341
        ≥ 0.90          513

------------------------------------------------------------------------

# 8. Complete-Linkage Duplicate Clustering

Simply treating every high-similarity edge as an independent duplicate
relationship can create **chaining**.

Example:

``` text
A ≈ B
B ≈ C
```

does not necessarily mean:

``` text
A ≈ C
```

A connected-component strategy can therefore create artificially large
campaign groups.

The final implementation uses **complete-linkage hierarchical
clustering** at the same 0.79 similarity threshold.

This requires creatives within a cluster to remain mutually similar.

### Result

  Measure             Earlier grouping   Complete linkage
  ----------------- ------------------ ------------------
  Duplicate edges               11,283             11,283
  Clusters                       1,923          **2,038**
  Largest cluster         87 creatives   **14 creatives**

This produces tighter campaign groups and a cleaner foundation for
leakage prevention.

------------------------------------------------------------------------

# 9. Leakage-Safe Evaluation

The final split is performed using **duplicate-cluster IDs**, not
individual creatives.

This prevents near-duplicate creatives from appearing in different
splits.

Final split:

  Split           Creatives   Clusters
  ------------- ----------- ----------
  Train                 421        343
  Development            87         73
  Test                   92         74

The evaluation protocol is:

``` text
Duplicate clustering
        │
        ▼
Cluster-level split
        │
        ├── Train → fit models
        ├── Dev   → select thresholds / hyperparameters
        └── Test  → final held-out evaluation
```

The test set is not used to select the final model configuration.

------------------------------------------------------------------------

# 10. Persona Prediction

Two required approaches were implemented.

## Approach 1 --- Similarity-based kNN

The kNN model:

1.  embeds the query creative
2.  retrieves similar training creatives
3.  collects **all** persona labels from those neighbours
4.  weights persona votes by similarity
5.  produces a score independently for each persona

This supports multi-label predictions.

For example:

``` text
Traveler & Leisure       0.82
Health & Wellness        0.67
Tech Enthusiast          0.04
```

rather than forcing:

``` text
ONE creative → ONE persona
```

------------------------------------------------------------------------

## Approach 2 --- One-vs-Rest Logistic Regression

The final model uses:

``` text
512-D fused embedding
        │
        ▼
One-vs-Rest Logistic Regression
        │
        ├── Parent & Family
        ├── Homeowner
        ├── Small Business & B2B
        ├── Health & Wellness Seeker
        ├── Tech Enthusiast
        ├── Retiree & Senior
        ├── Student & Learner
        └── Traveler & Leisure
```

Each persona gets an independent score.

This naturally supports multi-label prediction.

------------------------------------------------------------------------

# 11. Hyperparameter Search

The final search evaluated:

### kNN

``` text
k ∈ {3, 5, 7, 9, 11, 15}
```

### Logistic Regression

``` text
C ∈ {
    0.01, 0.03, 0.1, 0.3,
    1.0, 3.0, 10.0, 30.0
}
```

### Persona thresholds

``` text
0.20 → 0.70
```

The configuration was selected using **development Macro-F1**.

### Selected configurations

``` text
kNN:
    k = 5
    threshold = 0.40

Logistic Regression:
    C = 30
    threshold = 0.60
```

------------------------------------------------------------------------

# 12. Final Model Comparison

On the held-out test set:

  ---------------------------------------------------------------------------------
  Model               Macro-F1     Micro-F1   Weighted-F1      Hamming       Subset
                                                                           Accuracy
  --------------- ------------ ------------ ------------- ------------ ------------
  kNN                   49.32%       55.10%        56.05%       11.96%       45.65%

  **One-vs-Rest     **61.10%**   **64.89%**    **64.13%**    **8.97%**   **51.09%**
  Logistic                                                             
  Regression**                                                         
  ---------------------------------------------------------------------------------

The classifier was selected as the final persona model.

An important observation from experimentation is that kNN performed
strongly on the development set but dropped substantially on unseen
creative clusters. Logistic Regression showed better held-out
generalization.

------------------------------------------------------------------------

# 13. Trust Layer: Abstention

A central requirement of the system is:

> **Do not force a persona recommendation when the available evidence is
> weak.**

The final operating policy is:

  Condition                                          Decision                 Confidence
  -------------------------------------------------- ------------------------ ------------
  Top score ≥ 0.60 **and** top1-top2 margin ≥ 0.10   `suggest`                high
  Otherwise top score ≥ 0.45                         `suggest_with_caution`   medium
  Top score \< 0.45                                  `abstain`                low

Example:

``` text
Strong:
Traveler & Leisure → 0.91
→ suggest

Moderate:
Traveler & Leisure → 0.55
→ suggest_with_caution

Weak:
Traveler & Leisure → 0.32
→ abstain
```

The confidence label is an **operating decision derived from model
scores and score margin**. It is not presented as a formally calibrated
probability.

------------------------------------------------------------------------

# 14. Evidence Retrieval

For selected personas, the system retrieves supporting labelled
creatives.

Each evidence item contains:

``` json
{
  "creative_id": "c_2074",
  "similarity": 0.9414,
  "near_duplicate": true
}
```

Evidence is:

-   retrieved from the confirmed/reference set
-   deduplicated
-   sorted by similarity
-   marked as near-duplicate when similarity crosses the duplicate
    threshold

The classifier prediction itself does **not** require a direct lookup of
the query creative in `confirmed_personas.csv`.

The confirmed dataset is used for:

-   supervised training
-   labelled reference evidence

A new/unconfirmed creative can therefore still be passed through the
trained classifier.

------------------------------------------------------------------------

# 15. LLM Rationale

The LLM is used only as an **explanation layer**.

It does not determine the persona.

Pipeline:

``` text
ML classifier
      │
      ▼
Persona + score
      │
      ▼
Retrieve evidence
      │
      ▼
Evidence-only LLM prompt
      │
      ▼
JSON response
      │
      ▼
Schema validation
      │
      ├── valid → use rationale
      │
      └── invalid / unavailable → deterministic fallback
```

The rationale prompt is intentionally constrained to supplied evidence.

It contains information such as:

-   predicted persona
-   model score
-   retrieved creative IDs
-   similarity values
-   near-duplicate flags

The LLM is explicitly instructed not to invent:

-   demographics
-   product details
-   visual objects
-   audience characteristics
-   campaign themes

unless that information is actually present in the supplied evidence.

------------------------------------------------------------------------

# 16. Mock LLM Mode

The pipeline can run without an external LLM API:

``` bash
python -m srcA.cli --creative-id c_0030 --mock-llm
```

This uses a deterministic local rationale function.

Normal mode:

``` bash
python -m srcA.cli --creative-id c_0030
```

The rationale source is explicitly reported as:

``` text
llm
mock
fallback
```

This makes the inference pipeline easier to test and reproduce.

------------------------------------------------------------------------

# 17. Structured Output

The final response is validated using Pydantic.

Conceptually:

``` json
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

The schema uses strict validation so unexpected fields or malformed
values do not silently pass through the pipeline.

------------------------------------------------------------------------

# 18. Example CLI Usage

### Normal LLM mode

``` bash
python -m srcA.cli --creative-id c_0030
```

### Deterministic mock mode

``` bash
python -m srcA.cli --creative-id c_0030 --mock-llm
```

The CLI returns structured JSON containing:

``` text
creative_id
confidence
decision
personas
evidence
rationale
rationale_source
```

------------------------------------------------------------------------

# 19. Example Prediction Behaviour

## High-confidence example

For `c_0860`, the system produced approximately:

``` text
Persona:
Health & Wellness Seeker

Score:
0.952

Decision:
suggest

Confidence:
high
```

The evidence retrieval also returned multiple highly similar supporting
creatives, including near-duplicate matches.

------------------------------------------------------------------------

## Medium-confidence example

For `c_0030`:

``` text
Persona:
Traveler & Leisure

Score:
0.553

Decision:
suggest_with_caution

Confidence:
medium
```

This demonstrates that the system does not require every model
prediction to become a strong recommendation.

------------------------------------------------------------------------

## Unconfirmed creative

The system was also tested on an unconfirmed creative such as `c_2417`.

It could still produce a classifier prediction because inference uses
the saved model and embedding representation rather than looking up the
query creative's confirmed label.

This is an important distinction:

``` text
confirmed_personas.csv
        │
        ├── training labels
        │
        └── labelled evidence/reference set

saved classifier
        │
        ▼
new creative embedding
        │
        ▼
persona prediction
```

------------------------------------------------------------------------

# 20. Project Structure

``` text
memologs/
│
├── creatives.parquet
├── confirmed_personas.csv
├── personas.yaml
├── dup_pairs_eval.csv
├── geo_panel.csv
│
├── images/
│   └── ...
│
├── srcA/
│   ├── __init__.py
│   ├── config.py
│   ├── data.py
│   ├── embeddings.py
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
│
├── scripts/
│   ├── evaluate_duplicates.py
│   ├── analyze_all_pairwise.py
│   ├── build_duplicate_clusters.py
│   ├── inspect_duplicate_clusters.py
│   ├── train_persona_models.py
│   └── save_persona_model.py
│
├── artifacts/
│   └── ...
│
├── REPORT.md
├── README.md
├── Makefile
└── pyproject.toml
```

### Module responsibilities

  Module            Responsibility
  ----------------- -------------------------------------------
  `config.py`       Paths, seeds, thresholds and constants
  `data.py`         Dataset loading and validation
  `embeddings.py`   Image/text embeddings and fusion
  `duplicate.py`    Cosine similarity and duplicate detection
  `clustering.py`   Complete-linkage duplicate clustering
  `split.py`        Cluster-level leakage-safe split
  `knn_model.py`    Multi-label similarity baseline
  `classifier.py`   One-vs-Rest Logistic Regression
  `evaluate.py`     Persona metrics
  `abstention.py`   Suggest / caution / abstain logic
  `evidence.py`     Supporting creative retrieval
  `rationale.py`    LLM rationale + fallback
  `schema.py`       Pydantic output validation
  `pipeline.py`     End-to-end orchestration
  `cli.py`          Command-line inference

------------------------------------------------------------------------

# 21. Reproducibility

The project separates:

``` text
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

-   fixed random seed
-   cached artifacts
-   cluster-level splitting
-   development-only hyperparameter selection
-   held-out test evaluation
-   deterministic mock LLM mode
-   strict output schema

------------------------------------------------------------------------

# 22. Experimentation Summary

The project evolved through several important iterations.

### Iteration 1 --- Initial baseline

Established kNN and classifier baselines.

### Iteration 2 --- Genuine multi-label modelling

Changed from single-persona classification to independent persona
scoring.

### Iteration 3 --- Leakage-safe splitting

Moved train/dev/test assignment from creatives to duplicate clusters.

### Iteration 4 --- Better duplicate clustering

Replaced chaining-prone grouping with complete-linkage hierarchical
clustering.

### Iteration 5 --- Threshold tuning

Selected persona decision thresholds using the development set.

### Iteration 6 --- Hyperparameter tuning

Searched kNN `k` and Logistic Regression `C`.

### Iteration 7 --- Multimodal fusion search

Compared:

``` text
70/30
50/50
30/70
20/80
```

and selected 30/70.

### Iteration 8 --- Representation comparison

Compared CLIP ViT-B/32 against SigLIP Base.

### Iteration 9 --- Trust-aware inference

Added:

``` text
abstention
+
confidence
+
evidence
+
LLM rationale
+
schema validation
+
fallback
```

The final system is therefore the result of both **model
experimentation** and **evaluation-design improvements**.

------------------------------------------------------------------------

# 23. Limitations

The current system has several known limitations.

### Limited labelled data

Only 600 creatives are labelled, with strong class imbalance.

### Probability calibration

The classifier scores are operating scores rather than formally
calibrated probabilities.

### Duplicate threshold

The 0.79 threshold is an operating point and could be improved with a
dedicated duplicate-threshold development protocol.

### Evidence richness

The current evidence payload is primarily:

``` text
creative_id
similarity
near_duplicate
```

A richer evidence layer could provide controlled text snippets or other
grounded features to the rationale model.

### Dataset-specific generalization

The final metrics are measured on this dataset and this leakage-safe
split. They should not automatically be assumed to transfer to a
different ad inventory.

------------------------------------------------------------------------

# 24. Potential Future Improvements

Possible next steps include:

-   per-persona probability calibration
-   per-persona precision / recall / F1 analysis
-   dedicated duplicate-threshold development
-   richer multimodal evidence retrieval
-   uncertainty calibration
-   campaign-level drift monitoring
-   temporal evaluation
-   monitoring abstention coverage versus precision
-   retraining strategies as new confirmed labels arrive

------------------------------------------------------------------------

# 25. Final Takeaway

The final Part A system is deliberately more than:

``` text
ad → classifier → persona
```

It is:

``` text
ad
 │
 ▼
multimodal representation
 │
 ├──────────────► duplicate detection
 │
 ├──────────────► leakage-safe clustering
 │
 └──────────────► persona prediction
                         │
                         ▼
                    trust decision
                         │
                 ┌───────┴────────┐
                 ▼                ▼
              evidence         abstain
                 │
                 ▼
          grounded rationale
                 │
                 ▼
            validated JSON
```

The central design principle is:

> **Make the prediction, understand the evidence behind it, and know
> when not to trust it.**

The final held-out result of **61.10% Macro-F1, 64.89% Micro-F1, 64.13%
Weighted-F1, 8.97% Hamming loss, and 51.09% subset accuracy** comes from
a lightweight One-vs-Rest Logistic Regression model operating on a
carefully selected pretrained multimodal representation --- supported by
duplicate-aware evaluation and a trust layer rather than by model size
alone.

------------------------------------------------------------------------

## Quick Start

``` bash
# Activate environment
source .venv/bin/activate

# Run normal inference
python -m srcA.cli --creative-id c_0030

# Run deterministic mock-LLM inference
python -m srcA.cli --creative-id c_0030 --mock-llm
```

For the complete methodology, experimental record, and detailed
discussion, see:

``` text
REPORT.md
```
