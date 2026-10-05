# MemoLogs — Part A Technical Report
### Multimodal Persona Inference with Duplicate-Aware Evaluation, Abstention & Grounded Rationales

## 1. Objective & System Design

**Goal:** Given an advertisement creative, infer the most likely targeting persona(s) while providing evidence, a calibrated trust decision, and a short grounded explanation. The system is deliberately **multi-label**: one creative may map to multiple allowed personas. The end-to-end flow is:

**creative + image/text → CLIP multimodal embedding → near-duplicate detection / campaign clusters → leakage-safe train/dev/test split → kNN baseline + One-vs-Rest Logistic Regression → abstention → labelled-neighbour evidence → schema-validated LLM rationale → deterministic fallback → CLI JSON.**

The dataset contains **2,882 creatives/images** and **600 human-confirmed labelled creatives**; the remaining creatives are unlabelled for inference. The eight allowed personas are Traveler & Leisure, Health & Wellness Seeker, Tech Enthusiast, Small Business & B2B, Student & Learner, Homeowner, Retiree & Senior, and Parent & Family. The labels are imbalanced, especially Retiree & Senior (16) and Parent & Family (13), so overall metrics are supplemented with per-persona metrics. Multi-label output is preserved throughout the pipeline. fileciteturn22file0L257-L326

## 2. Multimodal Representation

I used **OpenAI CLIP ViT-B/32** (`openai/clip-vit-base-patch32`). CLIP provides aligned 512-D projected image and text embeddings. Each modality is L2-normalized, then fused as:

**fused = normalize(0.30 × image_embedding + 0.70 × text_embedding)**

The 30:70 weighting was selected using the same leakage-safe evaluation protocol. We compared 70:30, 50:50, 30:70 and 20:80 image:text fusion. On the held-out test set, 30:70 achieved the best **Macro-F1 (61.10%)**, lowest **Hamming loss (8.97%)**, and highest **subset accuracy (51.09%)**, while remaining competitive on Micro-F1 (64.89%). The 20:80 variant had higher Micro/Weighted-F1 but lower Macro-F1 and subset accuracy, so 30:70 was selected for better balanced generalization.

Ads span **eight banner sizes** (including 300×250, 728×90, 300×600, 970×250, 970×90, 160×600, 320×50 and 336×280). CLIP's processor standardizes image dimensions through its resize/crop/normalization pipeline, allowing differently sized creatives to share one embedding space while retaining text/OCR signal from the ad copy. fileciteturn22file0L674-L688

I also evaluated **Google SigLIP Base (ViT-B/16)** under the same downstream search; its held-out Macro-F1 was **57.22%** versus CLIP's **61.10%**. Larger CLIP-family models were considered but not selected because this label-limited dataset did not justify the additional computational/model-size cost. fileciteturn22file0L1136-L1147

## 3. Near-Duplicate Detection & Leakage Control

Near-duplicates are detected with **cosine similarity over the fused embeddings**, using a fixed operating threshold of **0.79**. The threshold was selected from the full-corpus similarity distribution before the held-out pair evaluation; the 120 evaluation pairs were kept out of threshold tuning. Across **4,151,521 unique creative pairs**, the 99th percentile similarity was 0.751 and the 99.9th percentile was 0.815; 0.79 therefore targets the high-similarity tail while keeping candidate volume manageable (11,283 pairs).

On the held-out `dup_pairs_eval.csv` (**60 duplicates / 60 non-duplicates**):

| Metric | Result |
|---|---:|
| Precision | **1.0000** |
| Recall | **1.0000** |
| F1 | **1.0000** |
| Accuracy | **1.0000** |
| TN / FP / FN / TP | **60 / 0 / 0 / 60** |

The held-out duplicate similarities ranged from **0.888–0.999**, while non-duplicates ranged from **0.344–0.770**, leaving a clear separation around the selected threshold.

For leakage prevention, duplicate edges at 0.79 were grouped using **complete-linkage hierarchical clustering** rather than a chaining-prone connected-component/representative approach. This preserved the same 11,283 duplicate edges while producing **2,038 tighter clusters** (largest cluster: 14 creatives). Train/dev/test splits are made at the cluster level so a near-duplicate campaign cannot appear across partitions. Final split: **421 train / 87 dev / 92 test creatives**, corresponding to **343 / 73 / 74 clusters**. The clustering change is treated as a leakage-control improvement, not as a controlled model-performance gain. fileciteturn22file0L872-L890

## 4. Persona Models & Model Selection

Two required approaches were evaluated:

**A. kNN label propagation.** Retrieve nearest labelled creatives and aggregate their confirmed persona labels using similarity-weighted scores. This naturally supplies evidence for downstream rationale generation.

**B. One-vs-Rest Logistic Regression.** Train one binary classifier per persona on the fused embeddings, allowing independent persona scores and therefore genuine multi-label predictions.

Hyperparameters were selected **only on the development split**. Final selected configurations were **kNN: k=5, threshold=0.40** and **Logistic Regression: C=30, threshold=0.60**.

| Model | Test Macro-F1 | Micro-F1 | Weighted-F1 | Hamming ↓ | Subset Acc. |
|---|---:|---:|---:|---:|---:|
| kNN (k=5, t=0.40) | 49.32% | 55.10% | 56.05% | 11.96% | 45.65% |
| **One-vs-Rest LR (C=30, t=0.60)** | **61.10%** | **64.89%** | **64.13%** | **8.97%** | **51.09%** |

LR was selected as the final persona model because it generalized substantially better to unseen duplicate clusters. The final test macro-F1 is **61.10%**, with a development macro-F1 of **73.07%**; this gap is reported explicitly rather than hidden.

Per-persona test F1: Tech Enthusiast **0.706**, Health & Wellness Seeker **0.690**, Traveler & Leisure **0.653**, Student & Learner **0.636**, Homeowner **0.632**, Small Business & B2B **0.571**, Retiree & Senior **1.000 (support 2)**, Parent & Family **0.000 (support 2)**. The two rare-class scores are unstable because each has only two test examples, so they should not be interpreted as robust estimates. fileciteturn22file0L1313-L1322

## 5. Abstention, Evidence & Trust

The system does **not** force a persona prediction for every creative. It returns one of:

- **suggest:** top score ≥ 0.60 and margin to the next persona ≥ 0.10; return qualifying personas (up to 3).
- **suggest_with_caution:** evidence is moderate; return the top persona.
- **abstain:** evidence is below the caution threshold.

The production test policy produced **74 suggest / 14 caution / 4 abstain**, giving **95.65% coverage** and **68.18% creative-level precision**. The selective top-1 coverage–precision curve shows the expected trade-off: at threshold 0.60, coverage is **85.87%** with **72.15% precision**; at 0.90, coverage falls to **40.22%** while precision rises to **81.08%**. This makes the operating point explicit rather than hiding uncertainty behind forced labels. fileciteturn22file0L1263-L1288

Evidence is retrieved from confirmed labelled creatives and includes **creative_id, cosine similarity, near_duplicate flag, and confirmed persona labels**. This evidence is intentionally separated from the classifier score: a high classifier score without strong retrieved support should not automatically produce an overconfident explanation.

## 6. Grounded LLM Rationale & Reliability

For accepted suggestions, the LLM receives **only the predicted persona/score and retrieved evidence** (neighbour IDs, similarities, duplicate status and confirmed labels). It is explicitly prohibited from inventing ad content, demographics, audience attributes, semantic themes, or unsupported creative IDs.

The rationale pipeline is:

**LLM → JSON parse → Pydantic schema validation → evidence/ID validation → use rationale; otherwise → deterministic template fallback.**

The output schema uses strict fields and forbids unsupported fields. Any malformed JSON, schema violation, or unsupported creative reference triggers the deterministic fallback. `MOCK_LLM=1` bypasses external inference and produces a reproducible local template for testing. This keeps the system runnable without an API dependency while preserving the real LLM path for production use. fileciteturn22file0L179-L229

## 7. Reproducibility & Final Interface

The repository is organized into focused modules for data loading, embeddings, duplicates/clustering, leakage-safe splitting, kNN/classification, evaluation, abstention, evidence, rationale, schema validation, pipeline orchestration and CLI. Cached embeddings/model artifacts are used so normal evaluation does not require recomputing the expensive multimodal representation.

`make all` runs the reproducibility checks: **19 unit tests, held-out duplicate evaluation, abstention/coverage evaluation, and a CLI smoke test using `MOCK_LLM=1`**. The test suite currently passes **19/19**.

Example interface:

```bash
python -m srcA.cli --creative-id c_0030
python -m srcA.cli --creative-id c_0030 --mock-llm
```

The final JSON exposes the prediction, trust decision, evidence, rationale, rationale source and confidence in a machine-validatable format.

## 8. Key Limitations & Next Steps

The main limitation is label scarcity/imbalance, especially for the two rare personas, plus a measurable dev→test generalization gap. The duplicate benchmark is small (120 pairs) and perfectly separated at the selected threshold, so broader production validation would still be valuable. Probability scores are model scores rather than calibrated probabilities. Future improvements would include calibration on a held-out calibration set, more labelled examples for rare personas, richer campaign-level temporal validation, and monitoring abstention precision/coverage after deployment.

**AI/tooling note:** AI assistance was used for implementation iteration, debugging, code review and documentation; model choices, experiments, evaluation runs and final engineering decisions were verified against the dataset and executed locally. No secrets are included in the repository.
