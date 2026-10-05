# MemoLogs — Part A Technical Report

### Multimodal Persona Inference with Duplicate-Aware Evaluation, Abstention & Grounded Rationales

## 1. Objective & System Design

**Goal:** Given an advertisement creative, infer the most likely targeting persona(s) while providing evidence, a trust decision, and a short grounded explanation. The system is deliberately **multi-label**: one creative may map to multiple allowed personas. The end-to-end flow is:

**creative + image/text → CLIP multimodal embedding → near-duplicate detection / campaign clusters → leakage-safe train/dev/test split → kNN baseline + One-vs-Rest Logistic Regression → abstention → labelled-neighbour evidence → schema-validated LLM rationale → deterministic fallback → CLI JSON.**

The dataset contains **2,882 creatives/images** and **600 human-confirmed labelled creatives**; the remaining **2,282 are unlabelled**. The eight allowed personas are Traveler & Leisure, Health & Wellness Seeker, Tech Enthusiast, Small Business & B2B, Student & Learner, Homeowner, Retiree & Senior, and Parent & Family. Labels are imbalanced, especially Retiree & Senior (16) and Parent & Family (13), so aggregate metrics are supplemented with per-persona metrics.

## 2. Multimodal Representation

I used **OpenAI CLIP ViT-B/32** (`openai/clip-vit-base-patch32`). CLIP provides aligned 512-D image and text embeddings. Each modality is L2-normalized, then fused as:

**fused = normalize(0.30 × image_embedding + 0.70 × text_embedding)**

The 30:70 weighting was selected using the same leakage-safe evaluation protocol. Compared with 70:30, 50:50 and 20:80, it gave the best held-out **Macro-F1 (61.10%)**, lowest **Hamming loss (8.97%)**, and highest **subset accuracy (51.09%)**, while remaining competitive on Micro-F1 (64.89%). The 20:80 variant had higher Micro/Weighted-F1 but lower Macro-F1 and subset accuracy.

Ads span eight banner sizes including 300×250, 728×90, 300×600, 970×250, 970×90, 160×600, 320×50 and 336×280. CLIP's processor standardizes image dimensions through resize/crop/normalization, allowing differently sized creatives to share one embedding space while retaining the text/OCR representation used by the pipeline.

I also evaluated **Google SigLIP Base (ViT-B/16)** under the same downstream search; its held-out Macro-F1 was **57.22%** versus CLIP's **61.10%**, so CLIP was retained.

## 3. Near-Duplicate Detection & Leakage Control

Near-duplicates are detected with **cosine similarity over fused embeddings**, with the final operating threshold changed from **0.79 to 0.88** after development/manual error analysis. The 120 held-out evaluation pairs were kept out of threshold selection.

Across **4,151,521 unique creative pairs**, p99 similarity was 0.7513 and p99.9 was 0.8155. Manual review showed hard negatives in the 0.80–0.87 region, while genuine campaign variants also appeared there; therefore 0.88 was chosen as a conservative operating point rather than claiming mathematical optimality.

On the held-out `dup_pairs_eval.csv` (**60 duplicates / 60 non-duplicates**), the final 0.88 threshold achieved:

| Metric | Result |
|---|---:|
| Precision | **1.0000** |
| Recall | **1.0000** |
| F1 | **1.0000** |
| Accuracy | **1.0000** |
| TN / FP / FN / TP | **60 / 0 / 0 / 60** |

The benchmark is clean but not difficult enough to distinguish nearby thresholds: duplicate similarities start around 0.888, while non-duplicates end around 0.770. The perfect score is therefore reported as benchmark performance, not proof that 0.88 is uniquely optimal.

At 0.88, complete-linkage clustering produced **1,655 duplicate edges, 2,191 clusters, 452 multi-creative clusters, 1,143 creatives in multi-creative clusters, and a largest cluster of 12**. Splits are performed at cluster level so near-duplicate campaigns cannot cross train/dev/test partitions.

## 4. Persona Models & Model Selection

Two required approaches were evaluated:

**A. kNN label propagation.** Retrieve nearest labelled creatives and aggregate confirmed persona labels using similarity-weighted scores.

**B. One-vs-Rest Logistic Regression.** Train one binary classifier per persona on the fused embeddings, allowing independent persona scores and genuine multi-label predictions.

Hyperparameters are selected **only on the development split**. In the latest reproducible run, the selected configurations were **kNN: k=5, threshold=0.20** and **Logistic Regression: C=3.0, threshold=0.45**.

| Model | Test Macro-F1 | Micro-F1 | Weighted-F1 | Hamming ↓ | Subset Acc. |
|---|---:|---:|---:|---:|---:|
| kNN (k=5, t=0.20) | 38.93% | 53.06% | 53.59% | 17.25% | 30.00% |
| **One-vs-Rest LR (C=3, t=0.45)** | **55.32%** | **63.52%** | **63.90%** | **10.63%** | **42.00%** |

LR is therefore the final persona model for the latest run. Per-persona metrics are reported with support; the rare personas remain unstable because the held-out test contains only **1 Retiree & Senior and 2 Parent & Family examples**.

## 5. Abstention, Evidence & Trust

The system does **not** force a persona prediction for every creative. It returns:

- **suggest:** strong evidence supports a persona;
- **suggest_with_caution:** evidence is moderate;
- **abstain:** evidence is insufficient.

The abstention evaluation was changed to **TRAIN → fit → DEV policy selection → locked TEST evaluation**, so the test set is not used to choose thresholds. The latest DEV-selected policy is:

- suggest threshold **0.70**
- caution threshold **0.60**
- minimum margin **0.00**

Latest locked TEST result: **56 suggest / 8 caution / 36 abstain**, giving **64.00% coverage and 81.25% creative-level precision**. This explicitly leaves uncertain creatives unresolved instead of forcing labels.

A full audit was also added for all **2,282 unlabelled creatives**. Under the audited operating policy it produced **243 suggest (10.65%), 1,199 caution (52.54%) and 840 abstain (36.81%)**. We specifically checked for suggestions with nearest-labelled similarity below 0.60 and found **zero**; therefore no artificial retrieval gate was added merely to reproduce the reviewer concern.

Evidence is retrieved from confirmed labelled creatives and includes **creative_id, cosine similarity, near_duplicate flag and confirmed persona labels**. Classifier scores are treated as model confidence signals, not calibrated probabilities.

## 6. Grounded LLM Rationale & Reliability

For accepted suggestions, the LLM receives **only the predicted persona/score and retrieved evidence** (neighbour IDs, similarities, duplicate status and confirmed labels). It is prohibited from inventing ad content, demographics, audience attributes, semantic themes or unsupported creative IDs.

The rationale pipeline is:

**LLM → JSON parse → Pydantic schema validation → evidence/ID validation → use rationale; otherwise → deterministic template fallback.**

`MOCK_LLM=1` bypasses external inference and provides a reproducible local path for testing.

## 7. Reproducibility & Final Interface

The repository is organized into focused modules for data loading, embeddings, duplicates/clustering, leakage-safe splitting, kNN/classification, evaluation, abstention, evidence, rationale, schema validation, orchestration and CLI. `srcA/embeddings.py` now uses **lazy imports** for torch/transformers, so the cached-embedding CLI path does not require loading the deep-learning stack.

The reproducibility path was expanded so `make all` rebuilds derived artifacts rather than relying on stale outputs:

**embeddings → duplicate clusters → persona training/evaluation → saved persona model → tests → duplicate evaluation → abstention evaluation → CLI smoke test → Part B**

Dependencies are pinned in `requirements.txt`. The latest complete validation run regenerated the embeddings, rebuilt duplicate clusters, retrained/evaluated persona models, ran **26/26 tests**, evaluated duplicates and abstention, ran the CLI smoke test and completed Part B.

The latest model-selection run selects **LR C=3.0 / threshold=0.45**. The saved-model script must use the same selected configuration rather than the historical C=30 setting so the persisted CLI model stays aligned with evaluation.

## 8. Key Limitations & Next Steps

The main limitations are label scarcity/imbalance, especially for the two rare personas; a measurable dev→test generalization gap; and the small 120-pair duplicate benchmark. The 0.88 duplicate threshold is a conservative operating choice supported by development/manual review, but broader human-labelled validation around the ambiguous similarity region would strengthen it. Persona scores are model scores rather than calibrated probabilities.

### AI Tools Usage

AI assistance was used for implementation iteration, debugging, code review, experiment planning, notebook/report drafting and reasoning through reviewer feedback. All model, experiments, visual inspections, evaluation runs and final engineering decisions were verified locally against the provided dataset. No secrets or API keys were committed.

### If We Had Two More Weeks

I would prioritize **confidence calibration and stronger evaluation**: create a dedicated calibration split for reliable persona probabilities, build a larger human-labelled duplicate set around the 0.75–0.92 similarity region, and run cluster-aware K-fold evaluation. In parallel, I would collect more confirmed examples for Retiree & Senior and Parent & Family, turn the label-quality notebook into an active human-review queue, and add production monitoring for abstention rate, confidence, retrieval similarity and human corrections.


## Reviewer-Requested Fixes & Notebooks

**1. Abstention on weak/no-persona-signal creatives — [A1]**  
Added `srcA/notebooks/abstention_unlabelled_audit.ipynb` to audit all 2,282 unlabelled creatives using classifier confidence, margin and nearest-labelled similarity.  
The abstention policy was moved to leakage-safe DEV selection and locked TEST evaluation; the final TEST result is **64.00% coverage / 81.25% precision**.

**2. Duplicate-threshold justification — [A2]**  
Added `srcA/notebooks/duplicate_threshold_analysis.ipynb` for threshold analysis, manual hard-negative/positive inspection, cluster-impact analysis and visual inspection of random 100 duplicates >0.88.  
The operating threshold was changed from **0.79 to 0.88**, with the held-out 120-pair benchmark reported separately: **1.00 precision / 1.00 recall / 1.00 F1**.

**3. Label-quality and rare-persona review — [A3]**  
Added `srcA/notebooks/Label_quality&personas.ipynb` to compare confirmed labels against similarity-weighted support from the top five labelled neighbours and surface candidates for human review.  
Rare-persona support is explicitly reported; the latest TEST has only **1 Retiree & Senior and 2 Parent & Family** examples, so these metrics are treated as unstable.

**4. CLI dependency issue — [A4]**  
Made `torch`/`transformers` imports lazy in `srcA/embeddings.py`, so cached-embedding CLI inference can run without loading the heavy model stack.

**5. Reproducibility from raw data — [A5]**  
Updated the build flow/Makefile to rebuild derived artifacts and added the saved-model step, with dependencies pinned and the full validation path rerun.

