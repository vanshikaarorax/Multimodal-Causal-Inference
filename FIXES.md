# MemoLogs — Take-home Fixes

## Summary

This document records the fixes made after the take-home review. The priority was to address the Must-fix items first, then the highest-value Should-fix items, without rewriting unrelated parts of the project.

Part A fixes focused on abstention, duplicate-threshold selection, label-quality inspection, CLI dependency isolation, and reproducibility. Part B fixes corrected the revenue scale, aligned estimator comparisons, investigated the failed pre-trend diagnostic, and made the Trust State monotonic with respect to failed diagnostics.

---

# Part A: Persona Suggester

## A1. Abstention on weak/no-persona-signal creatives — Must fix

**Problem.** The original abstention analysis used only labelled test creatives, so every evaluation example had a known persona. It did not demonstrate behaviour on the 2,282 unlabelled creatives that represent the real inference pool.

**Diagnosis.** The classifier scores are OvR model scores rather than calibrated probabilities, so a high score should not automatically be interpreted as strong evidence. The unlabelled pool was therefore audited separately rather than assuming the labelled-test coverage curve represented production behaviour.

**Changes.**
- Added `srcA/notebooks/abstention_unlabelled_audit.ipynb`.
- Audited all 2,282 unlabelled creatives and inspected weak-evidence cases.
- Changed the operating abstention policy through development-set selection.
- Final operating policy: top score ≥ 0.70 → `suggest`; top score ≥ 0.60 → `suggest_with_caution`; otherwise → `abstain`.
- Added explicit coverage/precision reporting.

**After.** On the held-out test set, the selected policy produced 56 `suggest`, 8 `suggest_with_caution`, and 36 `abstain` decisions: **64% coverage** with **81.25% precision among suggested predictions**.

The unlabelled audit is now part of the documented evaluation rather than treating the labelled test set as the only trust analysis.

---

## A2. Duplicate threshold — Must fix

**Problem.** The original 0.79 threshold was not supported by labelled development data. `duplicate_dev_candidates.csv` had an empty `is_duplicate` field, while 11,283 pairs exceeded 0.79.

**Diagnosis.** A manual development review was performed on high-similarity candidates. Among 34 inspected candidates, 31 were non-duplicates and 3 were judged near-duplicates under the creative/campaign-variant definition. Reviewed hard negatives included similarities around 0.856 and 0.869, while a reviewed genuine near-duplicate was 0.893.

**Changes.**
- Added/documented the development review in `srcA/notebooks/duplicate_threshold_analysis.ipynb`.
- Moved the operating threshold from **0.79 → 0.88**.
- The threshold was chosen conservatively from development review, not claimed to be mathematically optimal.
- The 120-pair held-out evaluation was run after the threshold decision.

**After.**
- Full pairwise unique pairs: **4,151,521**.
- Duplicate edges at 0.88: **1,655**, down from 11,283 at 0.79.
- On `dup_pairs_eval.csv` (60 duplicate + 60 non-duplicate), threshold 0.88 achieved:
  - Precision: **1.00**
  - Recall: **1.00**
  - F1: **1.00**
  - Accuracy: **1.00**
  - TN=60, FP=0, FN=0, TP=60.

The 120-pair evaluation is useful but not sufficiently hard to establish that 0.88 is globally optimal; additional labelled development pairs and broader manual review would be preferable.

**Clustering impact.**
- Complete-linkage clusters: **2,191**.
- Multi-creative clusters: **452**.
- Largest cluster: **12 creatives**.
- Train/dev/test split after clustering: **414 / 86 / 100 creatives**, with **343 / 74 / 74 clusters**.

The lower edge count makes the leakage groups substantially tighter than the original 0.79 operating point.

---

## A3. Label quality and rare personas — Should fix

**Problem.** The confirmed labels could contain noise, but the original pipeline did not actively flag likely label disagreements. Rare personas also had very small test support.

**Changes.**
- Added `srcA/notebooks/Label_quality&personas.ipynb`.
- For each confirmed creative, retrieved labelled neighbours and computed similarity-weighted persona support.
- Flagged potential mismatches where neighbour evidence strongly disagreed with confirmed labels.
- These flags are treated as **manual-review triage signals**, not automatic relabelling.

**Rare-persona interpretation.** The test set contains very few examples for some personas, so per-persona F1 is highly uncertain. In particular, Retiree & Senior has only 1 test example and Parent & Family has 2 in the latest split. Their individual scores should therefore not be treated as reliable population-level estimates.

A stronger future evaluation would use cluster-level cross-validation and confidence intervals/bootstrapping, especially for rare personas.

---

## A4. Cached CLI should not require torch — Should fix

**Problem.** `srcA/embeddings.py` imported heavy `torch`/`transformers` dependencies at module import time, even though cached-embedding CLI inference does not need them.

**Change.**
- Moved the heavy imports to the embedding-generation path in `srcA/embeddings.py`.
- Cached-embedding inference can therefore import the CLI without loading the transformer stack.

**After.** The deterministic CLI path works with:
```bash
MOCK_LLM=1 python -m srcA.cli --creative-id c_0412 --mock-llm
```

---

## A5. Reproducibility from raw data — Should fix

**Problem.** The original `make all` relied too heavily on cached artifacts and did not clearly regenerate the complete pipeline from source data.

**Changes.**
- Added the embedding-generation stage to the reproducibility flow.
- Added duplicate clustering, persona training/evaluation, model saving, duplicate evaluation, abstention evaluation, tests, CLI smoke testing, and Part B execution to the Makefile pipeline.
- Added a `save` target for the final persona model.
- Pinned the main Python dependencies in `requirements.txt`.
- Kept deterministic cached artifacts available so repeated runs remain practical.

The intended reproducibility flow is now:

```text
raw data
  ↓
embeddings
  ↓
duplicate detection / clustering
  ↓
cluster-level train/dev/test split
  ↓
persona search
  ↓
final saved model
  ↓
abstention / duplicate evaluation
  ↓
tests + CLI smoke test
```

---

# Part B: Lift Estimate

## B1. Incremental revenue scale — Must fix

**Problem.** The treated series originally used a mean across the six treated geos, so the headline represented an average treated-geo effect rather than incremental revenue across all six treated geos.

**Change.** Corrected the estimator in `srcB/estimators.py` so the treated quantity is on the required aggregate six-geo scale.

**Before → After**
- Incremental revenue: **₹266,422.99 → ₹1,598,537.92**
- 90% interval: **₹237,778.37–₹294,463.41 → ₹1,569,893.30–₹1,626,578.35**
- Relative effect: **2.28% → 2.28%**
- Placebo p-value: **0.0099 → 0.0099**

The sixfold increase is expected because the corrected quantity sums the six treated geos instead of reporting their mean.

**Final estimate:** incremental revenue across all six treated geos from **June 4–July 3, 2026 = ₹1,598,537.92**, with a 90% placebo-based interval of **₹1,569,893.30–₹1,626,578.35**.

---

## B2. DiD vs synthetic control scale — Must fix

**Problem.** The original comparison treated estimates from different scales as directly comparable.

**Change.** Reworked the comparison so the estimators are interpreted on the same treated-geo/day scope and units before comparing magnitude.

The corrected headline is the six-geo aggregate estimate of **₹1.599M**. The reported **2.28% relative effect** is a normalized effect, so it does not need to equal the percentage-style “mean lift” wording used previously. The documentation now distinguishes aggregate incremental revenue from relative lift instead of presenting them as competing quantities.

**Result.** The corrected analysis no longer rejects one estimator simply because an earlier version of the two numbers was on a different scale.

---

## B3. Failed pre-trend diagnostic — Must fix

**Problem.** The original analysis reported a failed parallel-trends test (p ≈ 0.0008) but did not investigate whether the violation was localized, geo-specific, or caused by another pre-treatment factor.

**Changes.** Added:
- weekly treated-vs-control pre-trend gap analysis;
- per-treated-geo trend regressions;
- leave-one-treated-out sensitivity;
- 30/60/90-day pre-treatment window sensitivity;
- a separate spend pre-trend check.

**Findings.**
- The 90-day trend test gives slope **+364.6 revenue units/week**, p=**0.0044**.
- The violation is not driven by one treated geo.
- All six treated geos show positive pre-treatment trends; G41 has the largest trend.
- Leave-one-treated-out tests remain significant regardless of which treated geo is removed.
- A separate spend pre-trend is also positive and significant, p=**0.00015**, suggesting broader changes affecting treated geos rather than a uniquely identifiable creative effect.
- The violation is stronger over the longer pre-period: the 60-day and 30-day tests are not statistically significant (p=0.142 and p=0.339).

**Sensitivity of the synthetic-control estimate.**

| Pre-treatment window | Estimate | Change vs primary |
|---|---:|---:|
| 30 days | ₹1.744M | +9.1% |
| 60 days | ₹1.770M | +10.7% |
| 90 days | ₹1.642M | +2.7% |
| Primary | ₹1.599M | — |

All estimates remain positive, and the alternative windows stay within roughly 11% of the primary estimate. Therefore the pre-trend issue changes the **strength of causal interpretation**, but it does not cause the estimated effect to disappear or reverse.

---

## B4. Trust State must penalize failed diagnostics — Must fix

**Problem.** The old Trust State logic only counted estimator disagreement when the parallel-trends diagnostic passed. A failed diagnostic could therefore remove a check and make the result appear more trusted.

**Change.** Rewrote `srcB/trust.py` so a failed diagnostic can never improve the Trust State.

The decision hierarchy is now:

```text
Direction supported?
    No  → not_trusted
    Yes
     ↓
Magnitude unstable?
    Yes → magnitude_uncertain
    No
     ↓
Pre-fit good?
Parallel trends valid?
No estimator disagreement?
     ↓
ALL true → trusted
ANY failure → directionally_trusted
```

For the current result:
- positive estimate: yes;
- placebo evidence: p=0.0099;
- magnitude stable under sensitivity: yes;
- pre-fit fit acceptable: yes;
- parallel-trends diagnostic: **failed**.

**Final Trust State: `directionally_trusted`.**

This means the direction of the estimated effect is reasonably supported, but the causal/magnitude interpretation is not strong enough to call the result fully trusted because the parallel-trends assumption is violated.

---

## B5. Currency — Nice to fix

The panel does not explicitly provide a currency field. The revenue figures are therefore reported using the ₹ notation already used by the analysis, but the interpretation should be treated as an explicit currency assumption rather than a fact supplied by the panel schema.

---

# Documentation

## D1. Part A report — Must fix

The previous `REPORT_PartA.md` was effectively a README copy rather than a concise decision-focused report.

**Change.** The final submission includes a dedicated Part A report structure that leads with:
- final model and held-out results;
- duplicate-threshold decision and evaluation;
- abstention results;
- label-quality findings;
- reproducibility;
- what did not work;
- limitations;
- what would be done with two more weeks;
- AI-tool usage/time disclosure.

The README remains the detailed technical reference; the report is intended to be the short reviewer-facing summary.

---

## D2. References and disclosures — Nice to fix

**Changes.**
- Updated documentation references to match the actual repository layout.
- Added the Part A notebook references:
  - `srcA/notebooks/abstention_unlabelled_audit.ipynb`
  - `srcA/notebooks/duplicate_threshold_analysis.ipynb`
  - `srcA/notebooks/embeddings.ipynb`
  - `srcA/notebooks/Label_quality&personas.ipynb`
- Added the Part A AI-tool disclosure.
- Added the reproducibility/test references used by the final pipeline.
- `venv/` is excluded from the submission.

---

# Final Before / After Summary

| Area | Before | After |
|---|---|---|
| Part A duplicate threshold | 0.79 | **0.88** |
| Duplicate edges | 11,283 | **1,655** |
| Part A split | 421 / 87 / 92 | **414 / 86 / 100** |
| LR C | 30 | **3.0** |
| Persona threshold | 0.60 | **0.45** |
| Part A abstention | Labelled-test only | **Unlabelled audit + held-out trade-off** |
| Part B incremental revenue | ₹266,422.99 | **₹1,598,537.92** |
| Part B 90% interval | ₹237,778.37–₹294,463.41 | **₹1,569,893.30–₹1,626,578.35** |
| Part B relative effect | 2.28% | **2.28%** |
| Part B placebo p-value | 0.0099 | **0.0099** |
| Part B Trust State | trusted | **directionally_trusted** |

## What remains intentionally conservative

The duplicate threshold is not claimed to be globally optimal because the 120-pair evaluation set is small and separates the tested operating points too easily. The persona scores are not treated as calibrated probabilities, and rare-persona metrics are not treated as stable population estimates. Most importantly, the Part B positive lift is not described as fully causal/trusted because the parallel-trends diagnostic fails, even though the direction and magnitude are reasonably stable under the sensitivity checks.

## AI Tools / Time

AI coding assistants were used for implementation support, debugging, test interpretation, documentation cleanup, and review of experimental results. The final decisions, threshold selection, diagnostic interpretation, and validation of before/after numbers were checked against the repository outputs.

Approximate time spent on this fixes round: **3–4 hours**, within the requested time box.

