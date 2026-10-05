# Part B — Causal Lift Estimate

## 1. Objective & approach

Estimate incremental revenue from the creative launch for the 6 treated geos over **2026-06-04 to 2026-07-03** using Synthetic Control, placebo inference, DiD as a secondary check, and robustness diagnostics.

Because the panel contains no currency metadata, all monetary quantities are reported as **revenue units**.

## 2. Evidence from notebook exploration & diagnostics

The pre-treatment synthetic-control fit is strong (**relative RMSE 2.04%, correlation 0.958**), but the treated and control groups do **not** satisfy the parallel-trends diagnostic (**p ≈ 0.0008**).

The gap is present throughout the pre-period rather than appearing only at launch. All 6 treated geos show positive pre-trends; G41 is the strongest, but leave-one-treated-out testing shows the violation persists after removing any one geo. Spend also has a significant treated-group pretrend (**p = 0.00015**).

Therefore, the positive effect is robust in direction, but causal magnitude is less certain.

## 3. Primary result

- **Incremental revenue:** **1,598,537.92 revenue units**
- **90% interval:** **1,569,893.30–1,626,578.35**
- **Mean daily incremental revenue:** **53,284.60**
- **Mean lift vs. synthetic counterfactual:** **10.47%**
- **Placebo p-value:** **0.0099**
- **Relative effect:** **2.28%** using aggregate treated pre-period revenue as the normalization

The original estimate was on the wrong scale because treated revenue was averaged across the 6 treated geos; the corrected implementation uses a daily **sum**.

## 4. Robustness, alternative estimator & “other things going on”

Synthetic-control window sensitivity remains positive:

| Pre-period | Estimate |
|---|---:|
| 30 days | 1.744M |
| 60 days | 1.770M |
| 90 days | 1.642M |
| Primary | **1.599M** |

DiD is approximately **2.133M revenue units**, measuring the same 6 geos × 30 post days. It is therefore about **33% higher** than Synthetic Control and represents genuine estimator disagreement, not a scale mismatch.

The 10.47% and 2.28% figures use different denominators: 10.47% is mean daily lift against the synthetic counterfactual; 2.28% normalizes the aggregate estimate by aggregate treated pre-period revenue.

## 5. Trust State

**Final Trust State: `directionally_trusted`**

The positive estimate, positive interval, placebo result, donor sensitivity, and strong pre-fit support the direction. However, failed parallel trends and estimator disagreement prevent a stronger `trusted` classification.

The revised rule ensures failed diagnostics can only lower trust.

## 6. Engineering / implementation

Key implementation changes:

- Corrected treated aggregation from `.mean()` to `.sum()` for the headline total.
- Added pre-trend, treated-geo, leave-one-out, spend, and SC-window diagnostics.
- Corrected Trust State logic.
- Removed unsupported currency assumptions.
- Full test suite: **26 passed**.

## 7. What I would do with two more weeks

1. Investigate the source of the treated-group pretrend and potential confounders.
2. Add stronger covariates and/or matching to improve counterfactual construction.
3. Pre-register an estimator/diagnostic decision rule and validate on additional holdout periods or launches.

## 8. Time & AI-tool disclosure

The analysis was developed within the assignment timebox. AI tools were used for coding assistance, debugging, interpretation checks, and documentation. Final methodology, diagnostics, code changes, and conclusions were reviewed against the supplied data and requirements.

## Bottom line

The corrected analysis estimates **1.599M incremental revenue units** with a **90% interval of 1.570M–1.627M** and **p = 0.0099**. The effect remains positive across robustness checks, but the failed pre-treatment trend and estimator disagreement mean the result should be interpreted as **directionally supported rather than a fully identified causal magnitude**.

# Evaluator Fixes — Final Validation

## B1. The incremental revenue total is on the wrong scale [Must fix]

Changed treated daily aggregation from `.mean()` to `.sum()`. The corrected total is **1,598,537.92 revenue units**, with a **1,569,893.30–1,626,578.35** 90% interval.

## B2. DiD and synthetic control are compared on different scales [Must fix]

Both estimators now refer to the same 6 treated geos × 30 post-treatment days. Synthetic Control is **1.599M** and DiD is **~2.133M revenue units**. The ~33% difference is genuine estimator disagreement.

**10.47%** = mean daily lift versus the synthetic counterfactual.  
**2.28%** = aggregate estimate relative to aggregate treated pre-period revenue.

## B3. Your failed pre-trend test was not investigated [Must fix]

The gap is persistent across the pre-period; all 6 treated geos contribute; G41 is strongest but not solely responsible. Spend also shows a significant treated pretrend. SC estimates remain positive across 30/60/90-day fitting windows. The investigation supports the **direction** but reduces confidence in the **causal magnitude**.

## B4. The Trust State rule rewards a failed check [Must fix]

Trust logic was rewritten so failed diagnostics cannot improve the grade. With positive direction but failed parallel trends and estimator disagreement, the final state is **`directionally_trusted`**.

## B5. Currency [Nice to fix]

No currency field exists in the panel. The report therefore uses **revenue units** and **spend units**, without assuming ₹ or another currency.
