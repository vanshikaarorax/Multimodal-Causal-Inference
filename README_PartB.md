# Part B — Geo-Panel Causal Lift Analysis

## Overview

Part B estimates the incremental revenue associated with a new creative introduced across a small set of treated geographies.

The analysis is designed around a geo-level panel containing **60 geographies over 180 days (10,800 geo-day observations)**. Six geographies receive the new creative beginning **2026-06-04**, while the remaining 54 geographies act as potential controls.

The implementation focuses on a reproducible, CPU-friendly causal workflow rather than a simple before/after comparison.

### Core question

> What incremental revenue is attributable to the new creative across the treated geographies during the post-treatment period?

The pipeline produces:

- A point estimate of incremental revenue
- A 90% uncertainty interval
- Placebo-based inference
- Pre-treatment fit diagnostics
- Donor sensitivity analysis
- Difference-in-differences as a secondary estimator
- Spend diagnostics
- A deterministic **Trust State**

---

## Dataset

The expected input file is:

```text
geo_panel.csv
```

The panel contains:

| Field | Description |
|---|---|
| `date` | Observation date |
| `geo` | Geography identifier |
| `revenue` | Daily revenue |
| `spend` | Daily spend |
| `treated_group` | Treatment assignment indicator |

The panel validation expects:

- **60 unique geographies**
- **180 unique dates**
- **10,800 total rows**
- **6 treated geographies**
- **54 control geographies**
- No duplicate `geo` + `date` observations
- No missing revenue/spend values
- No negative revenue/spend values

### Treatment window

```text
Pre-treatment:   2026-01-05 → 2026-06-03
Treatment starts: 2026-06-04
Post-treatment:  2026-06-04 → 2026-07-03
```

The six treated geographies are identified directly from `treated_group`; no geography needs to be supplied manually at runtime.

---

# Methodology

## 1. Panel construction

The raw panel is loaded and validated before analysis.

The pipeline creates:

```text
post         = 1 if date >= 2026-06-04
treated      = treated_group
treated_post = treated × post
```

This creates the treatment indicator required for the secondary DiD model while keeping the synthetic-control analysis based on the underlying geo-level time series.

---

## 2. Pre-treatment diagnostics

Before estimating lift, the pipeline checks whether treated and control groups exhibit similar historical movement.

Two daily pre-treatment correlations are calculated:

1. Raw revenue correlation
2. Baseline-indexed revenue correlation

The implementation also performs a weekly pre-treatment trend test using geo fixed effects and clustered inference.

### Observed diagnostic

The weekly pre-treatment trend test produced:

```text
Coefficient:  +379.24 revenue units / week
90% CI:       [194.03, 564.46]
p-value:      0.000757
```

This indicates that the treated and control groups did **not** exhibit statistically indistinguishable pre-treatment trends under this diagnostic.

This matters because a conventional DiD interpretation relies heavily on parallel pre-treatment trends. Therefore, the synthetic-control estimate is treated as the primary estimate, while DiD is retained as a secondary comparison rather than the main causal estimate.

---

# 3. Synthetic Control

The primary estimator is a normalized synthetic-control construction.

The six treated geographies are aggregated into a treated daily revenue series.

The 54 control geographies form the donor pool.

### Weight construction

Because the geographies have substantially different revenue scales, each donor series is normalized by its own pre-treatment mean.

The optimization finds non-negative donor weights satisfying:

```text
weight_i >= 0
Σ weight_i = 1
```

The objective minimizes the pre-treatment mean squared error between:

```text
normalized treated revenue
```

and

```text
weighted normalized donor revenue
```

The resulting weighted donor series is then rescaled to the treated group's revenue level.

This produces:

```text
Synthetic pre-treatment revenue
Synthetic post-treatment revenue
```

The post-treatment incremental revenue is:

```text
Incremental revenue_t
    = Treated revenue_t
    - Synthetic revenue_t
```

and total incremental revenue is the sum across the 30-day post-treatment window.

---

# 4. Synthetic-control pre-fit quality

The fitted synthetic control closely tracks the treated group during the pre-treatment period.

Observed metrics:

| Metric | Value |
|---|---:|
| Pre-period RMSE | 1,593.13 / day |
| Pre-period MAE | 1,269.07 / day |
| Correlation | 0.9583 |
| Relative RMSE | 2.04% |
| Treated pre-period mean | ₹78,062.52 |
| Synthetic pre-period mean | ₹78,062.52 |

The **2.04% relative RMSE** indicates a strong pre-treatment reconstruction of the treated trajectory.

---

# 5. Incremental Revenue Estimate

The primary synthetic-control estimate is:

```text
Point estimate: ₹266,422.99
```

Equivalent average daily incremental revenue:

```text
₹8,880.77 / day
```

The mean relative lift against the synthetic counterfactual is:

```text
10.47%
```

The evaluation layer expresses the overall effect relative to treated pre-period revenue as:

```text
2.28%
```

These percentages answer different questions:

- **10.47%** — average post-period lift relative to the synthetic counterfactual.
- **2.28%** — total estimated incremental revenue relative to aggregate treated pre-period revenue.

---

# 6. Placebo Inference

A placebo procedure is used to assess whether an effect of the observed magnitude could plausibly arise from applying the same synthetic-control procedure to untreated geographies.

For each placebo iteration:

1. Randomly select six control geographies.
2. Treat them as a placebo-treated group.
3. Use the remaining control geographies as donors.
4. Fit the same normalized synthetic-control procedure.
5. Calculate the placebo post-period effect.

The analysis uses:

```text
100 placebo assignments
Random seed: 42
```

### Placebo distribution

Observed placebo-effect summary:

| Statistic | Value |
|---|---:|
| Mean | -562.93 |
| Std. dev. | 18,535.82 |
| Minimum | -37,955.08 |
| 25th percentile | -14,118.66 |
| Median | -520.59 |
| 75th percentile | 13,306.21 |
| Maximum | 53,629.56 |

None of the 100 placebo effects exceeded the observed effect in absolute magnitude.

Using the corrected empirical placebo p-value:

```text
p = (exceeding + 1) / (N + 1)
  = (0 + 1) / (100 + 1)
  = 0.00990
```

---

# 7. Placebo-Based 90% Interval

The uncertainty interval is constructed from the placebo-effect distribution.

The placebo 5th and 95th percentiles are:

```text
q05 = -₹28,040.43
q95 =  ₹28,644.62
```

The resulting 90% interval around the observed estimate is:

```text
₹237,778.37  →  ₹294,463.41
```

Therefore the primary estimate is positive and the placebo-based interval does not cross zero.

---

# 8. Difference-in-Differences — Secondary Check

A two-way fixed-effects DiD model is also fitted:

```text
revenue ~ treated_post + geo fixed effects + date fixed effects
```

Inference is clustered by geography.

The estimated daily treatment effect per treated geography is:

```text
₹11,847.65
```

with a 90% interval of approximately:

```text
₹4,507.53 → ₹19,187.77
```

Aggregated across:

```text
6 treated geographies × 30 post-treatment days
```

the DiD estimate is:

```text
₹2,132,576.73
```

with a 90% interval of approximately:

```text
₹811,354.84 → ₹3,453,798.63
```

However, the pre-treatment trend diagnostic does not support the parallel-trends assumption. Consequently, DiD is treated as a **secondary estimator/check**, not the primary result.

---

# 9. Donor Sensitivity Analysis

Synthetic-control estimates can depend on the donor pool.

To assess this, the analysis performs leave-one-donor-out sensitivity analysis over the active synthetic-control donors.

The baseline estimate is:

```text
₹266,422.99
```

Observed leave-one-donor-out effects ranged approximately from:

```text
₹256,472.18
```

to:

```text
₹275,650.59
```

The mean across the sensitivity runs was approximately:

```text
₹265,669.26
```

The sensitivity range is approximately:

```text
₹19,178
```

or roughly **7.2% of the baseline estimate**.

All leave-one-donor-out estimates remained positive.

This provides evidence that the direction of the estimated effect is not being driven by a single active donor geography.

---

# 10. Spend Diagnostic

Revenue lift should be interpreted alongside changes in spend.

The observed spend averages were:

| Group | Pre | Post |
|---|---:|---:|
| Treated | ₹9,342.76 | ₹10,330.60 |
| Control | ₹4,958.93 | ₹5,464.42 |

Relative spend changes:

```text
Treated: +10.57%
Control: +10.19%
```

Spend difference-in-differences:

```text
₹482.34
```

The treated and control groups therefore experienced similar relative spend increases during the period.

This diagnostic does not by itself establish that spend had no effect on revenue, but it provides an important check for a concurrent differential spend shock.

---

# 11. Trust State

The pipeline converts the main evidence into a deterministic Trust State.

The rules consider:

- Direction of the estimated effect
- Whether the placebo-based interval excludes zero
- Placebo p-value
- Synthetic-control pre-fit quality
- Leave-one-donor-out stability
- Agreement with the secondary DiD estimator when its identifying assumptions are valid

For the observed analysis:

```text
Trust State: trusted
```

This state should be read together with the underlying diagnostics, particularly the pre-treatment trend test. The label is a deterministic output of the implemented rules; it is not a substitute for inspecting the individual diagnostics.

---

# Results Summary

| Quantity | Result |
|---|---:|
| Treated geographies | 6 |
| Control geographies | 54 |
| Post-treatment days | 30 |
| Primary estimator | Normalized Synthetic Control |
| Incremental revenue | **₹266,422.99** |
| Average daily incremental revenue | **₹8,880.77** |
| Mean counterfactual-relative lift | **10.47%** |
| Placebo-based 90% interval | **₹237,778.37 – ₹294,463.41** |
| Placebo p-value | **0.00990** |
| Pre-fit relative RMSE | **2.04%** |
| Active synthetic donors | **21** |
| Leave-one-donor-out range | **₹256,472 – ₹275,651** |
| DiD total effect | ₹2,132,576.73 |
| Weekly pretrend p-value | 0.000757 |
| Trust State | **trusted** |

---

# Project Structure

```text
memologs_aiml_takehome/
│
├── geo_panel.csv
│
├── srcB/
│   ├── __init__.py
│   ├── config.py
│   ├── data.py
│   ├── design.py
│   ├── diagnostics.py
│   ├── estimators.py
│   ├── placebo.py
│   ├── evaluate.py
│   ├── trust.py
│   ├── pipeline.py
│   └── cli.py
│
├── scripts/
│   └── run_partB.py
│
├── tests/
│   ├── test_partB_design.py
│   └── test_trust.py
│
├── notebooks/
│   └── partB.ipynb
│
└── artifacts/
    └── partB/
        ├── diagnostics/
        ├── estimates/
        ├── placebo/
        └── figures/
```

---

# Running Part B

## One-command execution

From the project root:

```bash
python scripts/run_partB.py
```

No runtime creative ID, geography ID, or treatment ID is required.

The treatment assignment and treatment date are already defined by the panel/configuration.

Expected terminal output:

```text
Part B Results
Point estimate: ₹266,422.99
90% interval: ₹237,778.37 to ₹294,463.41
Relative effect: 2.28%
Placebo p-value: 0.0099
Trust State: trusted
```

---

# Optional CLI Arguments

The CLI also supports optional reproducibility/configuration arguments:

```bash
python scripts/run_partB.py --panel geo_panel.csv --placebos 100 --seed 42
```

Available arguments:

```text
--panel       Path to geo_panel.csv
--placebos    Number of placebo assignments
--seed        Random seed
```

The default configuration is already suitable for reproducing the reported results.

---

# Generated Artifacts

Running the pipeline writes analysis outputs under:

```text
artifacts/partB/
```

### Diagnostics

```text
artifacts/partB/diagnostics/diagnostics.json
```

Contains:

- Pre-treatment revenue correlations
- Weekly pretrend test
- Synthetic-control pre-fit metrics
- Spend diagnostics
- Parallel-trends diagnostic flag

### Estimates

```text
artifacts/partB/estimates/estimates.json
```

Contains the main estimate, interval, placebo p-value, secondary DiD estimate, and Trust State.

```text
artifacts/partB/estimates/synthetic_weights.csv
```

Contains the active synthetic-control donor weights.

```text
artifacts/partB/estimates/leave_one_donor_out.csv
```

Contains the donor sensitivity results.

### Placebo Results

```text
artifacts/partB/placebo/placebo_results.csv
```

Contains the 100 placebo assignments and their estimated effects.

---

# Reproducibility

The implementation is deterministic under the configured random seed:

```text
Seed = 42
```

The placebo assignments use the seeded random generator, while the synthetic-control optimization uses deterministic constrained optimization.

The complete analysis can therefore be rerun locally without network access.

---

# Design Principles

### Primary estimate over naive before/after

The analysis does not interpret the raw treated-group revenue increase as causal lift. Instead, it constructs a counterfactual from untreated geographies.

### Pre-treatment fit before post-treatment inference

The synthetic control is evaluated on historical data before interpreting the post-treatment gap.

### Placebos instead of relying on a single parametric interval

The same estimator is applied to untreated geographies to characterize the empirical distribution of effects under placebo assignment.

### Sensitivity to donor composition

The active donor set is perturbed through leave-one-donor-out analysis to test whether the estimated direction depends heavily on a single geography.

### Secondary estimator as a diagnostic

DiD provides a useful comparison, but the pre-treatment trend diagnostic is explicitly checked before relying on its identifying assumption.

### Reproducibility

The entire workflow is executable from the command line and produces machine-readable artifacts for inspection.

---

# Final Output

The primary analysis estimates approximately:

> **₹266.4K of incremental revenue over the 30-day post-treatment period**, with a placebo-based **90% interval of approximately ₹237.8K–₹294.5K** and an empirical placebo p-value of **0.0099**.

The implemented deterministic Trust State is:

> **trusted**

The full diagnostics and sensitivity results should be reviewed alongside this summary rather than relying on the headline estimate alone.
