# Part B — Geo-Panel Causal Lift Analysis

Estimate the incremental revenue associated with a new creative across **6 treated geographies** using Synthetic Control, placebo inference, DiD as a secondary check, robustness diagnostics, and a deterministic Trust State.

> **Data:** 60 geos × 180 days = 10,800 observations  
> **Treatment period:** 2026-06-04 to 2026-07-03  
> **Treated / control:** 6 / 54  
> **Execution:** CPU/offline, seeded, reproducible

## Results at a glance

| Metric | Result |
|---|---:|
| **Incremental revenue** | **1,598,537.92 revenue units** |
| 90% placebo interval | **1,569,893.30 – 1,626,578.35** |
| Mean daily incremental revenue | **53,284.60 units** |
| Mean lift vs. synthetic counterfactual | **10.47%** |
| Relative effect | **2.28%** |
| Empirical placebo p-value | **0.0099** |
| Trust State | **`directionally_trusted`** |
| Synthetic pre-fit relative RMSE | **2.04%** |

> The supplied panel has no currency field, so results are reported in native **revenue/spend units**, not ₹ or USD.

## Method

### Primary estimator — Synthetic Control

The 6 treated geos are aggregated into a daily treated series. The 54 control geos form the donor pool.

Donors are normalized using their pre-treatment means, and non-negative weights summing to 1 are optimized to reconstruct the treated pre-period. The synthetic counterfactual is then rescaled to the treated level.

Post-treatment incremental revenue is the observed treated revenue minus the synthetic counterfactual, summed across the **6 treated geos × 30 post-treatment days**.

Synthetic Control is the primary estimator because treatment was non-randomized and the pre-treatment diagnostics do not support a clean parallel-trends assumption for conventional DiD.

### Secondary estimator — Difference-in-Differences

DiD is retained as a secondary diagnostic.

- DiD estimate: **~2,132,576.73 revenue units**
- Same 6 treated geos × 30 post-treatment days
- Approximately **33% higher** than Synthetic Control

This is treated as genuine estimator disagreement, not a scale mismatch.

## Key diagnostics

### Pre-treatment fit

- Revenue correlation: **0.9303**
- Synthetic pre-fit correlation: **0.9583**
- Relative RMSE: **2.04%**
- RMSE: **1,593 units/day**

### Parallel-trends limitation

The treated and control groups show different pre-treatment trends:

- Weekly pretrend: **+379.24 units/week**
- 90% CI: **+194.03 to +564.46**
- p-value: **0.000757**

Additional investigation found:

- The gap is present throughout the pre-period, not only around launch.
- **All 6 treated geos** show positive pre-treatment slopes.
- G41 is the strongest contributor, but the violation persists after removing any one treated geo.
- Treated spend also has a significant pre-treatment trend (**p = 0.00015**).
- Synthetic-control estimates remain positive across alternative fitting windows.

**Conclusion:** the positive direction is robust, but causal magnitude is less certain.

## Primary result

**Estimated incremental revenue: 1,598,537.92 revenue units**

- Mean daily incremental revenue: **53,284.60**
- Mean lift vs. synthetic counterfactual: **10.47%**
- Relative effect: **2.28%**
- 90% placebo interval: **1,569,893.30–1,626,578.35**
- Empirical placebo p-value: **0.0099**

The corrected estimate represents the total across all 6 treated geographies. The earlier **266,422.99** estimate resulted from averaging across treated geos instead of aggregating them.

## Robustness

### Synthetic-control window sensitivity

| Pre-treatment window | Incremental estimate | Mean lift |
|---|---:|---:|
| 30 days | 1,743,543.05 | 11.55% |
| 60 days | 1,770,173.17 | 11.75% |
| 90 days | 1,641,790.45 | 10.80% |
| Primary | **1,598,537.92** | **10.47%** |

All specifications remain positive.

### Donor sensitivity

Leave-one-active-donor-out estimates remain positive:

**~1,538,833 – 1,653,904 revenue units**

### Placebo inference

100 seeded placebo assignments were run using `seed=42`.

- **0/100** placebo effects were at least as extreme as the observed effect.
- Corrected empirical p-value: **0.00990**

## 10.47% vs. 2.28%

These percentages use different denominators:

- **10.47%** = mean daily lift relative to the synthetic counterfactual.
- **2.28%** = aggregate incremental estimate relative to aggregate treated pre-period revenue.

The 10.47% figure is the direct measure of average lift against the constructed counterfactual.

## Trust State

Final:

```text
directionally_trusted
```

The positive estimate, positive interval, placebo result, donor sensitivity, and strong pre-fit support the direction.

However:

- Parallel trends fail.
- DiD and Synthetic Control differ materially.

The Trust State logic was revised so failed diagnostics **can only lower trust**.

## Project structure

```text
srcB/
├── config.py        # configuration
├── data.py          # data + schema validation
├── design.py        # pre/post + treated/control construction
├── diagnostics.py   # diagnostics and fit metrics
├── estimators.py    # Synthetic Control, DiD, sensitivity
├── placebo.py       # placebo procedure
├── evaluate.py      # interval + p-value
├── trust.py         # Trust State rules
├── pipeline.py      # end-to-end pipeline
└── cli.py            # CLI

scripts/
└── run_partB.py

tests/
```

Generated artifacts are stored under:

```text
artifacts/partB/
```

## Run

```bash
python scripts/run_partB.py
pytest -q
```

Expected test result:

```text
26 passed
```

Expected CLI output:

```text
Part B Results
Point estimate: 1,598,537.92 revenue units
90% interval: 1,569,893.30 to 1,626,578.35 revenue units
Relative effect: 2.28%
Placebo p-value: 0.0099
Trust State: directionally_trusted
```

## Evaluator fixes

### B1 — Incremental revenue scale

Changed treated daily aggregation from `.mean()` to `.sum()` so the headline estimate represents total incremental revenue across all 6 treated geos.

### B2 — DiD vs. Synthetic Control scale

Verified both estimators use the same 6-geo × 30-day aggregate quantity. The remaining ~33% difference is genuine estimator disagreement.

### B3 — Failed pre-trend investigation

Added diagnostics for:

- Pre-period gap over time
- Individual treated-geo trends
- Leave-one-treated-out tests
- Pretrend window sensitivity
- Spend pretrend
- Synthetic-control window sensitivity

### B4 — Trust State

Reworked the Trust State rules so failed diagnostics cannot improve the grade.

### B5 — Currency

Removed unsupported currency assumptions and report native revenue/spend units.

## Reproducibility

The analysis is:

- **CPU/offline**
- **Seeded**
- **Modular**
- **Machine-readable**
- **Tested**

The supplied `partB.ipynb` was used as the analytical source of truth before translating the validated logic into the modular pipeline.

## Next steps

With additional time:

1. Improve causal identification with augmented Synthetic Control or related approaches.
2. Add an event-study/dynamic treatment-effect analysis.
3. Add negative-control and time-shifted placebo tests.
4. Incorporate spend as a carefully justified covariate.
5. Compare alternative uncertainty procedures such as block resampling.

## Bottom line

The analysis estimates **1.599M revenue units of incremental revenue** over the 30-day post-treatment period, with a **1.570M–1.627M** placebo-based interval and **p = 0.0099**.

The effect remains positive across robustness checks, but the persistent pre-treatment trend difference and estimator disagreement mean the result should be interpreted as:

> **Directionally supported, but not a fully identified causal magnitude.**
