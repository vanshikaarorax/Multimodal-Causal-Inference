# Part B — Geo-Panel Causal Lift Report

## 1. Objective & approach

**Goal:** estimate the incremental revenue caused by the new creative in the 6 geos with `treated_group=1` from **2026-06-04 to 2026-07-03**, while explicitly grading how much the result can be trusted. The panel contains **10,800 observations: 60 geos × 180 days**; treatment is non-randomized.

I used the supplied `partB.ipynb` as the analytical source of truth, first reproducing and inspecting its data/methodology/results, then translating the validated notebook logic into a modular CPU/offline Python pipeline. The notebook was used to visualize treated/control behavior, inspect pre-treatment fit and trends, evaluate placebo behavior, examine spend changes, and validate estimator/sensitivity results before implementation.

**Primary estimator — normalized Synthetic Control.** The 6 treated geos are aggregated into a daily treated series. The 54 controls form the donor pool. Each donor is normalized by its pre-period mean, then non-negative weights summing to 1 are optimized to minimize pre-period reconstruction error. The weighted counterfactual is rescaled to the treated level; post-period treated minus synthetic revenue is summed as incremental lift.

I chose this as primary because assignment was not randomized and the pre-period diagnostics do not support a clean parallel-trends assumption for a conventional DiD interpretation. **DiD is retained as a secondary diagnostic, not the headline estimate.**

## 2. Evidence from notebook exploration & diagnostics

| Diagnostic | Result | Interpretation |
|---|---:|---|
| Treated / control geos | 6 / 54 | Sufficient donor pool |
| Pre-period revenue correlation | **0.9303** | Strong shared movement |
| Synthetic pre-fit correlation | **0.9583** | Strong reconstruction |
| Synthetic pre-fit RMSE | **₹1,593/day** | Low absolute error |
| Relative pre-fit RMSE | **2.04%** | Strong pre-period fit |
| Weekly pretrend coefficient | **+₹379/week** | Treated trend differs from controls |
| Weekly pretrend 90% CI | **₹194–₹564/week** | Does not contain 0 |
| Weekly pretrend p-value | **0.000757** | Parallel trends not supported |
| Treated spend change | **+10.57%** | Spend increased post-treatment |
| Control spend change | **+10.19%** | Similar increase |
| Spend DiD | **₹482** | Small differential relative to overall spend change |

The notebook therefore supports a strong synthetic pre-fit but also exposes an important limitation: **treated and control trends were not fully parallel before treatment**. I do not hide this; it is why the synthetic-control construction is the primary estimate and DiD is treated cautiously.

## 3. Primary result

**Estimated incremental revenue: ₹266,422.99 over the 30-day post period.**

- Mean daily incremental revenue: **₹8,880.77**
- Mean lift versus the synthetic counterfactual: **10.47%**
- Estimate relative to aggregate treated pre-period revenue: **2.28%**
- Placebo-based 90% interval: **₹237,778.37–₹294,463.41**
- Empirical placebo p-value: **0.00990**

### Placebo inference

I ran **100 seeded placebo assignments (`seed=42`)**, each randomly selecting 6 control geos and applying the identical synthetic-control procedure. The placebo effects had:

`mean = -₹562.93`, `SD = ₹18,535.82`, `min = -₹37,955.08`, `median = -₹520.59`, `max = ₹53,629.56`.

**0/100** placebo effects were at least as extreme as the observed effect. Using the corrected empirical calculation `(0+1)/(100+1)`, the p-value is **0.00990**. The placebo 5th/95th percentiles were **-₹28,040.43 / +₹28,644.62**, producing the reported interval.

## 4. Robustness, alternative estimator & “other things going on”

**Donor sensitivity:** leave-one-active-donor-out estimates remained positive, ranging from approximately **₹256,472 to ₹275,651**, versus the baseline ₹266,423. The range is only about **7.2% of the baseline**, so direction is not driven by one donor.

**DiD secondary check:** two-way fixed-effects DiD estimated **₹11,847.65/day per treated geo**, or **₹2.133M** across 6 geos × 30 days, with a 90% interval of approximately **₹811K–₹3.454M**. Because the pretrend test rejects parallel trends, I do **not** use this much larger number as the primary result.

**Spend / concurrent changes:** treated spend rose **10.57%**, while controls rose **10.19%**; spend DiD was only **₹482**. This does not prove spend was irrelevant, but it gives no large differential-spend shock explaining the result. The pretrend difference remains the main causal limitation.

## 5. Trust State

The Trust State is deterministic and considers: positive point estimate, interval excluding zero, placebo p-value, synthetic pre-fit quality, leave-one-donor-out stability, and estimator disagreement when DiD's parallel-trends condition is valid.

**Final Trust State: `trusted`**

This is the implemented rule output, not a claim that all causal assumptions are perfect. In particular, the significant pretrend diagnostic should be disclosed when using the result for decisions.

## 6. Engineering / implementation

The notebook logic was converted into modular code:

```text
srcB/
├── config.py       # dates, columns, seed, artifact paths
├── data.py         # load + schema/panel validation
├── design.py       # pre/post + treated/control construction
├── diagnostics.py  # correlations, pretrend, spend, fit metrics
├── estimators.py   # synthetic control, DiD, donor sensitivity
├── placebo.py      # seeded placebo procedure
├── evaluate.py     # interval + empirical p-value
├── trust.py        # deterministic Trust State rules
├── pipeline.py     # end-to-end orchestration
└── cli.py           # command-line interface
scripts/run_partB.py
tests/
```

The implementation is **CPU/offline and seeded**, with machine-readable artifacts under `artifacts/partB/` for diagnostics, estimates, placebo results and donor weights/sensitivity.

### Reproduction / test run

Primary execution:

```bash
python scripts/run_partB.py
```

Output:

```text
Part B Results
Point estimate: ₹266,422.99
90% interval: ₹237,778.37 to ₹294,463.41
Relative effect: 2.28%
Placebo p-value: 0.0099
Trust State: trusted
```

The Part B pipeline was also invoked directly and returned the same values. The CLI run completed successfully, and the generated artifacts were verified for the expected result files and placebo/sensitivity outputs.

## 7. What I would do with two more weeks

1. **Improve causal identification:** investigate matching/augmented synthetic control or Bayesian structural time-series approaches that explicitly model the observed pretrend difference.
2. **Event-study / dynamic effects:** estimate daily treatment effects and inspect whether the lift appears immediately, grows, decays, or anticipates treatment.
3. **Negative-control / additional placebo designs:** add time-shifted and in-space falsification tests.
4. **Spend-adjusted modeling:** incorporate spend as a carefully justified covariate and test whether the lift survives alternative specifications.
5. **Uncertainty robustness:** compare placebo intervals with bootstrap/block-resampling and alternative donor-pool restrictions.
6. **Automated report generation:** have the reproducible pipeline generate the result tables/figures directly from raw data so the narrative cannot drift from computed outputs.

## 8. Time & AI-tool disclosure

**Time:** approximately **6–8 focused hours** were allocated to the take-home, including notebook investigation, experimentation, implementation, testing and documentation.

**AI tools:** ChatGPT was used as an engineering/research assistant for code scaffolding, debugging, experiment review, documentation and cross-checking. The implementation was executed and validated locally; the final methodology, assumptions, outputs and limitations were reviewed against the supplied notebook and runnable pipeline. No external API or online service is required for Part B execution.

---

### Bottom line

The analysis estimates **₹266.4K incremental revenue** with a **₹237.8K–₹294.5K placebo-based 90% interval** and **p=0.0099**. The synthetic counterfactual fits the pre-period well (**2.04% relative RMSE**) and donor sensitivity is stable, while the significant pretrend test is the key limitation. The implemented deterministic Trust State is **`trusted`**, with that limitation explicitly retained in the report rather than hidden.
