from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.optimize import minimize

from .config import (
    DATE_COL,
    GEO_COL,
    REVENUE_COL,
)
from .design import split_pre_post


def _fit_normalized_weights(
    treated_pre: pd.Series,
    control_pre: pd.DataFrame,
) -> pd.Series:
    """Fit non-negative donor weights on normalized pre-period trajectories."""
    treated_scale = treated_pre.mean()

    control_scales = control_pre.mean(axis=0)

    X = control_pre.div(control_scales, axis=1).values
    y = (treated_pre / treated_scale).values

    def objective(weights: np.ndarray) -> float:
        return float(np.mean((y - X @ weights) ** 2))

    n_donors = len(control_pre.columns)
    initial_weights = np.ones(n_donors) / n_donors

    result = minimize(
        objective,
        initial_weights,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * n_donors,
        constraints={
            "type": "eq",
            "fun": lambda weights: np.sum(weights) - 1.0,
        },
        options={
            "maxiter": 10000,
            "ftol": 1e-9,
        },
    )

    if not result.success:
        raise RuntimeError(
            f"Synthetic-control optimization failed: {result.message}"
        )

    weights = pd.Series(
        result.x,
        index=control_pre.columns,
        name="weight",
    )

    return weights[weights > 1e-4].sort_values(ascending=False)


def fit_synthetic_control(
    panel: pd.DataFrame,
) -> dict:
    """Fit the scaled synthetic-control model using pre-treatment revenue."""
    pre, post = split_pre_post(panel)

    treated_pre = (
        pre[pre["treated"] == 1]
        .groupby(DATE_COL)[REVENUE_COL]
        .sum()
        .sort_index()
    )

    treated_post = (
        post[post["treated"] == 1]
        .groupby(DATE_COL)[REVENUE_COL]
        .sum()
        .sort_index()
    )

    control_pre = (
        pre[pre["treated"] == 0]
        .pivot(
            index=DATE_COL,
            columns=GEO_COL,
            values=REVENUE_COL,
        )
        .sort_index()
    )

    control_post = (
        post[post["treated"] == 0]
        .pivot(
            index=DATE_COL,
            columns=GEO_COL,
            values=REVENUE_COL,
        )
        .sort_index()
    )

    treated_scale = treated_pre.mean()
    control_scales = control_pre.mean(axis=0)

    weights = _fit_normalized_weights(
        treated_pre,
        control_pre,
    )

    control_pre_norm = control_pre.div(
        control_scales,
        axis=1,
    )

    control_post_norm = control_post.div(
        control_scales,
        axis=1,
    )

    synthetic_pre = (
        control_pre_norm[weights.index].values
        @ weights.values
    ) * treated_scale

    synthetic_post = (
        control_post_norm[weights.index].values
        @ weights.values
    ) * treated_scale

    synthetic_pre = pd.Series(
        synthetic_pre,
        index=treated_pre.index,
        name="synthetic_revenue",
    )

    synthetic_post = pd.Series(
        synthetic_post,
        index=treated_post.index,
        name="synthetic_revenue",
    )

    incremental_revenue = (
        treated_post - synthetic_post
    ).rename("incremental_revenue")

    return {
        "weights": weights,
        "treated_pre": treated_pre,
        "treated_post": treated_post,
        "synthetic_pre": synthetic_pre,
        "synthetic_post": synthetic_post,
        "incremental_revenue": incremental_revenue,
        "point_estimate": float(incremental_revenue.sum()),
        "mean_daily_incremental": float(
            incremental_revenue.mean()
        ),
        "mean_relative_lift": float(
            (
                incremental_revenue / synthetic_post
            ).mean()
        ),
    }


def fit_did(
    panel: pd.DataFrame,
) -> dict:
    """Estimate two-way fixed-effects DiD as a robustness diagnostic."""
    model = smf.ols(
        f"{REVENUE_COL} ~ treated_post + C({GEO_COL}) + C({DATE_COL})",
        data=panel,
    ).fit(
        cov_type="cluster",
        cov_kwds={"groups": panel[GEO_COL]},
    )

    coefficient = model.params["treated_post"]
    standard_error = model.bse["treated_post"]
    confidence_interval = model.conf_int(alpha=0.10).loc[
        "treated_post"
    ]

    treated_geos = panel.loc[
        panel["treated"] == 1,
        GEO_COL,
    ].nunique()

    post_days = panel.loc[
        panel["post"] == 1,
        DATE_COL,
    ].nunique()

    total_effect = coefficient * treated_geos * post_days
    total_lower = confidence_interval[0] * treated_geos * post_days
    total_upper = confidence_interval[1] * treated_geos * post_days

    return {
        "daily_effect": float(coefficient),
        "standard_error": float(standard_error),
        "daily_ci_lower_90": float(confidence_interval[0]),
        "daily_ci_upper_90": float(confidence_interval[1]),
        "total_effect": float(total_effect),
        "total_ci_lower_90": float(total_lower),
        "total_ci_upper_90": float(total_upper),
    }


def leave_one_donor_out(
    panel: pd.DataFrame,
    baseline_result: dict | None = None,
) -> pd.DataFrame:
    """Re-fit the synthetic control after removing each active donor."""
    pre, post = split_pre_post(panel)

    treated_pre = (
        pre[pre["treated"] == 1]
        .groupby(DATE_COL)[REVENUE_COL]
        .sum()
        .sort_index()
    )

    treated_post = (
        post[post["treated"] == 1]
        .groupby(DATE_COL)[REVENUE_COL]
        .sum()
        .sort_index()
    )

    control_pre = (
        pre[pre["treated"] == 0]
        .pivot(
            index=DATE_COL,
            columns=GEO_COL,
            values=REVENUE_COL,
        )
        .sort_index()
    )

    control_post = (
        post[post["treated"] == 0]
        .pivot(
            index=DATE_COL,
            columns=GEO_COL,
            values=REVENUE_COL,
        )
        .sort_index()
    )

    if baseline_result is None:
        baseline_result = fit_synthetic_control(panel)

    active_donors = baseline_result["weights"].index
    treated_scale = treated_pre.mean()

    results = []

    for removed_geo in active_donors:
        donor_geos = [
            geo for geo in control_pre.columns
            if geo != removed_geo
        ]

        donor_pre = control_pre[donor_geos]
        donor_post = control_post[donor_geos]

        donor_scales = donor_pre.mean(axis=0)

        weights = _fit_normalized_weights(
            treated_pre,
            donor_pre,
        )

        donor_post_norm = donor_post.div(
            donor_scales,
            axis=1,
        )

        synthetic_post = (
            donor_post_norm[weights.index].values
            @ weights.values
        ) * treated_scale

        effect = float(
            (treated_post.values - synthetic_post).sum()
        )

        donor_pre_norm = donor_pre.div(
            donor_scales,
            axis=1,
        )

        synthetic_pre = (
            donor_pre_norm[weights.index].values
            @ weights.values
        ) * treated_scale

        pre_rmse = float(
            np.sqrt(
                np.mean(
                    (
                        treated_pre.values
                        - synthetic_pre
                    ) ** 2
                )
            )
        )

        results.append(
            {
                "removed_geo": removed_geo,
                "effect": effect,
                "pre_rmse": pre_rmse,
            }
        )

    return pd.DataFrame(results)
def pretrend_window_synthetic_sensitivity(
    panel: pd.DataFrame,
    windows: tuple[int, ...] = (30, 60, 90),
) -> pd.DataFrame:
    """Re-fit synthetic control using recent pre-treatment windows."""

    pre, post = split_pre_post(panel)
    pre_end = pre[DATE_COL].max()

    results = []

    for days in windows:
        start = pre_end - pd.Timedelta(days=days - 1)

        recent_pre = pre[pre[DATE_COL] >= start]
        window_panel = pd.concat(
            [recent_pre, post],
            ignore_index=True,
        )

        result = fit_synthetic_control(window_panel)

        results.append(
            {
                "window_days": days,
                "point_estimate": result["point_estimate"],
                "mean_daily_incremental": result[
                    "mean_daily_incremental"
                ],
                "mean_relative_lift": result[
                    "mean_relative_lift"
                ],
            }
        )

    return pd.DataFrame(results)