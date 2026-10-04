from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.metrics import mean_absolute_error, mean_squared_error

from .config import (
    DATE_COL,
    GEO_COL,
    REVENUE_COL,
    SPEND_COL,
    TREATMENT_DATE,
)
from .design import split_pre_post


def pre_treatment_correlation(
    panel: pd.DataFrame,
    value_col: str = REVENUE_COL,
) -> float:
    """Correlation between treated and control daily mean trajectories."""
    pre, _ = split_pre_post(panel)

    daily = (
        pre.groupby([DATE_COL, "treated"])[value_col]
        .mean()
        .reset_index()
    )

    pivot = daily.pivot(
        index=DATE_COL,
        columns="treated",
        values=value_col,
    )

    return float(pivot[1].corr(pivot[0]))


def indexed_pre_treatment_correlation(
    panel: pd.DataFrame,
    value_col: str = REVENUE_COL,
) -> float:
    """Correlation after normalizing each group by its pre-period mean."""
    pre, _ = split_pre_post(panel)

    daily = (
        pre.groupby([DATE_COL, "treated"])[value_col]
        .mean()
        .reset_index()
    )

    baseline = daily.groupby("treated")[value_col].mean()

    daily["indexed"] = daily.apply(
        lambda row: row[value_col] / baseline[row["treated"]],
        axis=1,
    )

    pivot = daily.pivot(
        index=DATE_COL,
        columns="treated",
        values="indexed",
    )

    return float(pivot[1].corr(pivot[0]))


def spend_diagnostic(
    panel: pd.DataFrame,
) -> dict[str, float]:
    """Compare treated/control pre-post spend changes."""
    pre, post = split_pre_post(panel)

    treated_pre = pre.loc[
        pre["treated"] == 1, SPEND_COL
    ].mean()

    treated_post = post.loc[
        post["treated"] == 1, SPEND_COL
    ].mean()

    control_pre = pre.loc[
        pre["treated"] == 0, SPEND_COL
    ].mean()

    control_post = post.loc[
        post["treated"] == 0, SPEND_COL
    ].mean()

    treated_change = treated_post - treated_pre
    control_change = control_post - control_pre

    return {
        "treated_pre_spend": float(treated_pre),
        "treated_post_spend": float(treated_post),
        "control_pre_spend": float(control_pre),
        "control_post_spend": float(control_post),
        "treated_spend_change": float(treated_change),
        "control_spend_change": float(control_change),
        "treated_relative_change": float(
            treated_change / treated_pre
        ),
        "control_relative_change": float(
            control_change / control_pre
        ),
        "spend_did": float(
            treated_change - control_change
        ),
    }


def synthetic_pre_fit_metrics(
    actual: pd.Series,
    synthetic: pd.Series,
) -> dict[str, float]:
    """Calculate pre-treatment synthetic-control fit metrics."""
    aligned = pd.concat(
        [
            actual.rename("actual"),
            synthetic.rename("synthetic"),
        ],
        axis=1,
    ).dropna()

    rmse = np.sqrt(
        mean_squared_error(
            aligned["actual"],
            aligned["synthetic"],
        )
    )

    mae = mean_absolute_error(
        aligned["actual"],
        aligned["synthetic"],
    )

    correlation = aligned["actual"].corr(
        aligned["synthetic"]
    )

    relative_rmse = rmse / aligned["actual"].mean()

    return {
        "rmse": float(rmse),
        "mae": float(mae),
        "correlation": float(correlation),
        "relative_rmse": float(relative_rmse),
        "actual_mean": float(aligned["actual"].mean()),
        "synthetic_mean": float(aligned["synthetic"].mean()),
    }


def weekly_pretrend_test(
    panel: pd.DataFrame,
) -> dict[str, float]:
    """Test whether treated and control geos had different
    pre-treatment weekly revenue trends.
    """
    pre, _ = split_pre_post(panel)

    pre = pre.copy()
    pre["week"] = (
        pre[DATE_COL]
        .dt.to_period("W")
        .dt.start_time
    )

    weekly_geo = (
        pre.groupby(
            [GEO_COL, "treated", "week"],
            as_index=False,
        )[REVENUE_COL]
        .mean()
    )

    weekly_geo["week_index"] = (
        weekly_geo["week"] - weekly_geo["week"].min()
    ).dt.days / 7

    model = smf.ols(
        f"{REVENUE_COL} ~ treated:week_index + C({GEO_COL})",
        data=weekly_geo,
    ).fit(
        cov_type="cluster",
        cov_kwds={"groups": weekly_geo[GEO_COL]},
    )

    coefficient = model.params["treated:week_index"]
    confidence_interval = model.conf_int(alpha=0.10).loc[
        "treated:week_index"
    ]

    return {
        "coefficient_per_week": float(coefficient),
        "ci_lower_90": float(confidence_interval[0]),
        "ci_upper_90": float(confidence_interval[1]),
        "p_value": float(
            model.pvalues["treated:week_index"]
        ),
    }