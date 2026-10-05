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
def pretrend_gap_by_week(
    panel: pd.DataFrame,
) -> pd.DataFrame:
    """Measure the treated-vs-control revenue gap for each pre-treatment week."""

    pre, _ = split_pre_post(panel)

    pre = pre.copy()
    pre["week"] = (
        pre[DATE_COL]
        .dt.to_period("W")
        .dt.start_time
    )

    weekly = (
        pre.groupby(["week", "treated"])[REVENUE_COL]
        .mean()
        .reset_index()
    )

    pivot = weekly.pivot(
        index="week",
        columns="treated",
        values=REVENUE_COL,
    ).rename(
        columns={
            0: "control_mean_revenue",
            1: "treated_mean_revenue",
        }
    )

    pivot["gap"] = (
        pivot["treated_mean_revenue"]
        - pivot["control_mean_revenue"]
    )

    first_gap = pivot["gap"].iloc[0]

    pivot["gap_vs_first_week"] = (
        pivot["gap"] - first_gap
    )

    return pivot.reset_index()

def treated_geo_pretrend_test(
    panel: pd.DataFrame,
) -> pd.DataFrame:
    """Estimate the pre-treatment revenue trend for each treated geo."""

    pre, _ = split_pre_post(panel)

    treated = pre[pre["treated"] == 1].copy()

    treated["week"] = (
        treated[DATE_COL]
        .dt.to_period("W")
        .dt.start_time
    )

    weekly_geo = (
        treated.groupby(
            [GEO_COL, "week"],
            as_index=False,
        )[REVENUE_COL]
        .mean()
    )

    weekly_geo["week_index"] = (
        weekly_geo["week"]
        - weekly_geo["week"].min()
    ).dt.days / 7

    results = []

    for geo, group in weekly_geo.groupby(GEO_COL):

        model = smf.ols(
            f"{REVENUE_COL} ~ week_index",
            data=group,
        ).fit()

        ci = model.conf_int(alpha=0.10).loc["week_index"]

        results.append(
            {
                GEO_COL: geo,
                "coefficient_per_week": float(
                    model.params["week_index"]
                ),
                "ci_lower_90": float(ci[0]),
                "ci_upper_90": float(ci[1]),
                "p_value": float(
                    model.pvalues["week_index"]
                ),
                "r_squared": float(
                    model.rsquared
                ),
            }
        )

    return (
        pd.DataFrame(results)
        .sort_values("coefficient_per_week")
        .reset_index(drop=True)
    )
def leave_one_treated_out_pretrend(
    panel: pd.DataFrame,
) -> pd.DataFrame:
    """Re-run the aggregate pre-trend test after removing each treated geo."""

    treated_geos = (
        panel.loc[
            panel["treated"] == 1,
            GEO_COL,
        ]
        .drop_duplicates()
        .tolist()
    )

    results = []

    for geo in treated_geos:

        reduced = panel[panel[GEO_COL] != geo].copy()

        result = weekly_pretrend_test(reduced)

        results.append(
            {
                "excluded_geo": geo,
                **result,
            }
        )

    return (
        pd.DataFrame(results)
        .sort_values("p_value")
        .reset_index(drop=True)
    )

def pretrend_window_sensitivity(
    panel: pd.DataFrame,
    windows: tuple[int, ...] = (30, 60, 90),
) -> pd.DataFrame:
    """Test the treated-vs-control pre-trend over recent pre-period windows."""

    pre, _ = split_pre_post(panel)

    end_date = pre[DATE_COL].max()

    results = []

    for days in windows:

        start_date = end_date - pd.Timedelta(days=days - 1)

        window = pre[
            pre[DATE_COL] >= start_date
        ].copy()

        result = weekly_pretrend_test(window)

        results.append(
            {
                "window_days": days,
                **result,
            }
        )

    return pd.DataFrame(results)

def weekly_covariate_pretrend_test(
    panel: pd.DataFrame,
    value_col: str,
) -> dict[str, float]:
    """Test treated-vs-control pre-treatment weekly trends for a covariate."""

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
        )[value_col]
        .mean()
    )

    weekly_geo["week_index"] = (
        weekly_geo["week"]
        - weekly_geo["week"].min()
    ).dt.days / 7

    model = smf.ols(
        f"{value_col} ~ treated:week_index + C({GEO_COL})",
        data=weekly_geo,
    ).fit(
        cov_type="cluster",
        cov_kwds={
            "groups": weekly_geo[GEO_COL]
        },
    )

    coefficient = model.params[
        "treated:week_index"
    ]

    ci = model.conf_int(alpha=0.10).loc[
        "treated:week_index"
    ]

    return {
        "coefficient_per_week": float(coefficient),
        "ci_lower_90": float(ci[0]),
        "ci_upper_90": float(ci[1]),
        "p_value": float(
            model.pvalues["treated:week_index"]
        ),
    }