from __future__ import annotations

import pandas as pd

from .config import (
    DATE_COL,
    GEO_COL,
    REVENUE_COL,
    SPEND_COL,
    TREATMENT_DATE,
    POST_PERIOD_END,
)


def split_pre_post(
    panel: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split the prepared panel into pre-treatment and post-treatment periods."""
    pre = panel[
        panel[DATE_COL] < pd.Timestamp(TREATMENT_DATE)
    ].copy()

    post = panel[
        (panel[DATE_COL] >= pd.Timestamp(TREATMENT_DATE))
        & (panel[DATE_COL] <= pd.Timestamp(POST_PERIOD_END))
    ].copy()

    return pre, post


def get_treated_geos(panel: pd.DataFrame) -> list[str]:
    """Return sorted treated geo identifiers."""
    return sorted(
        panel.loc[panel["treated"] == 1, GEO_COL]
        .unique()
        .tolist()
    )


def get_control_geos(panel: pd.DataFrame) -> list[str]:
    """Return sorted control geo identifiers."""
    return sorted(
        panel.loc[panel["treated"] == 0, GEO_COL]
        .unique()
        .tolist()
    )


def daily_geo_matrix(
    panel: pd.DataFrame,
    geos: list[str],
    value_col: str = REVENUE_COL,
) -> pd.DataFrame:
    """Return a date × geo matrix for a selected metric."""
    return (
        panel[panel[GEO_COL].isin(geos)]
        .pivot(index=DATE_COL, columns=GEO_COL, values=value_col)
        .sort_index()
    )


def treated_daily_mean(
    panel: pd.DataFrame,
    value_col: str = REVENUE_COL,
) -> pd.Series:
    """Return daily mean metric across treated geos."""
    return (
        panel[panel["treated"] == 1]
        .groupby(DATE_COL)[value_col]
        .mean()
        .sort_index()
        .rename("treated")
    )


def control_daily_mean(
    panel: pd.DataFrame,
    value_col: str = REVENUE_COL,
) -> pd.Series:
    """Return daily mean metric across control geos."""
    return (
        panel[panel["treated"] == 0]
        .groupby(DATE_COL)[value_col]
        .mean()
        .sort_index()
        .rename("control")
    )


def pre_period_geo_matrix(
    panel: pd.DataFrame,
    geos: list[str],
    value_col: str = REVENUE_COL,
) -> pd.DataFrame:
    """Return the pre-treatment date × geo matrix."""
    pre, _ = split_pre_post(panel)
    return daily_geo_matrix(pre, geos, value_col)


def post_period_geo_matrix(
    panel: pd.DataFrame,
    geos: list[str],
    value_col: str = REVENUE_COL,
) -> pd.DataFrame:
    """Return the post-treatment date × geo matrix."""
    _, post = split_pre_post(panel)
    return daily_geo_matrix(post, geos, value_col)


def aggregate_period_means(
    panel: pd.DataFrame,
    value_col: str = REVENUE_COL,
) -> pd.DataFrame:
    """Return treated/control pre/post means for a selected metric."""
    pre, post = split_pre_post(panel)

    return pd.DataFrame(
        {
            "treated_pre": [
                pre.loc[pre["treated"] == 1, value_col].mean()
            ],
            "control_pre": [
                pre.loc[pre["treated"] == 0, value_col].mean()
            ],
            "treated_post": [
                post.loc[post["treated"] == 1, value_col].mean()
            ],
            "control_post": [
                post.loc[post["treated"] == 0, value_col].mean()
            ],
        }
    )