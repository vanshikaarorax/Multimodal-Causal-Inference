from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.utils import check_random_state

from .config import (
    RANDOM_SEED,
    N_PLACEBOS,
    DATE_COL,
    GEO_COL,
    REVENUE_COL,
)
from .design import split_pre_post


def _fit_placebo_weights(
    placebo_pre: pd.Series,
    donor_pre: pd.DataFrame,
) -> pd.Series:
    """
    Fit placebo synthetic-control weights.

    This intentionally matches the optimizer configuration
    used in the Part B notebook.
    """

    placebo_scale = placebo_pre.mean()
    donor_scales = donor_pre.mean(axis=0)

    X = donor_pre.div(
        donor_scales,
        axis=1,
    ).values

    y = (
        placebo_pre / placebo_scale
    ).values

    n_donors = len(donor_pre.columns)

    w0 = np.ones(n_donors) / n_donors

    def objective(weights):
        return np.mean(
            (y - X @ weights) ** 2
        )

    result = minimize(
        objective,
        w0,
        method="SLSQP",
        bounds=[(0, 1)] * n_donors,
        constraints={
            "type": "eq",
            "fun": lambda w: np.sum(w) - 1,
        },
        options={
            "maxiter": 5000,
            "ftol": 1e-8,
        },
    )

    if not result.success:
        return None

    weights = pd.Series(
        result.x,
        index=donor_pre.columns,
    )

    return weights[weights > 1e-4]


def run_placebos(
    panel: pd.DataFrame,
    n_placebos: int = N_PLACEBOS,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """
    Run six-geo placebo synthetic-control assignments.

    Matches the Part B notebook implementation:
    - sklearn check_random_state
    - six placebo-treated geos
    - remaining geos as donors
    - normalized trajectories
    - SLSQP synthetic-control weights
    """

    pre, post = split_pre_post(panel)

    all_control_geos = (
        panel.loc[
            panel["treated"] == 0,
            GEO_COL,
        ]
        .unique()
        .tolist()
    )

    # Exact RNG used in the notebook.
    rng = check_random_state(seed)

    placebo_results = []

    for iteration in range(n_placebos):

        # Exact placebo assignment from notebook.
        placebo_geos = rng.choice(
            all_control_geos,
            size=6,
            replace=False,
        ).tolist()

        donor_geos = [
            geo
            for geo in all_control_geos
            if geo not in placebo_geos
        ]

        placebo_pre = (
            pre[
                pre[GEO_COL].isin(placebo_geos)
            ]
            .groupby(DATE_COL)[REVENUE_COL]
            .mean()
        )

        placebo_post = (
            post[
                post[GEO_COL].isin(placebo_geos)
            ]
            .groupby(DATE_COL)[REVENUE_COL]
            .mean()
        )

        donor_pre = (
            pre[
                pre[GEO_COL].isin(donor_geos)
            ]
            .pivot(
                index=DATE_COL,
                columns=GEO_COL,
                values=REVENUE_COL,
            )
        )

        donor_post = (
            post[
                post[GEO_COL].isin(donor_geos)
            ]
            .pivot(
                index=DATE_COL,
                columns=GEO_COL,
                values=REVENUE_COL,
            )
        )

        placebo_scale = placebo_pre.mean()
        donor_scales = donor_pre.mean()

        weights = _fit_placebo_weights(
            placebo_pre,
            donor_pre,
        )

        if weights is None:
            continue

        donor_post_norm = donor_post.div(
            donor_scales,
            axis=1,
        )

        synthetic_post = (
            donor_post_norm[
                weights.index
            ].values
            @ weights.values
        ) * placebo_scale

        placebo_effect = float(
            (
                placebo_post.values
                - synthetic_post
            ).sum()
        )

        relative_effect = float(
            placebo_effect
            / placebo_pre.sum()
        )

        placebo_results.append(
            {
                "iteration": iteration,
                "placebo_geos": ",".join(
                    placebo_geos
                ),
                "effect": placebo_effect,
                "relative_effect": relative_effect,
            }
        )

    return pd.DataFrame(placebo_results)