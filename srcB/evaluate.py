from __future__ import annotations

import numpy as np
import pandas as pd

from .config import CONFIDENCE_LEVEL


def placebo_interval(
    observed_effect: float,
    placebo_effects: pd.Series,
    confidence_level: float = CONFIDENCE_LEVEL,
) -> dict:
    """
    Construct a placebo-based uncertainty interval.

    The interval is obtained by inverting the placebo distribution:
        [observed - placebo_q_high,
         observed - placebo_q_low]
    """

    alpha = 1.0 - confidence_level

    lower_quantile = float(
        placebo_effects.quantile(alpha / 2)
    )

    upper_quantile = float(
        placebo_effects.quantile(1.0 - alpha / 2)
    )

    lower = observed_effect - upper_quantile
    upper = observed_effect - lower_quantile

    return {
        "lower": float(lower),
        "upper": float(upper),
        "placebo_q05": lower_quantile,
        "placebo_q95": upper_quantile,
    }


def placebo_p_value(
    observed_effect: float,
    placebo_effects: pd.Series,
) -> dict:
    """
    Calculate a two-sided empirical placebo p-value.

    Uses the finite-sample correction:
        (exceeding + 1) / (N + 1)
    """

    placebo_effects = np.asarray(
        placebo_effects,
        dtype=float,
    )

    exceeding = int(
        np.sum(
            np.abs(placebo_effects)
            >= abs(observed_effect)
        )
    )

    n_placebos = len(placebo_effects)

    corrected_p_value = (
        exceeding + 1
    ) / (
        n_placebos + 1
    )

    return {
        "n_placebos": n_placebos,
        "exceeding": exceeding,
        "p_value": float(corrected_p_value),
    }


def evaluate_estimate(
    observed_effect: float,
    treated_pre_total: float,
    placebo_results: pd.DataFrame,
    confidence_level: float = CONFIDENCE_LEVEL,
) -> dict:
    """
    Combine the primary estimate with placebo-based uncertainty.
    """

    placebo_effects = placebo_results["effect"]

    interval = placebo_interval(
        observed_effect=observed_effect,
        placebo_effects=placebo_effects,
        confidence_level=confidence_level,
    )

    p_value = placebo_p_value(
        observed_effect=observed_effect,
        placebo_effects=placebo_effects,
    )

    relative_effect = (
        observed_effect / treated_pre_total
    )

    return {
        "point_estimate": float(observed_effect),
        "relative_effect": float(relative_effect),
        "ci_lower_90": interval["lower"],
        "ci_upper_90": interval["upper"],
        "placebo_q05": interval["placebo_q05"],
        "placebo_q95": interval["placebo_q95"],
        "n_placebos": p_value["n_placebos"],
        "n_placebos_exceeding": p_value["exceeding"],
        "placebo_p_value": p_value["p_value"],
    }