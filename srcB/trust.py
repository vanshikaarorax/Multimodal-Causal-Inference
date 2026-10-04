from __future__ import annotations


def assign_trust_state(
    point_estimate: float,
    interval_low: float,
    interval_high: float,
    placebo_p_value: float,
    pre_rmse_relative: float,
    sensitivity_min: float,
    sensitivity_max: float,
    did_estimate: float,
    did_parallel_trends_valid: bool,
) -> str:
    """
    Deterministic Trust State rules for Part B.

    trusted:
        Positive effect with a non-zero 90% interval, strong pre-fit,
        stable magnitude, and no material estimator disagreement.

    directionally_trusted:
        Direction is robust, but magnitude has meaningful model dependence.

    magnitude_uncertain:
        Direction is supported, but magnitude is unstable.

    not_trusted:
        Direction itself is not reliably supported.
    """

    direction_supported = (
        point_estimate > 0
        and interval_low > 0
        and placebo_p_value < 0.10
        and sensitivity_min > 0
    )

    fit_good = pre_rmse_relative <= 0.05

    sensitivity_ratio = (
        (sensitivity_max - sensitivity_min)
        / abs(point_estimate)
    )

    magnitude_stable = sensitivity_ratio <= 0.20

    estimator_disagreement = (
        did_parallel_trends_valid
        and abs(did_estimate - point_estimate)
        / abs(point_estimate)
        > 0.50
    )

    if not direction_supported:
        return "not_trusted"

    if not magnitude_stable:
        return "magnitude_uncertain"

    if fit_good and not estimator_disagreement:
        return "trusted"

    return "directionally_trusted"