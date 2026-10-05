from __future__ import annotations

import json
import pandas as pd

from .config import (
    ARTIFACTS_DIR, DIAGNOSTICS_DIR, ESTIMATES_DIR, PLACEBO_DIR,
    RANDOM_SEED, N_PLACEBOS, CONFIDENCE_LEVEL, SPEND_COL,
)
from .data import load_and_prepare_panel
from .diagnostics import (
    pre_treatment_correlation, indexed_pre_treatment_correlation,
    spend_diagnostic, synthetic_pre_fit_metrics, weekly_pretrend_test,
    pretrend_gap_by_week, treated_geo_pretrend_test,
    leave_one_treated_out_pretrend, pretrend_window_sensitivity,
    weekly_covariate_pretrend_test,
)
from .estimators import (
    fit_synthetic_control,
    fit_did,
    leave_one_donor_out,
    pretrend_window_synthetic_sensitivity,
)
from .placebo import run_placebos
from .evaluate import evaluate_estimate
from .trust import assign_trust_state


def _save_json(data, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def run_pipeline(panel_path=None, n_placebos=N_PLACEBOS, seed=RANDOM_SEED):
    panel = load_and_prepare_panel(panel_path) if panel_path else load_and_prepare_panel()

    synthetic = fit_synthetic_control(panel)
    pre_fit = synthetic_pre_fit_metrics(
        synthetic["treated_pre"], synthetic["synthetic_pre"]
    )

    did = fit_did(panel)
    placebos = run_placebos(panel, n_placebos=n_placebos, seed=seed)

    evaluation = evaluate_estimate(
        synthetic["point_estimate"],
        synthetic["treated_pre"].sum(),
        placebos,
        confidence_level=CONFIDENCE_LEVEL,
    )

    sensitivity = leave_one_donor_out(panel, baseline_result=synthetic)

    spend = spend_diagnostic(panel)
    revenue_corr = pre_treatment_correlation(panel)
    indexed_corr = indexed_pre_treatment_correlation(panel)
    pretrend = weekly_pretrend_test(panel)

    # Deeper pre-trend investigation
    pretrend_gap = pretrend_gap_by_week(panel)
    geo_pretrend = treated_geo_pretrend_test(panel)
    geo_leave_one_out = leave_one_treated_out_pretrend(panel)
    pretrend_windows = pretrend_window_sensitivity(panel)
    sc_window_sensitivity = pretrend_window_synthetic_sensitivity(panel)
    spend_pretrend = weekly_covariate_pretrend_test(panel, SPEND_COL)

    did_parallel_trends_valid = pretrend["p_value"] >= 0.10

    trust_state = assign_trust_state(
        point_estimate=synthetic["point_estimate"],
        interval_low=evaluation["ci_lower_90"],
        interval_high=evaluation["ci_upper_90"],
        placebo_p_value=evaluation["placebo_p_value"],
        pre_rmse_relative=pre_fit["relative_rmse"],
        sensitivity_min=sensitivity["effect"].min(),
        sensitivity_max=sensitivity["effect"].max(),
        did_estimate=did["total_effect"],
        did_parallel_trends_valid=did_parallel_trends_valid,
    )

    diagnostics = {
        "pre_treatment_revenue_correlation": revenue_corr,
        "indexed_pre_treatment_revenue_correlation": indexed_corr,
        "weekly_pretrend": pretrend,
        "synthetic_pre_fit": pre_fit,
        "spend_diagnostic": spend,
        "did_parallel_trends_valid": did_parallel_trends_valid,
        "pretrend_window_sensitivity": pretrend_windows.to_dict("records"),
        "spend_pretrend": spend_pretrend,
    }

    estimates = {
        "point_estimate": synthetic["point_estimate"],
        "mean_daily_incremental": synthetic["mean_daily_incremental"],
        "mean_relative_lift": synthetic["mean_relative_lift"],
        "pre_fit_relative_rmse": pre_fit["relative_rmse"],
        "ci_lower_90": evaluation["ci_lower_90"],
        "ci_upper_90": evaluation["ci_upper_90"],
        "placebo_p_value": evaluation["placebo_p_value"],
        "did_total_effect": did["total_effect"],
        "did_ci_lower_90": did["total_ci_lower_90"],
        "did_ci_upper_90": did["total_ci_upper_90"],
        "trust_state": trust_state,
    }

    for directory in [
        ARTIFACTS_DIR, DIAGNOSTICS_DIR, ESTIMATES_DIR, PLACEBO_DIR
    ]:
        directory.mkdir(parents=True, exist_ok=True)

    _save_json(diagnostics, DIAGNOSTICS_DIR / "diagnostics.json")
    _save_json(estimates, ESTIMATES_DIR / "estimates.json")

    placebos.to_csv(PLACEBO_DIR / "placebo_results.csv", index=False)
    sensitivity.to_csv(ESTIMATES_DIR / "leave_one_donor_out.csv", index=False)
    synthetic["weights"].rename("weight").to_csv(
        ESTIMATES_DIR / "synthetic_weights.csv"
    )

    pretrend_gap.to_csv(
        DIAGNOSTICS_DIR / "pretrend_gap_by_week.csv", index=False
    )
    geo_pretrend.to_csv(
        DIAGNOSTICS_DIR / "treated_geo_pretrend.csv", index=False
    )
    geo_leave_one_out.to_csv(
        DIAGNOSTICS_DIR / "leave_one_treated_out_pretrend.csv", index=False
    )
    pretrend_windows.to_csv(
        DIAGNOSTICS_DIR / "pretrend_window_sensitivity.csv", index=False
    )
    pd.DataFrame([spend_pretrend]).to_csv(
        DIAGNOSTICS_DIR / "spend_pretrend.csv", index=False
    )
    sc_window_sensitivity.to_csv(
    DIAGNOSTICS_DIR / "synthetic_control_window_sensitivity.csv",
    index=False,
)
    return {
        "panel": panel,
        "synthetic": synthetic,
        "pre_fit": pre_fit,
        "did": did,
        "placebos": placebos,
        "evaluation": evaluation,
        "sensitivity": sensitivity,
        "diagnostics": diagnostics,
        "pretrend_gap": pretrend_gap,
        "treated_geo_pretrend": geo_pretrend,
        "treated_geo_leave_one_out": geo_leave_one_out,
        "pretrend_windows": pretrend_windows,
        "spend_pretrend": spend_pretrend,
        "trust_state": trust_state,
        "synthetic_control_window_sensitivity": (
    sc_window_sensitivity.to_dict("records")
)
    }