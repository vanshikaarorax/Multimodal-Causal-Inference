import numpy as np
import pandas as pd

from srcB.config import RANDOM_SEED, N_PLACEBOS
from srcB.data import load_and_prepare_panel
from srcB.design import split_pre_post, get_treated_geos, get_control_geos
from srcB.diagnostics import synthetic_pre_fit_metrics, weekly_pretrend_test
from srcB.estimators import fit_synthetic_control, fit_did, leave_one_donor_out
from srcB.placebo import run_placebos
from srcB.evaluate import evaluate_estimate
from srcB.trust import assign_trust_state


def test_panel_design():
    panel = load_and_prepare_panel()
    pre, post = split_pre_post(panel)

    assert len(panel) == 10800
    assert len(pre) == 9000
    assert len(post) == 1800
    assert len(get_treated_geos(panel)) == 6
    assert len(get_control_geos(panel)) == 54
    assert panel["treated_post"].sum() == 180


def test_synthetic_control_result():
    panel = load_and_prepare_panel()
    result = fit_synthetic_control(panel)

    assert result["point_estimate"] > 0
    assert result["mean_daily_incremental"] > 0
    assert result["mean_relative_lift"] > 0

    assert len(result["weights"]) == 21
    assert np.isclose(result["weights"].sum(), 1.0, atol=1e-6)

    fit = synthetic_pre_fit_metrics(
        result["treated_pre"],
        result["synthetic_pre"],
    )

    assert fit["relative_rmse"] < 0.05
    assert fit["correlation"] > 0.90


def test_placebo_reproducibility():
    panel = load_and_prepare_panel()

    first = run_placebos(panel, n_placebos=N_PLACEBOS, seed=RANDOM_SEED)
    second = run_placebos(panel, n_placebos=N_PLACEBOS, seed=RANDOM_SEED)

    pd.testing.assert_frame_equal(first, second)
    assert len(first) == N_PLACEBOS
    assert first["effect"].notna().all()


def test_evaluation_has_positive_interval():
    panel = load_and_prepare_panel()
    synthetic = fit_synthetic_control(panel)
    placebos = run_placebos(panel, n_placebos=N_PLACEBOS, seed=RANDOM_SEED)

    evaluation = evaluate_estimate(
        synthetic["point_estimate"],
        synthetic["treated_pre"].sum(),
        placebos,
    )

    assert evaluation["point_estimate"] > 0
    assert evaluation["ci_lower_90"] < evaluation["point_estimate"]
    assert evaluation["ci_upper_90"] > evaluation["point_estimate"]
    assert evaluation["ci_lower_90"] > 0
    assert evaluation["placebo_p_value"] < 0.10


def test_did_and_pretrend_are_computed():
    panel = load_and_prepare_panel()

    did = fit_did(panel)
    pretrend = weekly_pretrend_test(panel)

    assert np.isfinite(did["total_effect"])
    assert np.isfinite(did["total_ci_lower_90"])
    assert np.isfinite(did["total_ci_upper_90"])

    assert np.isfinite(pretrend["coefficient_per_week"])
    assert np.isfinite(pretrend["p_value"])


def test_donor_sensitivity():
    panel = load_and_prepare_panel()
    synthetic = fit_synthetic_control(panel)
    sensitivity = leave_one_donor_out(panel, baseline_result=synthetic)

    assert len(sensitivity) == len(synthetic["weights"])
    assert sensitivity["effect"].notna().all()
    assert (sensitivity["effect"] > 0).all()


def test_trust_state_rules():
    assert assign_trust_state(
        point_estimate=266422.99,
        interval_low=237778.37,
        interval_high=294463.41,
        placebo_p_value=0.0099,
        pre_rmse_relative=0.0204,
        sensitivity_min=256472.18,
        sensitivity_max=275650.59,
        did_estimate=2132576.73,
        did_parallel_trends_valid=False,
    ) == "trusted"

    assert assign_trust_state(
        point_estimate=100000,
        interval_low=-50000,
        interval_high=200000,
        placebo_p_value=0.50,
        pre_rmse_relative=0.10,
        sensitivity_min=-100000,
        sensitivity_max=200000,
        did_estimate=100000,
        did_parallel_trends_valid=False,
    ) == "not_trusted"