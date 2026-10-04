from __future__ import annotations

import argparse
from .config import PANEL_PATH, N_PLACEBOS, RANDOM_SEED
from .pipeline import run_pipeline


def main():
    parser = argparse.ArgumentParser(description="Run Part B geo-panel causal lift analysis.")
    parser.add_argument("--panel", default=str(PANEL_PATH), help="Path to geo_panel.csv")
    parser.add_argument("--placebos", type=int, default=N_PLACEBOS, help="Number of placebo assignments")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help="Random seed")
    args = parser.parse_args()

    results = run_pipeline(panel_path=args.panel, n_placebos=args.placebos, seed=args.seed)

    evaluation = results["evaluation"]
    print("\nPart B Results")
    print(f"Point estimate: ₹{evaluation['point_estimate']:,.2f}")
    print(f"90% interval: ₹{evaluation['ci_lower_90']:,.2f} to ₹{evaluation['ci_upper_90']:,.2f}")
    print(f"Relative effect: {evaluation['relative_effect']:.2%}")
    print(f"Placebo p-value: {evaluation['placebo_p_value']:.4f}")
    print(f"Trust State: {results['trust_state']}")


if __name__ == "__main__":
    main()