from pathlib import Path


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts" / "partB"

PANEL_PATH = DATA_DIR / "geo_panel.csv"


# ============================================================
# ARTIFACT DIRECTORIES
# ============================================================

DIAGNOSTICS_DIR = ARTIFACTS_DIR / "diagnostics"
ESTIMATES_DIR = ARTIFACTS_DIR / "estimates"
PLACEBO_DIR = ARTIFACTS_DIR / "placebo"
FIGURES_DIR = ARTIFACTS_DIR / "figures"


# ============================================================
# REPRODUCIBILITY
# ============================================================

RANDOM_SEED = 42


# ============================================================
# EXPERIMENT WINDOWS
# ============================================================

PANEL_START_DATE = "2026-01-05"
PANEL_END_DATE = "2026-07-03"

TREATMENT_DATE = "2026-06-04"

PRE_PERIOD_START = "2026-01-05"
PRE_PERIOD_END = "2026-06-03"

POST_PERIOD_START = "2026-06-04"
POST_PERIOD_END = "2026-07-03"


# ============================================================
# DATA COLUMNS
# ============================================================

DATE_COL = "date"
GEO_COL = "geo"
REVENUE_COL = "revenue"
SPEND_COL = "spend"
TREATMENT_COL = "treated_group"


# ============================================================
# ESTIMATION SETTINGS
# ============================================================

CONFIDENCE_LEVEL = 0.90

# Number of placebo assignments for robustness checks.
N_PLACEBOS = 100