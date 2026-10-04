import pandas as pd
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from srcB.config import (
    PANEL_PATH, DATE_COL, GEO_COL, REVENUE_COL, SPEND_COL, TREATMENT_COL,
    TREATMENT_DATE, PANEL_START_DATE, PANEL_END_DATE
)


REQUIRED_COLUMNS = {DATE_COL, GEO_COL, REVENUE_COL, SPEND_COL, TREATMENT_COL}


def load_panel(path=PANEL_PATH):
    df = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df = df.sort_values([GEO_COL, DATE_COL]).reset_index(drop=True)

    if df[[DATE_COL, GEO_COL]].duplicated().any():
        raise ValueError("Duplicate geo-date rows found.")

    if df[REVENUE_COL].isna().any() or df[SPEND_COL].isna().any():
        raise ValueError("Missing revenue/spend values found.")

    if (df[REVENUE_COL] < 0).any() or (df[SPEND_COL] < 0).any():
        raise ValueError("Negative revenue/spend values found.")

    return df


def validate_panel(df):
    dates = df[DATE_COL]
    expected_dates = pd.date_range(PANEL_START_DATE, PANEL_END_DATE, freq="D")

    assert len(df) == 10800, f"Expected 10800 rows, got {len(df)}"
    assert df[GEO_COL].nunique() == 60, "Expected 60 geos."
    assert dates.nunique() == 180, "Expected 180 dates."
    assert dates.min() == pd.Timestamp(PANEL_START_DATE)
    assert dates.max() == pd.Timestamp(PANEL_END_DATE)
    assert df[TREATMENT_COL].isin([0, 1]).all(), "treated_group must be 0/1."
    assert df[df[TREATMENT_COL] == 1][GEO_COL].nunique() == 6, "Expected 6 treated geos."
    assert df[df[TREATMENT_COL] == 0][GEO_COL].nunique() == 54, "Expected 54 control geos."
    assert set(df[DATE_COL].unique()) == set(expected_dates), "Unexpected date range."

    return True


def prepare_panel(df):
    df = df.copy()
    df["post"] = (df[DATE_COL] >= pd.Timestamp(TREATMENT_DATE)).astype(int)
    df["treated"] = df[TREATMENT_COL].astype(int)
    df["treated_post"] = df["treated"] * df["post"]
    return df


def load_and_prepare_panel(path=PANEL_PATH):
    df = load_panel(path)
    validate_panel(df)
    return prepare_panel(df)