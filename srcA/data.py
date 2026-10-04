from pathlib import Path

import pandas as pd
import yaml

from srcA.config import (
    CREATIVES_PATH,
    CONFIRMED_PERSONAS_PATH,
    PERSONAS_PATH,
    DUP_PAIRS_EVAL_PATH,
    REQUIRED_CREATIVE_COLUMNS,
    REQUIRED_PERSONA_COLUMNS,
    REQUIRED_DUPLICATE_COLUMNS,
    PERSONA_NAMES,
)


# ============================================================
# Generic validation helpers
# ============================================================

def _validate_columns(
    dataframe: pd.DataFrame,
    required_columns: set[str],
    dataset_name: str,
) -> None:
    """Validate that a dataframe contains all required columns."""

    missing_columns = required_columns - set(dataframe.columns)

    if missing_columns:
        raise ValueError(
            f"{dataset_name} is missing required columns: "
            f"{sorted(missing_columns)}"
        )


def _validate_no_duplicate_ids(
    dataframe: pd.DataFrame,
    column: str,
    dataset_name: str,
) -> None:
    """Validate uniqueness of an ID column."""

    duplicate_count = dataframe[column].duplicated().sum()

    if duplicate_count > 0:
        raise ValueError(
            f"{dataset_name} contains {duplicate_count} duplicate "
            f"values in '{column}'."
        )


# ============================================================
# Creatives
# ============================================================

def load_creatives(path: Path = CREATIVES_PATH) -> pd.DataFrame:
    """Load and validate the creatives dataset."""

    dataframe = pd.read_parquet(path)

    _validate_columns(
        dataframe,
        REQUIRED_CREATIVE_COLUMNS,
        "creatives.parquet",
    )

    _validate_no_duplicate_ids(
        dataframe,
        "creative_id",
        "creatives.parquet",
    )

    if dataframe["creative_id"].isna().any():
        raise ValueError("creatives.parquet contains missing creative_id values.")

    if dataframe["image_path"].isna().any():
        raise ValueError("creatives.parquet contains missing image_path values.")

    if dataframe["ad_text"].isna().any():
        raise ValueError("creatives.parquet contains missing ad_text values.")

    return dataframe


# ============================================================
# Confirmed persona labels
# ============================================================

def load_confirmed_personas(
    path: Path = CONFIRMED_PERSONAS_PATH,
) -> pd.DataFrame:
    """Load and validate confirmed persona labels."""

    dataframe = pd.read_csv(path)

    _validate_columns(
        dataframe,
        REQUIRED_PERSONA_COLUMNS,
        "confirmed_personas.csv",
    )

    if dataframe["creative_id"].isna().any():
        raise ValueError(
            "confirmed_personas.csv contains missing creative_id values."
        )

    if dataframe["personas"].isna().any():
        raise ValueError(
            "confirmed_personas.csv contains missing persona values."
        )

    return dataframe


# ============================================================
# Persona definitions
# ============================================================

def load_personas(
    path: Path = PERSONAS_PATH,
) -> dict[str, dict]:
    """Load and validate persona definitions from personas.yaml."""

    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)

    if not isinstance(data, dict):
        raise ValueError("personas.yaml must contain a top-level mapping.")

    personas = data.get("personas")

    if not isinstance(personas, list):
        raise ValueError(
            "personas.yaml must contain a 'personas' list."
        )

    persona_map = {}

    for persona in personas:
        if not isinstance(persona, dict):
            raise ValueError(
                "Each persona entry must be a mapping."
            )

        name = persona.get("name")
        definition = persona.get("definition")

        if not name or not definition:
            raise ValueError(
                "Each persona must contain 'name' and 'definition'."
            )

        persona_map[name] = {
            "definition": definition,
        }

    missing_personas = set(PERSONA_NAMES) - set(persona_map.keys())

    if missing_personas:
        raise ValueError(
            "personas.yaml is missing personas: "
            f"{sorted(missing_personas)}"
        )

    return persona_map


# ============================================================
# Duplicate evaluation pairs
# ============================================================

def load_duplicate_pairs(
    path: Path = DUP_PAIRS_EVAL_PATH,
) -> pd.DataFrame:
    """Load and validate the held-out duplicate evaluation pairs."""

    dataframe = pd.read_csv(path)

    _validate_columns(
        dataframe,
        REQUIRED_DUPLICATE_COLUMNS,
        "dup_pairs_eval.csv",
    )

    _validate_no_duplicate_ids(
        dataframe,
        "pair_id",
        "dup_pairs_eval.csv",
    )

    if not dataframe["is_duplicate"].isin([0, 1]).all():
        raise ValueError(
            "is_duplicate must contain only 0/1 values."
        )

    return dataframe


# ============================================================
# Image path validation
# ============================================================

def validate_image_paths(
    creatives: pd.DataFrame,
    images_dir: Path,
) -> None:
    """Verify that every creative points to an existing image."""

    missing_images = []

    for image_path in creatives["image_path"]:
        image_path = Path(image_path)

        # image_path may already include the images/ directory.
        if image_path.is_absolute():
            full_path = image_path
        elif image_path.parts and image_path.parts[0] == images_dir.name:
            full_path = images_dir.parent / image_path
        else:
            full_path = images_dir / image_path

        if not full_path.exists():
            missing_images.append(str(full_path))

    if missing_images:
        preview = missing_images[:10]

        raise FileNotFoundError(
            f"{len(missing_images)} creative images are missing. "
            f"Examples: {preview}"
        )