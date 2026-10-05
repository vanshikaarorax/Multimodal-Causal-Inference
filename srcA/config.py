from pathlib import Path


# ============================================================
# Project paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CREATIVES_PATH = PROJECT_ROOT / "creatives.parquet"
CONFIRMED_PERSONAS_PATH = PROJECT_ROOT / "confirmed_personas.csv"
PERSONAS_PATH = PROJECT_ROOT / "personas.yaml"
DUP_PAIRS_EVAL_PATH = PROJECT_ROOT / "dup_pairs_eval.csv"
GEO_PANEL_PATH = PROJECT_ROOT / "geo_panel.csv"
IMAGES_DIR = PROJECT_ROOT / "images"

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"


# ============================================================
# Reproducibility
# ============================================================

SEED = 42
DUPLICATE_THRESHOLD =0.88


# ============================================================
# Part A embedding configuration
# ============================================================

IMAGE_TEXT_MODEL = "openai/clip-vit-base-patch32"
                    
# Persona embedding experiment:
# Image = 30%, Text = 70%
IMAGE_EMBEDDING_WEIGHT = 0.30
TEXT_EMBEDDING_WEIGHT = 0.70


# ============================================================
# Persona configuration
# ============================================================

PERSONA_NAMES = [
    "Parent & Family",
    "Homeowner",
    "Small Business & B2B",
    "Health & Wellness Seeker",
    "Tech Enthusiast",
    "Retiree & Senior",
    "Student & Learner",
    "Traveler & Leisure",
]


# ============================================================
# Validation configuration
# ============================================================

REQUIRED_CREATIVE_COLUMNS = {
    "creative_id",
    "ad_size",
    "ad_text",
    "image_path",
    "launched_at",
}

REQUIRED_PERSONA_COLUMNS = {
    "creative_id",
    "personas",
    "confirmed_by",
    "confirmed_at",
}

REQUIRED_DUPLICATE_COLUMNS = {
    "pair_id",
    "creative_id_a",
    "creative_id_b",
    "is_duplicate",
}


# ============================================================
# Embedding runtime configuration
# ============================================================

EMBEDDING_BATCH_SIZE = 32

EMBEDDING_DIMENSION = 512

EMBEDDINGS_DIR = ARTIFACTS_DIR / "embeddings"

IMAGE_EMBEDDINGS_DIR = EMBEDDINGS_DIR / "image"
TEXT_EMBEDDINGS_DIR = EMBEDDINGS_DIR / "text"
FUSED_EMBEDDINGS_DIR = EMBEDDINGS_DIR / "fused"

IMAGE_EMBEDDINGS_PATH = IMAGE_EMBEDDINGS_DIR / "embeddings.npy"
TEXT_EMBEDDINGS_PATH = TEXT_EMBEDDINGS_DIR / "embeddings.npy"
FUSED_EMBEDDINGS_PATH = FUSED_EMBEDDINGS_DIR / "embeddings.npy"

EMBEDDING_METADATA_PATH = EMBEDDINGS_DIR / "metadata.parquet"