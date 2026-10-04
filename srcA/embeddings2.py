from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, SiglipModel


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from srcA.config import (
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_DIMENSION,
    EMBEDDING_METADATA_PATH,
    FUSED_EMBEDDINGS_DIR,
    FUSED_EMBEDDINGS_PATH,
    IMAGE_EMBEDDINGS_DIR,
    IMAGE_EMBEDDINGS_PATH,
    IMAGE_EMBEDDING_WEIGHT,
    IMAGE_TEXT_MODEL,
    TEXT_EMBEDDINGS_DIR,
    TEXT_EMBEDDINGS_PATH,
    TEXT_EMBEDDING_WEIGHT,
)

from srcA.data import load_creatives


# ============================================================
# 1. TEXT PREPROCESSING
# ============================================================


def normalize_text(text: str) -> str:
    """
    Apply minimal normalization to the provided ad text.

    The assignment already provides OCR-derived ad_text,
    so no additional OCR correction is performed.
    """
    if not isinstance(text, str):
        return ""

    return " ".join(text.split())


def prepare_texts(texts: list[str]) -> list[str]:
    """Normalize a list of ad texts."""
    return [normalize_text(text) for text in texts]


# ============================================================
# 2. DEVICE
# ============================================================


def get_device() -> torch.device:
    """
    Use Apple MPS when available, otherwise fall back to CPU.
    """
    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


# ============================================================
# 3. MODEL LOADING
# ============================================================


def load_embedding_model(model_name: str = IMAGE_TEXT_MODEL):
    """
    Load the SigLIP processor and model.
    """
    device = get_device()

    processor = AutoProcessor.from_pretrained(model_name)
    model = SiglipModel.from_pretrained(model_name)

    model.to(device)
    model.eval()

    return processor, model, device


# ============================================================
# 4. IMAGE LOADING
# ============================================================


def load_image(image_path: Path) -> Image.Image:
    """Load one image and convert it to RGB."""
    with Image.open(image_path) as image:
        return image.convert("RGB")


def load_images(image_paths: list[Path]) -> list[Image.Image]:
    """Load a batch of images."""
    return [load_image(path) for path in image_paths]


# ============================================================
# 5. IMAGE ENCODING
# ============================================================


def encode_image_batch(
    images,
    processor,
    model,
    device,
) -> np.ndarray:
    """
    Encode one batch of images into SigLIP embeddings.
    """
    inputs = processor(
        images=images,
        return_tensors="pt",
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.inference_mode():
        features = model.get_image_features(**inputs)

    features = torch.nn.functional.normalize(
        features,
        p=2,
        dim=1,
    )

    return features.cpu().numpy().astype(np.float32)


def encode_images(
    image_paths: list[Path],
    processor,
    model,
    device,
    batch_size: int = EMBEDDING_BATCH_SIZE,
) -> np.ndarray:
    """
    Encode all creative images in batches.
    """
    embeddings = []

    for start in tqdm(
        range(0, len(image_paths), batch_size),
        desc="Encoding images",
    ):
        batch_paths = image_paths[start:start + batch_size]

        images = load_images(batch_paths)

        batch_embeddings = encode_image_batch(
            images=images,
            processor=processor,
            model=model,
            device=device,
        )

        embeddings.append(batch_embeddings)

    return np.vstack(embeddings)


# ============================================================
# 6. TEXT ENCODING
# ============================================================


def encode_text_batch(
    texts,
    processor,
    model,
    device,
) -> np.ndarray:
    """
    Encode one batch of ad texts into SigLIP embeddings.
    """
    inputs = processor(
        text=texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.inference_mode():
        features = model.get_text_features(**inputs)

    features = torch.nn.functional.normalize(
        features,
        p=2,
        dim=1,
    )

    return features.cpu().numpy().astype(np.float32)


def encode_texts(
    texts: list[str],
    processor,
    model,
    device,
    batch_size: int = EMBEDDING_BATCH_SIZE,
) -> np.ndarray:
    """
    Encode all ad texts in batches.
    """
    texts = prepare_texts(texts)

    embeddings = []

    for start in tqdm(
        range(0, len(texts), batch_size),
        desc="Encoding text",
    ):
        batch_texts = texts[start:start + batch_size]

        batch_embeddings = encode_text_batch(
            texts=batch_texts,
            processor=processor,
            model=model,
            device=device,
        )

        embeddings.append(batch_embeddings)

    return np.vstack(embeddings)


# ============================================================
# 7. EMBEDDING NORMALIZATION
# ============================================================


def normalize_embeddings(
    embeddings: np.ndarray,
) -> np.ndarray:
    """
    L2-normalize embeddings row-wise.

    After normalization, cosine similarity can be computed
    efficiently using a dot product.
    """
    norms = np.linalg.norm(
        embeddings,
        axis=1,
        keepdims=True,
    )

    return (
        embeddings /
        np.clip(norms, 1e-12, None)
    ).astype(np.float32)


# ============================================================
# 8. IMAGE + TEXT FUSION
# ============================================================


def fuse_embeddings(
    image_embeddings: np.ndarray,
    text_embeddings: np.ndarray,
    image_weight: float = 0.3,
    text_weight: float = 0.7,
) -> np.ndarray:
    """
    Fuse image and text embeddings using a weighted average.

    Both modalities are normalized before fusion and the fused
    representation is normalized again afterwards.
    """
    if image_embeddings.shape != text_embeddings.shape:
        raise ValueError(
            "Image and text embeddings must have the same shape."
        )

    if image_weight < 0 or text_weight < 0:
        raise ValueError(
            "Image and text weights must be non-negative."
        )

    if not np.isclose(image_weight + text_weight, 1.0):
        raise ValueError(
            "Image and text weights must sum to 1."
        )

    image_embeddings = normalize_embeddings(image_embeddings)
    text_embeddings = normalize_embeddings(text_embeddings)

    fused_embeddings = (
        image_weight * image_embeddings
        + text_weight * text_embeddings
    )

    return normalize_embeddings(fused_embeddings)


# ============================================================
# 9. METADATA
# ============================================================


def build_embedding_metadata(
    creatives: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create metadata aligned row-by-row with embeddings.

    Row i in every embedding matrix corresponds to row i here.
    """
    return creatives[
        [
            "creative_id",
            "ad_size",
            "image_path",
            "launched_at",
        ]
    ].reset_index(drop=True)


# ============================================================
# 10. ARTIFACT DIRECTORIES
# ============================================================


def create_embedding_directories() -> None:
    """Create directories used to store embedding artifacts."""
    for directory in (
        IMAGE_EMBEDDINGS_DIR,
        TEXT_EMBEDDINGS_DIR,
        FUSED_EMBEDDINGS_DIR,
    ):
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )


# ============================================================
# 11. SAVE EMBEDDINGS
# ============================================================


def save_embeddings(
    image_embeddings: np.ndarray,
    text_embeddings: np.ndarray,
    fused_embeddings: np.ndarray,
    metadata: pd.DataFrame,
) -> None:
    """
    Save image, text and fused embeddings plus metadata.
    """
    create_embedding_directories()

    np.save(
        IMAGE_EMBEDDINGS_PATH,
        image_embeddings,
    )

    np.save(
        TEXT_EMBEDDINGS_PATH,
        text_embeddings,
    )

    np.save(
        FUSED_EMBEDDINGS_PATH,
        fused_embeddings,
    )

    metadata.to_parquet(
        EMBEDDING_METADATA_PATH,
        index=False,
    )


# ============================================================
# 12. LOAD SAVED EMBEDDINGS
# ============================================================


def load_embeddings():
    """
    Load previously generated embedding artifacts.
    """
    image_embeddings = np.load(
        IMAGE_EMBEDDINGS_PATH
    )

    text_embeddings = np.load(
        TEXT_EMBEDDINGS_PATH
    )

    fused_embeddings = np.load(
        FUSED_EMBEDDINGS_PATH
    )

    metadata = pd.read_parquet(
        EMBEDDING_METADATA_PATH
    )

    return (
        image_embeddings,
        text_embeddings,
        fused_embeddings,
        metadata,
    )


# ============================================================
# 13. VALIDATION
# ============================================================


def validate_embedding_shapes(
    image_embeddings: np.ndarray,
    text_embeddings: np.ndarray,
    fused_embeddings: np.ndarray,
    metadata: pd.DataFrame,
) -> None:
    """
    Verify that all embedding matrices have the expected
    number of rows and embedding dimension.
    """
    n = len(metadata)

    expected_shape = (
        n,
        EMBEDDING_DIMENSION,
    )

    for name, embeddings in (
        ("image", image_embeddings),
        ("text", text_embeddings),
        ("fused", fused_embeddings),
    ):
        if embeddings.shape != expected_shape:
            raise ValueError(
                f"{name} embeddings have shape "
                f"{embeddings.shape}; expected "
                f"{expected_shape}."
            )


# ============================================================
# 14. FULL EMBEDDING PIPELINE
# ============================================================


def build_embeddings(
    creatives: pd.DataFrame,
):
    """
    Build and persist image, text and fused embeddings.
    """
    processor, model, device = load_embedding_model()

    image_paths = [
        PROJECT_ROOT / path
        for path in creatives["image_path"]
    ]

    texts = creatives["ad_text"].tolist()

    image_embeddings = encode_images(
        image_paths=image_paths,
        processor=processor,
        model=model,
        device=device,
    )

    text_embeddings = encode_texts(
        texts=texts,
        processor=processor,
        model=model,
        device=device,
    )

    fused_embeddings = fuse_embeddings(
        image_embeddings=image_embeddings,
        text_embeddings=text_embeddings,
        image_weight=IMAGE_EMBEDDING_WEIGHT,
        text_weight=TEXT_EMBEDDING_WEIGHT,
    )

    metadata = build_embedding_metadata(
        creatives
    )

    validate_embedding_shapes(
        image_embeddings=image_embeddings,
        text_embeddings=text_embeddings,
        fused_embeddings=fused_embeddings,
        metadata=metadata,
    )

    save_embeddings(
        image_embeddings=image_embeddings,
        text_embeddings=text_embeddings,
        fused_embeddings=fused_embeddings,
        metadata=metadata,
    )

    return (
        image_embeddings,
        text_embeddings,
        fused_embeddings,
        metadata,
    )


def main() -> None:
    """
    Build and save all image, text, and fused embeddings.
    """
    print("=" * 60)
    print("MemoLogs - SigLIP Embedding Pipeline")
    print("=" * 60)

    print("\n[1/4] Loading creatives...")

    creatives = load_creatives()

    print(f"Loaded {len(creatives):,} creatives.")

    print("\n[2/4] Building embeddings...")

    (
        image_embeddings,
        text_embeddings,
        fused_embeddings,
        metadata,
    ) = build_embeddings(creatives)

    print("\n[3/4] Embedding shapes:")

    print(
        f"Image embeddings : {image_embeddings.shape}"
    )

    print(
        f"Text embeddings  : {text_embeddings.shape}"
    )

    print(
        f"Fused embeddings : {fused_embeddings.shape}"
    )

    print("\n[4/4] Saved artifacts:")

    print(
        f"Image : {IMAGE_EMBEDDINGS_PATH}"
    )

    print(
        f"Text  : {TEXT_EMBEDDINGS_PATH}"
    )

    print(
        f"Fused : {FUSED_EMBEDDINGS_PATH}"
    )

    print(
        f"Meta  : {EMBEDDING_METADATA_PATH}"
    )

    print("\n" + "=" * 60)
    print("SigLIP embedding pipeline completed successfully.")
    print("=" * 60)


if __name__ == "__main__":
    main()