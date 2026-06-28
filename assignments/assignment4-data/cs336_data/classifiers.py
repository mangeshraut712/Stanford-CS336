"""Load shared classifier assets."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import fasttext

from cs336_data.common import get_shared_assets_path


def classifier_path(name: str) -> Path:
    return get_shared_assets_path() / "classifiers" / name


@lru_cache(maxsize=8)
def load_fasttext_model(filename: str) -> fasttext.FastText._FastText:
    path = classifier_path(filename)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run: uv run scripts/download_data.py --offline-only"
        )
    # Clear cache if a previously partial download was cached.
    if filename == "dolma_fasttext_nsfw_jigsaw_model.bin" and path.stat().st_size < 900_000_000:
        raise FileNotFoundError(f"Incomplete {path}; re-run offline download.")
    return fasttext.load_model(str(path))


def predict_label(model: fasttext.FastText._FastText, text: str) -> tuple[str, float]:
    labels, probs = model.predict(text.replace("\n", " "), k=1)
    label = labels[0].removeprefix("__label__")
    return label, float(probs[0])
