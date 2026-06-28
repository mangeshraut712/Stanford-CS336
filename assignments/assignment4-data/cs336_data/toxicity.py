"""NSFW and toxic speech classifiers (Dolma fastText)."""

from __future__ import annotations

from cs336_data.classifiers import load_fasttext_model, predict_label

NSFW_LABELS = {"nsfw", "non-nsfw"}
TOXIC_LABELS = {"toxic", "non-toxic"}


def classify_nsfw(text: str) -> tuple[str, float]:
    model = load_fasttext_model("dolma_fasttext_nsfw_jigsaw_model.bin")
    label, score = predict_label(model, text)
    if label not in NSFW_LABELS:
        label = "nsfw" if "nsfw" in label else "non-nsfw"
    return label, score


def classify_toxic_speech(text: str) -> tuple[str, float]:
    model = load_fasttext_model("dolma_fasttext_hatespeech_jigsaw_model.bin")
    label, score = predict_label(model, text)
    if label not in TOXIC_LABELS:
        label = "toxic" if "toxic" in label else "non-toxic"
    return label, score
