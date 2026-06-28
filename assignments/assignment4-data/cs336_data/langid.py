"""Language identification with fastText LID."""

from __future__ import annotations

from cs336_data.classifiers import load_fasttext_model, predict_label


def identify_language(text: str) -> tuple[str, float]:
    model = load_fasttext_model("lid.176.bin")
    return predict_label(model, text)


def make_is_english(threshold: float = 0.7):
    model = load_fasttext_model("lid.176.bin")

    def is_english(text: str) -> bool:
        label, prob = predict_label(model, text)
        return label == "en" and prob >= threshold

    return is_english
