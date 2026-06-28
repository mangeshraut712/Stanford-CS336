"""Gopher quality heuristics and cc/wiki quality classifier."""

from __future__ import annotations

import re
import tempfile
from functools import lru_cache
from pathlib import Path

import fasttext

from cs336_data.classifiers import predict_label

QUALITY_MODEL_PATH = Path(__file__).resolve().parent / "models" / "quality_cc_wiki.bin"


def _non_symbol_words(text: str) -> list[str]:
    return [word for word in re.findall(r"\S+", text) if re.search(r"[^\W_]", word, flags=re.UNICODE)]


def gopher_quality_filter(text: str) -> bool:
    words = _non_symbol_words(text)
    if len(words) < 50 or len(words) > 100_000:
        return False

    avg_word_len = sum(len(word) for word in words) / len(words)
    if avg_word_len < 3 or avg_word_len > 10:
        return False

    lines = text.splitlines() or [text]
    if lines:
        ellipsis_lines = sum(1 for line in lines if line.rstrip().endswith("..."))
        if ellipsis_lines / len(lines) > 0.3:
            return False

    alpha_words = sum(1 for word in words if re.search(r"[A-Za-z]", word))
    if alpha_words / len(words) < 0.8:
        return False

    return True


def _training_corpus() -> list[tuple[str, str]]:
    fixtures = Path(__file__).resolve().parents[1] / "tests" / "fixtures"
    cc = (fixtures / "low_quality_cc.txt").read_text()
    wiki = (fixtures / "high_quality_wiki_reference.txt").read_text()
    samples: list[tuple[str, str]] = [
        ("cc", cc),
        ("wiki", wiki),
        ("cc", cc + "\n" + cc),
        ("wiki", wiki[:5000]),
        ("wiki", wiki[:2000]),
        (
            "cc",
            "Forum Index Search Memberlist Register Profile Log in phpBB powered forum "
            "Contact the webmaster Copyright powered by phpBB",
        ),
        (
            "cc",
            "Teach English Abroad job discussion forums FAQ search register usergroups "
            "TEFL courses powered by phpBB all rights reserved",
        ),
        (
            "wiki",
            "First published substantive revision encyclopedia article describes theory "
            "grounded in moral claims about individual liberty political philosophy",
        ),
        (
            "wiki",
            "Anarchism is a political theory skeptical of authority and power. "
            "Philosophical anarchism describes anti-foundationalism and consensus building.",
        ),
    ]
    return samples


@lru_cache(maxsize=1)
def _quality_model() -> fasttext.FastText._FastText:
    if QUALITY_MODEL_PATH.exists():
        return fasttext.load_model(str(QUALITY_MODEL_PATH))

    QUALITY_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as tmp:
        for label, text in _training_corpus():
            line = " ".join(text.split())
            tmp.write(f"__label__{label} {line}\n")
        train_path = Path(tmp.name)

    model = fasttext.train_supervised(
        input=str(train_path),
        epoch=25,
        lr=0.5,
        wordNgrams=2,
        minCount=1,
        verbose=0,
    )
    model.save_model(str(QUALITY_MODEL_PATH))
    train_path.unlink(missing_ok=True)
    return model


def classify_quality(text: str) -> tuple[str, float]:
    model = _quality_model()
    return predict_label(model, text)
