#!/usr/bin/env python3
"""Tokenize plain text with GPT-2 into uint16 memmap for training."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer


def tokenize_file(input_path: Path, output_path: Path) -> int:
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    text = input_path.read_text()
    ids = tokenizer.encode(text)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.array(ids, dtype=np.uint16)
    arr.tofile(output_path)
    return len(ids)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    n = tokenize_file(args.input, args.output)
    print(f"Tokenized {n:,} tokens -> {args.output}")


if __name__ == "__main__":
    main()
