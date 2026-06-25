#!/usr/bin/env python3
"""Encode a text corpus into a uint16 token memmap array."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

from cs336_basics.tokenizer import Tokenizer

SPECIAL_TOKEN = "<|endoftext|>"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokenizer-dir", type=Path, required=True)
    parser.add_argument("--chunk-size", type=int, default=1_000_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tokenizer = Tokenizer.from_files(
        args.tokenizer_dir / "vocab.json",
        args.tokenizer_dir / "merges.txt",
        special_tokens=[SPECIAL_TOKEN],
    )
    start = time.time()
    num_tokens = 0
    batch: list[int] = []
    batch_limit = 500_000
    input_bytes = args.input.stat().st_size

    def tracked_chunks(handle, size: int, pbar):
        while True:
            data = handle.read(size)
            if not data:
                break
            pbar.update(len(data.encode("utf-8")))
            yield data

    with open(args.output, "wb") as out_f, open(args.input, encoding="utf-8") as in_f:
        with tqdm(total=input_bytes, unit="B", unit_scale=True, desc=f"tokenize {args.input.name}") as pbar:
            for token_id in tokenizer.encode_iterable(tracked_chunks(in_f, args.chunk_size, pbar)):
                batch.append(token_id)
                if len(batch) >= batch_limit:
                    np.asarray(batch, dtype=np.uint16).tofile(out_f)
                    num_tokens += len(batch)
                    batch = []
            if batch:
                np.asarray(batch, dtype=np.uint16).tofile(out_f)
                num_tokens += len(batch)
            pbar.update(input_bytes - pbar.n)
    meta = {
        "input": str(args.input),
        "output": str(args.output),
        "num_tokens": num_tokens,
        "dtype": "uint16",
        "elapsed_sec": time.time() - start,
        "max_token_id": int(np.memmap(args.output, dtype=np.uint16, mode="r").max()) if num_tokens else 0,
        "why_uint16": "vocab sizes up to 32k fit in uint16; halves storage vs int32/int64 for memmap training.",
    }
    with open(args.output.with_suffix(".meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
