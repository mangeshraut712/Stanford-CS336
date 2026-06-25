#!/usr/bin/env python3
"""Train a BPE tokenizer and save vocab/merges to disk."""

from __future__ import annotations

import argparse
import json
import resource
import time
from pathlib import Path

import psutil

from cs336_basics.tokenizer import Tokenizer, train_bpe

SPECIAL_TOKEN = "<|endoftext|>"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--vocab-size", type=int, required=True)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--name", type=str, default="tokenizer")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    process = psutil.Process()
    start_rss = process.memory_info().rss
    start_time = time.time()
    vocab, merges = train_bpe(
        args.input,
        vocab_size=args.vocab_size,
        special_tokens=[SPECIAL_TOKEN],
        num_workers=args.num_workers,
    )
    elapsed = time.time() - start_time
    peak_rss = process.memory_info().rss
    tokenizer = Tokenizer(vocab, merges, special_tokens=[SPECIAL_TOKEN])
    tokenizer.save(args.output_dir)
    longest_token = max(vocab.values(), key=len)
    stats = {
        "name": args.name,
        "input": str(args.input),
        "vocab_size": len(vocab),
        "num_merges": len(merges),
        "elapsed_sec": elapsed,
        "peak_rss_gb": peak_rss / (1024**3),
        "start_rss_gb": start_rss / (1024**3),
        "longest_token_bytes": len(longest_token),
        "longest_token_repr": longest_token.decode("utf-8", errors="replace")[:200],
    }
    with open(args.output_dir / "train_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
