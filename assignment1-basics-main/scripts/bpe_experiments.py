#!/usr/bin/env python3
"""Run §2.7 tokenizer experiments and write results to artifacts/."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

from cs336_basics.tokenizer import Tokenizer

SPECIAL_TOKEN = "<|endoftext|>"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    return parser.parse_args()


def sample_documents(path: Path, n: int = 10) -> list[str]:
    """Reservoir-sample documents split on <|endoftext|> without loading the full file."""
    random.seed(0)
    samples: list[str] = []
    seen = 0
    buffer = ""
    with open(path, encoding="utf-8") as f:
        while True:
            chunk = f.read(2_000_000)
            if not chunk:
                break
            buffer += chunk
            while SPECIAL_TOKEN in buffer:
                doc, buffer = buffer.split(SPECIAL_TOKEN, 1)
                if not doc.strip():
                    continue
                seen += 1
                if len(samples) < n:
                    samples.append(doc)
                else:
                    replace_at = random.randint(0, seen - 1)
                    if replace_at < n:
                        samples[replace_at] = doc
        if buffer.strip():
            seen += 1
            if len(samples) < n:
                samples.append(buffer)
            else:
                replace_at = random.randint(0, seen - 1)
                if replace_at < n:
                    samples[replace_at] = buffer
    return samples


def compression_ratio(tokenizer: Tokenizer, text: str) -> float:
    num_bytes = len(text.encode("utf-8"))
    num_tokens = len(tokenizer.encode(text))
    return num_bytes / max(num_tokens, 1)


def throughput_bytes_per_sec(tokenizer: Tokenizer, text: str, repeats: int = 3) -> float:
    payload = text * 20
    nbytes = len(payload.encode("utf-8"))
    start = time.perf_counter()
    for _ in range(repeats):
        tokenizer.encode(payload)
    elapsed = time.perf_counter() - start
    return nbytes * repeats / elapsed


def main() -> None:
    args = parse_args()
    artifacts = args.artifacts_dir
    ts_tok = Tokenizer.from_files(
        artifacts / "bpe/tinystories_10k/vocab.json",
        artifacts / "bpe/tinystories_10k/merges.txt",
        special_tokens=[SPECIAL_TOKEN],
    )
    owt_tok = Tokenizer.from_files(
        artifacts / "bpe/owt_32k/vocab.json",
        artifacts / "bpe/owt_32k/merges.txt",
        special_tokens=[SPECIAL_TOKEN],
    )
    ts_docs = sample_documents(args.data_dir / "TinyStoriesV2-GPT4-train.txt")
    owt_docs = sample_documents(args.data_dir / "owt_train.txt")
    results = {
        "compression_tinystories_docs": {
            "tinystories_tokenizer": [compression_ratio(ts_tok, doc) for doc in ts_docs],
            "owt_tokenizer": [compression_ratio(owt_tok, doc) for doc in ts_docs],
        },
        "compression_owt_docs": {
            "tinystories_tokenizer": [compression_ratio(ts_tok, doc) for doc in owt_docs],
            "owt_tokenizer": [compression_ratio(owt_tok, doc) for doc in owt_docs],
        },
        "avg_compression_tinystories_docs": {},
        "avg_compression_owt_docs": {},
        "owt_with_tinystories_tokenizer_note": (
            "OWT web text is more heterogeneous than TinyStories children's fiction; "
            "the 10K TinyStories tokenizer typically yields lower compression (more bytes/token) on OWT."
        ),
    }
    for split in ("compression_tinystories_docs", "compression_owt_docs"):
        results[f"avg_{split}"] = {
            key: sum(values) / len(values) for key, values in results[split].items()
        }
    bench_text = "\n".join(ts_docs[:3])
    throughput = throughput_bytes_per_sec(ts_tok, bench_text)
    pile_gb = 825
    pile_hours = pile_gb * (1024**3) / throughput / 3600
    results["throughput_bytes_per_sec"] = throughput
    results["pile_825gb_hours_estimate"] = pile_hours
    out_path = artifacts / "bpe_experiments.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
