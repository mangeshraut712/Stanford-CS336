"""Exact and MinHash document deduplication."""

from __future__ import annotations

import os
from collections import defaultdict
from pathlib import Path

import mmh3


def exact_line_deduplication(
    input_files: list[os.PathLike], output_directory: os.PathLike
) -> None:
    output_dir = Path(output_directory)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = sorted([Path(p) for p in input_files], key=lambda p: p.name)

    file_lines: list[list[str]] = []
    line_file_counts: dict[str, int] = {}

    for path in paths:
        with open(path) as f:
            lines = f.readlines()
        file_lines.append(lines)
        for line in set(lines):
            line_file_counts[line] = line_file_counts.get(line, 0) + 1

    for path, lines in zip(paths, file_lines):
        kept = [line for line in lines if line_file_counts[line] == 1]
        with open(output_dir / path.name, "w") as f:
            f.writelines(kept)


def _word_shingles(text: str, n: int) -> set[str]:
    words = text.split()
    if len(words) < n:
        return {" ".join(words)} if words else set()
    return {" ".join(words[i : i + n]) for i in range(len(words) - n + 1)}


def _minhash_signature(shingles: set[str], num_hashes: int) -> list[int]:
    if not shingles:
        return [2**32 - 1] * num_hashes
    return [
        min(mmh3.hash(shingle, seed, signed=False) for shingle in shingles)
        for seed in range(num_hashes)
    ]


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    union = len(left | right)
    if union == 0:
        return 0.0
    return len(left & right) / union


def _band_ranges(num_hashes: int, num_bands: int) -> list[tuple[int, int]]:
    if num_bands <= 0 or num_hashes <= 0:
        raise ValueError("num_hashes and num_bands must be positive")
    if num_hashes < num_bands:
        raise ValueError("num_hashes must be at least num_bands")
    base, extra = divmod(num_hashes, num_bands)
    ranges: list[tuple[int, int]] = []
    start = 0
    for band_idx in range(num_bands):
        width = base + (1 if band_idx < extra else 0)
        ranges.append((start, start + width))
        start += width
    return ranges


class _UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def minhash_deduplication(
    input_files: list[os.PathLike],
    num_hashes: int,
    num_bands: int,
    ngrams: int,
    jaccard_threshold: float,
    output_directory: os.PathLike,
) -> None:
    output_dir = Path(output_directory)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = sorted([Path(p) for p in input_files], key=lambda p: p.name)
    texts = [p.read_text() for p in paths]
    shingle_sets = [_word_shingles(text, ngrams) for text in texts]
    signatures = [_minhash_signature(shingles, num_hashes) for shingles in shingle_sets]

    uf = _UnionFind(len(paths))
    buckets: dict[tuple[int, tuple[int, ...]], list[int]] = defaultdict(list)

    for doc_idx, sig in enumerate(signatures):
        for band_idx, (start, stop) in enumerate(_band_ranges(num_hashes, num_bands)):
            band = tuple(sig[start:stop])
            key = (band_idx, band)
            for other_idx in buckets[key]:
                if _jaccard(shingle_sets[doc_idx], shingle_sets[other_idx]) >= jaccard_threshold:
                    uf.union(doc_idx, other_idx)
            buckets[key].append(doc_idx)

    groups: dict[int, list[int]] = defaultdict(list)
    for idx in range(len(paths)):
        groups[uf.find(idx)].append(idx)

    keep_indices = {min(members) for members in groups.values()}

    for idx, path in enumerate(paths):
        if idx in keep_indices:
            (output_dir / path.name).write_text(texts[idx])
