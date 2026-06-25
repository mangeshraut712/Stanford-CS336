from __future__ import annotations

import json
import os
import regex
from collections import Counter
from multiprocessing import Pool
from typing import BinaryIO, Iterable

from cs336_basics.pretokenization_example import find_chunk_boundaries

GPT2_PAT = r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def _split_on_special_tokens(text: str, special_tokens: list[str]) -> list[str]:
    if not special_tokens:
        return [text]
    ordered = sorted(special_tokens, key=len, reverse=True)
    pattern = "|".join(regex.escape(token) for token in ordered)
    return regex.split(f"({pattern})", text)


def _pretokenize_bytes(text: str) -> list[bytes]:
    return [match.encode("utf-8") for match in regex.findall(GPT2_PAT, text)]


def _merge_pair_in_word(word: tuple[bytes, ...], pair: tuple[bytes, bytes]) -> tuple[bytes, ...]:
    merged: list[bytes] = []
    i = 0
    while i < len(word):
        if i < len(word) - 1 and (word[i], word[i + 1]) == pair:
            merged.append(word[i] + word[i + 1])
            i += 2
        else:
            merged.append(word[i])
            i += 1
    return tuple(merged)


def _add_word_pair_counts(
    pair_counts: Counter[tuple[bytes, bytes]],
    word: tuple[bytes, ...],
    count: int,
) -> None:
    for index in range(len(word) - 1):
        pair_counts[(word[index], word[index + 1])] += count


def _remove_word_pair_counts(
    pair_counts: Counter[tuple[bytes, bytes]],
    word: tuple[bytes, ...],
    count: int,
) -> None:
    for index in range(len(word) - 1):
        pair = (word[index], word[index + 1])
        pair_counts[pair] -= count
        if pair_counts[pair] <= 0:
            del pair_counts[pair]


def _word_has_pair(word: tuple[bytes, ...], pair: tuple[bytes, bytes]) -> bool:
    return any((word[index], word[index + 1]) == pair for index in range(len(word) - 1))


def _count_words_in_text(text: str, special_tokens: list[str]) -> Counter[tuple[bytes, ...]]:
    special_bytes = {token.encode("utf-8") for token in special_tokens}
    word_counts: Counter[tuple[bytes, ...]] = Counter()
    for chunk in _split_on_special_tokens(text, special_tokens):
        if not chunk:
            continue
        if chunk.encode("utf-8") in special_bytes:
            continue
        for pretoken in _pretokenize_bytes(chunk):
            word_counts[tuple(bytes([byte]) for byte in pretoken)] += 1
    return word_counts


def _pretokenize_byte_range(args: tuple[str | os.PathLike, int, int, list[str]]) -> Counter[tuple[bytes, ...]]:
    path, start, end, special_tokens = args
    with open(path, "rb") as file:
        file.seek(start)
        chunk_text = file.read(end - start).decode("utf-8", errors="ignore")
    return _count_words_in_text(chunk_text, special_tokens)


def _pretokenize_file_parallel(input_path: str | os.PathLike, special_tokens: list[str], num_workers: int) -> Counter[tuple[bytes, ...]]:
    split_token = special_tokens[0].encode("utf-8") if special_tokens else b"<|endoftext|>"
    file_size = os.path.getsize(input_path)
    target_chunk_bytes = 256 * 1024 * 1024
    num_chunks = max(num_workers, min(128, max(8, file_size // target_chunk_bytes or 1)))
    with open(input_path, "rb") as file:
        boundaries = find_chunk_boundaries(file, num_chunks, split_token)
    ranges = [
        (input_path, start, end, special_tokens)
        for start, end in zip(boundaries[:-1], boundaries[1:])
    ]
    if num_workers <= 1 or len(ranges) <= 1:
        merged: Counter[tuple[bytes, ...]] = Counter()
        for args in ranges:
            merged.update(_pretokenize_byte_range(args))
        return merged
    merged = Counter()
    with Pool(processes=num_workers) as pool:
        for partial in pool.imap_unordered(_pretokenize_byte_range, ranges, chunksize=1):
            merged.update(partial)
    return merged


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    num_workers: int = 1,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    word_counts = _pretokenize_file_parallel(input_path, special_tokens, num_workers)

    pair_counts: Counter[tuple[bytes, bytes]] = Counter()
    for word, count in word_counts.items():
        _add_word_pair_counts(pair_counts, word, count)

    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    for token in special_tokens:
        vocab[len(vocab)] = token.encode("utf-8")

    merges: list[tuple[bytes, bytes]] = []
    while len(vocab) < vocab_size and pair_counts:
        best_pair = max(pair_counts, key=lambda pair: (pair_counts[pair], pair))
        new_token = best_pair[0] + best_pair[1]
        vocab[len(vocab)] = new_token
        merges.append(best_pair)

        words_to_update = [word for word in word_counts if _word_has_pair(word, best_pair)]
        for word in words_to_update:
            count = word_counts.pop(word)
            _remove_word_pair_counts(pair_counts, word, count)
            new_word = _merge_pair_in_word(word, best_pair)
            word_counts[new_word] += count
            _add_word_pair_counts(pair_counts, new_word, count)

    return vocab, merges


class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or []
        self.bytes_to_id = {token_bytes: token_id for token_id, token_bytes in vocab.items()}
        self.id_to_bytes = dict(vocab)
        self.bpe_ranks = {left + right: i for i, (left, right) in enumerate(merges)}
        self._special_bytes = {token.encode("utf-8") for token in self.special_tokens}

    def _bpe_encode_bytes(self, piece: bytes) -> list[int]:
        parts = [bytes([b]) for b in piece]
        while len(parts) >= 2:
            best_rank: int | None = None
            best_index: int | None = None
            for i in range(len(parts) - 1):
                pair = parts[i] + parts[i + 1]
                rank = self.bpe_ranks.get(pair)
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank = rank
                    best_index = i
            if best_index is None:
                break
            merged = parts[best_index] + parts[best_index + 1]
            parts = parts[:best_index] + [merged] + parts[best_index + 2 :]
        return [self.bytes_to_id[part] for part in parts]

    def encode(self, text: str) -> list[int]:
        ids: list[int] = []
        for chunk in _split_on_special_tokens(text, self.special_tokens):
            if not chunk:
                continue
            chunk_bytes = chunk.encode("utf-8")
            if chunk_bytes in self._special_bytes:
                ids.append(self.bytes_to_id[chunk_bytes])
                continue
            for pretoken in _pretokenize_bytes(chunk):
                ids.extend(self._bpe_encode_bytes(pretoken))
        return ids

    def decode(self, ids: list[int]) -> str:
        return b"".join(self.id_to_bytes[token_id] for token_id in ids).decode("utf-8", errors="replace")

    def encode_iterable(self, iterable: Iterable[str]) -> Iterable[int]:
        buffer = ""
        for chunk in iterable:
            buffer += chunk
            while buffer:
                if self.special_tokens:
                    earliest_index: int | None = None
                    earliest_token: str | None = None
                    for token in sorted(self.special_tokens, key=len, reverse=True):
                        index = buffer.find(token)
                        if index != -1 and (earliest_index is None or index < earliest_index):
                            earliest_index = index
                            earliest_token = token
                    if earliest_index is not None and earliest_index > 0:
                        prefix = buffer[:earliest_index]
                        buffer = buffer[earliest_index:]
                        for token_id in self.encode(prefix):
                            yield token_id
                        continue
                    if earliest_index == 0 and earliest_token is not None:
                        buffer = buffer[len(earliest_token) :]
                        yield self.bytes_to_id[earliest_token.encode("utf-8")]
                        continue

                if "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    line += "\n"
                    for token_id in self.encode(line):
                        yield token_id
                    continue

                if len(buffer) > 4096:
                    for token_id in self.encode(buffer):
                        yield token_id
                    buffer = ""
                    break
                break

        if buffer:
            for token_id in self.encode(buffer):
                yield token_id

    def save(self, output_dir: str | os.PathLike) -> None:
        output_dir = os.fspath(output_dir)
        os.makedirs(output_dir, exist_ok=True)
        vocab_json = {str(token_id): list(token_bytes) for token_id, token_bytes in self.vocab.items()}
        with open(os.path.join(output_dir, "vocab.json"), "w") as f:
            json.dump(vocab_json, f)
        with open(os.path.join(output_dir, "merges.txt"), "w") as f:
            for left, right in self.merges:
                f.write(f"{left.hex()} {right.hex()}\n")
        if self.special_tokens:
            with open(os.path.join(output_dir, "special_tokens.json"), "w") as f:
                json.dump(self.special_tokens, f)

    @classmethod
    def from_files(cls, vocab_path: str | os.PathLike, merges_path: str | os.PathLike, special_tokens: list[str] | None = None):
        with open(vocab_path) as f:
            raw_vocab = json.load(f)
        vocab = {int(token_id): bytes(values) for token_id, values in raw_vocab.items()}
        merges: list[tuple[bytes, bytes]] = []
        with open(merges_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                left_hex, right_hex = line.split()
                merges.append((bytes.fromhex(left_hex), bytes.fromhex(right_hex)))
        return cls(vocab=vocab, merges=merges, special_tokens=special_tokens)
