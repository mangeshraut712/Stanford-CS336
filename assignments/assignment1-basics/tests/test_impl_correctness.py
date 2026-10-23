"""Correctness checks that the staff fixtures do not cover."""

from __future__ import annotations

import torch

from cs336_basics.tokenizer import Tokenizer
from cs336_basics.train_model import BasicsTransformerLM, CausalMultiHeadSelfAttention


def _byte_vocab() -> dict[int, bytes]:
    return {index: bytes([index]) for index in range(256)}


def test_bpe_ranks_do_not_merge_a_different_pair_with_the_same_bytes() -> None:
    vocab = _byte_vocab()
    vocab[256] = b"bc"
    vocab[257] = b"ab"
    vocab[258] = b"abc"
    merges = [(b"b", b"c"), (b"a", b"b"), (b"ab", b"c")]
    tokenizer = Tokenizer(vocab, merges)

    # (b, c) is the lowest rank, so "abc" becomes the tokens a + bc.
    # Looking merges up by left+right would also match (a, bc) against the
    # unrelated merge (ab, c).
    assert tokenizer.encode("abc") == [ord("a"), 256]
    assert tokenizer.decode(tokenizer.encode("abc")) == "abc"


def test_encode_iterable_matches_encode_when_chunks_split_a_pretoken() -> None:
    vocab = _byte_vocab()
    vocab[256] = b"ab"
    tokenizer = Tokenizer(vocab, [(b"a", b"b")])
    text = ("a" * 4097) + "b"
    chunks = [text[:4097], text[4097:]]

    assert tokenizer.encode(text)[-1] == 256
    assert list(tokenizer.encode_iterable(chunks)) == tokenizer.encode(text)


def test_encode_iterable_keeps_a_special_token_split_across_chunks() -> None:
    vocab = _byte_vocab()
    vocab[256] = b"<|e|>"
    tokenizer = Tokenizer(vocab, [], special_tokens=["<|e|>"])
    text = ("a" * 4096) + "<|e|>" + "z"
    chunks = [text[:4097], text[4097:]]

    assert list(tokenizer.encode_iterable(chunks)) == tokenizer.encode(text)
    assert 256 in tokenizer.encode(text)


def test_attention_accepts_missing_and_multiple_batch_dims() -> None:
    layer = CausalMultiHeadSelfAttention(d_model=8, num_heads=2)
    unbatched = torch.randn(5, 8)
    assert layer(unbatched).shape == (5, 8)
    nested = torch.randn(2, 3, 5, 8)
    assert layer(nested).shape == (2, 3, 5, 8)


def test_generate_returns_every_new_token_after_the_context_slides() -> None:
    model = BasicsTransformerLM(
        vocab_size=32,
        context_length=4,
        d_model=16,
        num_layers=1,
        num_heads=2,
        d_ff=32,
        rope_theta=10_000.0,
    )
    model.eval()
    prompt = torch.randint(0, 32, (1, 4))
    torch.manual_seed(0)
    generated = model.generate(prompt, max_new_tokens=6, temperature=1.0)
    assert generated.shape == (1, 6)


def test_generate_supports_a_batch_and_greedy_decoding() -> None:
    model = BasicsTransformerLM(
        vocab_size=32,
        context_length=8,
        d_model=16,
        num_layers=1,
        num_heads=2,
        d_ff=32,
        rope_theta=10_000.0,
    )
    model.eval()
    prompt = torch.randint(0, 32, (2, 3))
    first = model.generate(prompt, max_new_tokens=3, temperature=0.0)
    second = model.generate(prompt, max_new_tokens=3, temperature=0.0)
    torch.testing.assert_close(first, second)
    batched = model.generate(prompt, max_new_tokens=4, temperature=1.0, eos_token_id=1)
    assert batched.shape[0] == 2
