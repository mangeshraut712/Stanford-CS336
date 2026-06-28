# CS336 Assignment 1 — Key Learnings (Self-Study)

## BPE & tokenization

- **Parallel pretokenization** matters: splitting on `<|endoftext|>` before regex pretokenization keeps merges document-local and scales on multi-core CPUs.
- **Streaming** (`encode_iterable`, chunked file reads) is required on macOS for large corpora — loading multi-GB files into RAM fails quickly.
- **uint16 memmaps** halve disk and memory bandwidth vs int32 for vocab ≤ 32k.

## Transformer from scratch

- Implementing **RoPE, RMSNorm, SwiGLU, and causal attention** with pytest snapshot tests forces precise shape and numerics understanding.
- **Pre-norm** + gradient clipping stabilizes small-model training without fancy schedulers.

## Training loop

- **Memmap token bins** decouple preprocessing from training — tokenize once, train many times.
- On Apple Silicon, **batch size vs RAM** dominates wall-clock: batch 64 with swap was ~2× slower than batch 32 with headroom.
- **Cosine LR + warmup** gives smooth loss curves; val loss ~2.07 on TinyStories at 20M tokens (2500 steps) is reasonable for a Mac run.

## Systems / ops

- macOS **System Data** bloat can be leaked `code_sign_clone` temp dirs (Chrome/Codex) — hundreds of GB in `var/folders`, invisible in normal folder views.
- Long pipelines need **resumable scripts** (`continue_fast.sh`) and `caffeinate` for overnight runs.

## What we ran (Mac fast path)

| Component | Setting |
|-----------|---------|
| TinyStories BPE | 10k vocab, ~10 min |
| TinyStories train | 2500 steps, batch 32, val loss **2.07** |
| OWT BPE | 32k vocab, **1GB subset** (full 11GB too slow on Mac) |
| OWT train | 2000 steps, val loss **5.38** |
| Leaderboard | Skipped (optional) |

Full metrics: `artifacts/experiment_log.md`, `writeup.md` §9.
