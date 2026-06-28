# Assignment 2 — Project Status

**Purpose:** Self-study / portfolio (learning project).  
**Updated:** 2026-06-28

## Tests (Mac CPU)

```bash
cd assignments/assignment2-systems
uv run pytest -q
# 10 passed, 4 skipped (Triton/CUDA)
```

| Component | Status |
|-----------|--------|
| Flash Attention (PyTorch) | Done |
| Flash Attention (Triton) | Stub (CUDA only; skipped on Mac) |
| DDP (async per-param all-reduce) | Done |
| FSDP (shard Linear/Embedding + gather hooks) | Done |
| Sharded AdamW optimizer | Done |
| Staff `cs336_basics` (A1 model classes) | Wired via `cs336-basics/` |

## Layout

```
cs336_systems/
  flash_attention.py          # PyTorch FA2 autograd
  flash_attention_triton.py     # delegates to PyTorch on Mac
  ddp.py
  fsdp.py                       # hook-based weight gather + grad shard
  sharded_optimizer.py
tests/adapters.py               # wires tests → cs336_systems
```

## Optional next steps

1. Triton FA kernels on CUDA / Linux
2. Profiling & benchmarking sections from handout
3. Leaderboard forward+backward task (B200 / CUDA)

## Run

```bash
uv sync
uv run pytest -q
```
