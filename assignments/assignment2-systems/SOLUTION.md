# CS336 Assignment 2 — Our Solution

**Author:** Mangesh Raut  
**Course:** Stanford CS336 — Language Modeling from Scratch (self-study)  
**Base repo:** [stanford-cs336/assignment2-systems](https://github.com/stanford-cs336/assignment2-systems)

This folder is our Assignment 2 solution: distributed training and flash attention in `cs336_systems/`, wired through `tests/adapters.py`.

---

## What we built

| Component | Location | Status |
|-----------|----------|--------|
| Flash Attention 2 (PyTorch) | `cs336_systems/flash_attention.py` | Done |
| Flash Attention (Triton) | `cs336_systems/flash_attention_triton.py` | Stub on Mac |
| Distributed Data Parallel | `cs336_systems/ddp.py` | Done |
| FSDP (Linear + Embedding) | `cs336_systems/fsdp.py` | Done |
| Sharded AdamW (ZeRO-style) | `cs336_systems/sharded_optimizer.py` | Done |

---

## Verify

```bash
cd assignments/assignment2-systems
uv run pytest -q    # 10 passed, 4 skipped on Mac (no CUDA/Triton)
```

Skipped tests require CUDA and Triton (Linux).

---

## Key design choices

**FSDP:** Staff `Linear` / `Embedding` weights are sharded along dim 0. Forward all-gathers full weights under `no_grad`, then uses a detached leaf tensor with `register_hook` to all-reduce and shard gradients after backward. This keeps standard autograd for activations while manually routing parameter grads to shards.

**DDP:** Broadcast all parameters from rank 0 on init; async `all_reduce` per parameter during backward; `finish_gradient_synchronization()` waits and averages.

**Sharded optimizer:** Each rank runs AdamW on its local weight shard; `step()` broadcasts updated shards from the owning rank pattern (ZeRO-1 style).

---

## Docs

- `PROJECT_STATUS.md` — completion tracker
- `LEARNINGS.md` — self-study notes
