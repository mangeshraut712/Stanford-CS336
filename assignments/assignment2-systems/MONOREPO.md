# Assignment 2 in this monorepo

**Status: Complete** (self-study) — see [`PROJECT_STATUS.md`](PROJECT_STATUS.md) and [`SOLUTION.md`](SOLUTION.md).

## What we implemented

| Component | File |
|-----------|------|
| Flash Attention (PyTorch) | `cs336_systems/flash_attention.py` |
| Flash Attention (Triton stub) | `cs336_systems/flash_attention_triton.py` |
| DDP | `cs336_systems/ddp.py` |
| FSDP | `cs336_systems/fsdp.py` |
| Sharded AdamW | `cs336_systems/sharded_optimizer.py` |

**Tests:** 10 passed, 4 skipped (Triton/CUDA on Mac).

## Assignment 1 wiring

Staff ships `cs336-basics/` with model classes for FSDP. Our full A1 implementation lives in `../assignment1-basics/cs336_basics/`. For monorepo work, keep staff `cs336-basics/pyproject.toml` and use staff model stubs in `cs336-basics/cs336_basics/` (FSDP tests require staff `Linear`/`Embedding`).

```bash
cd assignments/assignment2-systems
uv sync
uv run pytest -q
```

Handout: [`cs336_assignment2_systems.pdf`](cs336_assignment2_systems.pdf)
