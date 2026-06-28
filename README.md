# Stanford CS336 — Language Modeling from Scratch

Self-study / portfolio implementations for [Stanford CS336](https://cs336.stanford.edu/spring2025/).  
**Not** enrolled for course credit — learning and building in public on GitHub.

**Author:** [Mangesh Raut](https://github.com/mangeshraut712) · **Repo:** https://github.com/mangeshraut712/Stanford-CS336

## Assignments

| # | Name | Path | Status |
|---|------|------|--------|
| 1 | [Basics](https://github.com/stanford-cs336/assignment1-basics) | [`assignments/assignment1-basics/`](assignments/assignment1-basics/) | **Complete** — TS val 2.07, OWT val 5.38 |
| 2 | [Systems](https://github.com/stanford-cs336/assignment2-systems) | [`assignments/assignment2-systems/`](assignments/assignment2-systems/) | **Complete** — 10/10 tests pass on Mac |
| 3 | [Scaling](https://github.com/stanford-cs336/assignment3-scaling) | [`assignments/assignment3-scaling/`](assignments/assignment3-scaling/) | **Scaffolded** — API experiments + writeup |
| 4 | [Data](https://github.com/stanford-cs336/assignment4-data) | — | Not started |
| 5 | [Alignment](https://github.com/stanford-cs336/assignment5-alignment) | — | Not started |

## Assignment 1 (done)

```bash
cd assignments/assignment1-basics
uv run pytest -q
```

- All unit tests pass (47 + 1 intentional xfail)
- TinyStories + OWT pipelines run on Mac (fast path)
- Docs: `writeup.md`, `writeup.pdf`, `SOLUTION.md`, `LEARNINGS.md`
- **No submission zips** — source and small artifacts live in git

See [`assignments/assignment1-basics/PROJECT_STATUS.md`](assignments/assignment1-basics/PROJECT_STATUS.md).

## Assignment 2 (done)

```bash
cd assignments/assignment2-systems
uv run pytest -q
```

- 10 passed, 4 skipped (Triton/CUDA on Mac)
- Flash Attention (PyTorch), DDP, FSDP, Sharded AdamW
- Docs: `SOLUTION.md`, `LEARNINGS.md`, `PROJECT_STATUS.md`

See [`assignments/assignment2-systems/PROJECT_STATUS.md`](assignments/assignment2-systems/PROJECT_STATUS.md).

## Assignment 3 (scaffolded)

```bash
cd assignments/assignment3-scaling
uv sync --extra server
export A3_API_KEY=06123456   # 8-digit key for hosted API
```

- Staff-provided API client + JAX training stack
- **Your work:** submit runs on [hyperturing](http://hyperturing.stanford.edu:8000), fit scaling laws, writeup
- Start with `examples/client_example.ipynb`

See [`assignments/assignment3-scaling/PROJECT_STATUS.md`](assignments/assignment3-scaling/PROJECT_STATUS.md).

## Assignment 4+ (not started)

## What belongs on GitHub

| Track in git | Keep local only |
|--------------|-----------------|
| Source (`cs336_basics/`, `train.py`, `tests/`, `scripts/`) | `data/` corpora |
| Docs (`writeup.md`, `writeup.pdf`, `SOLUTION.md`, …) | Token `.bin` memmaps |
| Small artifacts (BPE JSON, `run_summary.json`, `*.meta.json`) | Model checkpoints (`.pt`) |
| | `.venv/`, `*.zip`, pipeline logs |

## Course links

- https://cs336.stanford.edu
- https://github.com/stanford-cs336/spring2025-lectures
