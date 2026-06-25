# Stanford CS336 — Language Modeling from Scratch

Self-study solutions for [Stanford CS336](https://cs336.stanford.edu/spring2025/).

**Author:** [Mangesh Raut](https://github.com/mangeshraut712)

## Assignments

| # | Name | Path | Status |
|---|------|------|--------|
| 1 | [Basics](https://github.com/stanford-cs336/assignment1-basics) | [`assignments/assignment1-basics/`](assignments/assignment1-basics/) | **Code + tests done** · TS val loss 2.07 · OWT pending |
| 2 | [Systems](https://github.com/stanford-cs336/assignment2-systems) | — | Not started |
| 3 | [Scaling](https://github.com/stanford-cs336/assignment3-scaling) | — | Not started |
| 4 | [Data](https://github.com/stanford-cs336/assignment4-data) | — | Not started |
| 5 | [Alignment](https://github.com/stanford-cs336/assignment5-alignment) | — | Not started |

## Assignment 1 quick start

```bash
cd assignments/assignment1-basics
uv run pytest -q
```

See [`assignments/assignment1-basics/README.md`](assignments/assignment1-basics/README.md) for data download, training pipelines, and writeup.

## What belongs on GitHub

- Source code (`cs336_basics/`, `train.py`, `tests/`, `scripts/`)
- Docs (`writeup.md`, `SOLUTION.md`, `LEARNINGS.md`)
- Small artifacts (BPE vocab JSON, `run_summary.json`, `*.meta.json`, generated samples)
- **Not** in git: `data/` corpora, token `.bin` files, model checkpoints, `.venv/`, submission `.zip`

## Course links

- https://cs336.stanford.edu
- https://github.com/stanford-cs336/spring2025-lectures
