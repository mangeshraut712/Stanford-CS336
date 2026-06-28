# Stanford CS336 — Language Modeling from Scratch

Self-study portfolio for [Stanford CS336](https://cs336.stanford.edu/).  
**Author:** [Mangesh Raut](https://github.com/mangeshraut712) · **Repo:** https://github.com/mangeshraut712/Stanford-CS336

See [`PROJECT_PURPOSE.md`](PROJECT_PURPOSE.md) for goals, layout, and verification.

## Assignments

| # | Name | Path | Status |
|---|------|------|--------|
| 1 | [Basics](https://github.com/stanford-cs336/assignment1-basics) | [`assignments/assignment1-basics/`](assignments/assignment1-basics/) | **Complete** — TS val 2.07, OWT val 5.38 |
| 2 | [Systems](https://github.com/stanford-cs336/assignment2-systems) | [`assignments/assignment2-systems/`](assignments/assignment2-systems/) | **Complete** — 10 passed, 4 skipped (Mac) |
| 3 | [Scaling](https://github.com/stanford-cs336/assignment3-scaling) | [`assignments/assignment3-scaling/`](assignments/assignment3-scaling/) | **Complete** — scaling fits, prediction L≈7.17 |
| 4 | [Data](https://github.com/stanford-cs336/assignment4-data) | [`assignments/assignment4-data/`](assignments/assignment4-data/) | **Complete** — 21/21 tests, data pipeline |
| 5 | [Alignment](https://github.com/stanford-cs336/assignment5-alignment) | [`assignments/assignment5-alignment/`](assignments/assignment5-alignment/) | **Complete** — 19/19 GRPO tests |

## Verify all completed work

```bash
bash scripts/verify_all.sh
```

## Assignment 1

```bash
cd assignments/assignment1-basics && uv run pytest -q
```

- 47 passed, 1 xfail · TinyStories + OWT training on Mac
- Docs: `writeup.md`, `writeup.pdf`, `SOLUTION.md`, `LEARNINGS.md`

## Assignment 2

```bash
cd assignments/assignment2-systems && uv run pytest -q
```

- Flash Attention (PyTorch), DDP, FSDP, Sharded AdamW
- Docs: `SOLUTION.md`, `LEARNINGS.md`, `MONOREPO.md`

## Assignment 3

```bash
cd assignments/assignment3-scaling
brew services start postgresql@15   # once
bash scripts/finalize_assignment.sh
```

- 7/7 pytest · isoflops analysis · Chinchilla-style fits · predicted loss 7.169
- Docs: `writeup.md`, `SOLUTION.md`, `GOALS.md`

## Assignment 4

```bash
cd assignments/assignment4-data
bash scripts/finalize_assignment.sh
```

- 21/21 pytest · HTML extract, LID, PII, classifiers, Gopher, dedup
- Docs: `writeup.md`, `SOLUTION.md`, `GOALS.md`
- Offline assets: `uv run python scripts/download_data.py --offline-only` (~2GB, first run)

## Assignment 5

```bash
cd assignments/assignment5-alignment
bash scripts/finalize_assignment.sh
```

- 19/19 GRPO pytest · tokenization, advantages, PPO/GSPO clip, train step
- Optional: SFT packing, MMLU/GSM8K parsers (pass); DPO optional test pending snapshot alignment
- Docs: `writeup.md`, `SOLUTION.md`, `GOALS.md`

## What belongs on GitHub

| Track in git | Keep local only |
|--------------|-----------------|
| Source, tests, scripts | `data/` corpora, token `.bin` memmaps |
| Docs + small artifacts | Model checkpoints, `.venv/`, `*.zip` |

## Course links

- https://cs336.stanford.edu
- https://github.com/stanford-cs336/spring2025-lectures
