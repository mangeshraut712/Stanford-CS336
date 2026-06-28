# Stanford CS336 — Language Modeling from Scratch

Self-study portfolio for [Stanford CS336](https://cs336.stanford.edu/).  
**Author:** [Mangesh Raut](https://github.com/mangeshraut712) · **Repo:** https://github.com/mangeshraut712/Stanford-CS336

Complete implementations for all five assignments: tokenizer & transformer training, systems optimizations, scaling laws, data pipelines, and alignment (GRPO / DPO).

See [`PROJECT_PURPOSE.md`](PROJECT_PURPOSE.md) and [`MONOREPO.md`](MONOREPO.md) for layout and verification.

## Status at a glance

| # | Topic | Tests | Highlights |
|---|-------|-------|------------|
| 1 | Basics | 47 pass, 1 xfail | TS val **2.07**, OWT val **5.38** |
| 2 | Systems | 10 pass, 4 skip | Flash attn, DDP, FSDP, sharded AdamW |
| 3 | Scaling | 7/7 + artifacts | Chinchilla fits, predicted **L≈7.17** |
| 4 | Data | 21/21 | Extract, LID, PII, classifiers, dedup |
| 5 | Alignment | 26/26 | GRPO, SFT packing, DPO, eval parsers |

## Verify all assignments

```bash
bash scripts/verify_all.sh
```

## Assignment quick links

| Assignment | Path | Verify |
|------------|------|--------|
| [1 — Basics](assignments/assignment1-basics/) | BPE, transformer, training | `uv run pytest -q && bash scripts/verify_complete.sh` |
| [2 — Systems](assignments/assignment2-systems/) | Flash attention, distributed | `uv run pytest -q` |
| [3 — Scaling](assignments/assignment3-scaling/) | Isoflops, scaling laws | `bash scripts/finalize_assignment.sh` |
| [4 — Data](assignments/assignment4-data/) | Filtering, deduplication | `bash scripts/finalize_assignment.sh` |
| [5 — Alignment](assignments/assignment5-alignment/) | GRPO, SFT, DPO | `bash scripts/finalize_assignment.sh` |

### Documentation per assignment

Each folder includes `PROJECT_STATUS.md`, `SOLUTION.md`, and `LEARNINGS.md` where applicable, plus `writeup.md` and `GOALS.md` for A3–A5.

## What belongs on GitHub

| In git | Local only |
|--------|------------|
| Source, tests, scripts, small artifacts | Token `.bin` memmaps, checkpoints |
| Docs, scaling fits, run summaries | `local-shared-data/` (~2GB classifiers for A4) |
| A5 staff eval fixtures (~85MB) | `.venv/`, Gradescope zips |

## Course links

- [CS336 course page](https://cs336.stanford.edu)
- [Spring 2025 lectures](https://github.com/stanford-cs336/spring2025-lectures)
- [Staff assignment repos](https://github.com/stanford-cs336)

## Note on self-study scope

All **local unit tests** pass. Not included: Gradescope submissions, Modal GPU training (A4 full WET pipeline, A5 GSM8K GRPO), or leaderboard entries requiring CUDA/B200.
