# CS336 Assignment 1 — Experiment Log

This log tracks all training runs and experiments for Assignment 1.

## Environment

- Machine: MacBook Pro (Apple Silicon, MPS)
- Python: 3.12 via `uv`
- Repo: `assignment1-basics-main`
- Last sync: 2026-06-25T17:13:53.377154+00:00

## §2.5 BPE Training

| Run | Vocab | Input | Time | Merges | Longest token |
|-----|-------|-------|------|--------|---------------|
| `tinystories_10k` | 10,000 | `TinyStoriesV2-GPT4-train.txt` | 10.3 min | 9743 |  accomplishment |
| `owt_32k` | 32,000 | `owt_train_2gb.txt` (Mac subset) | pending | — | — |

## §2.7 Tokenizer Experiments

Results: `artifacts/bpe_experiments.json` (pending)

## §7.2 TinyStories LM Training

| Run | Steps | Batch | Val loss | Train loss | Tokens seen | Time |
|-----|-------|-------|----------|------------|-------------|------|
| `tinystories_main` | 2500 | 32 | 2.0679 | 2.0726 | 20480000 | 36.1 min |

Tokenization: 541,229,509 train tokens in 33.6 min (`uint16` memmap).

Target on B200: 327M tokens, val loss ≤ 2.00. Mac fast path uses ~20M tokens at 2500 steps.

## §7.4 OpenWebText

| Run | Steps | Batch | Val loss | Status |
|-----|-------|-------|----------|--------|
| `owt_main` | 2500 | 32 | — | pending |

## §7.5 Leaderboard Modification

Skipped on Mac fast path (optional). Script: `scripts/continue_fast.sh` / full `run_all.sh`.

## Submission

```bash
uv run pytest -q
uv run python scripts/finalize_assignment.py
bash make_submission.sh
```

Zip: `cs336-spring2025-assignment-1-submission.zip`
