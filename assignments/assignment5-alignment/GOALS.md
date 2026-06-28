# Assignment 5 — Goal & Workflow

## Goal

Implement **alignment / reasoning RL** with GRPO on GSM8K-style math rollouts, plus optional SFT packing, DPO, and eval parsers.

## Required (Gradescope)

| Component | Module |
|-----------|--------|
| Prompt/output tokenization + response mask | `cs336_alignment/grpo.py` |
| Per-token log-probs + entropy | `cs336_alignment/grpo.py` |
| Rollout rewards + group advantages | `cs336_alignment/grpo.py` |
| Policy loss (on/off-policy, GRPO/GSPO clip) | `cs336_alignment/grpo.py` |
| Loss aggregation + train step | `cs336_alignment/grpo.py` |

**Tests:** `tests/test_grpo.py` — **19/19 pass** (numpy snapshots).

## Optional supplement

| Component | Tests |
|-----------|-------|
| Packed SFT dataset + batch iterator | `tests/test_data.py` — pass |
| MMLU / GSM8K parsers | `tests/test_metrics.py` — pass |
| Per-instance DPO loss | `tests/test_dpo.py` — pass |

## Local workflow

```bash
cd assignments/assignment5-alignment
uv sync --no-install-package flash-attn
uv sync
bash scripts/finalize_assignment.sh
```

## Enrolled-only (not done here)

- Modal 8×B200 GRPO training on GSM8K
- Gradescope zip submission
