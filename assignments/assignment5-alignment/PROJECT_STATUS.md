# Assignment 5 — Project Status

**Status:** Complete (self-study)  
**Last verified:** 2026-06-28

## Test results

| Suite | Result |
|-------|--------|
| `tests/test_grpo.py` | 19/19 pass |
| `tests/test_data.py` | 2/2 pass |
| `tests/test_metrics.py` | 4/4 pass |
| `tests/test_dpo.py` | 1/1 pass |
| **Total** | **26/26 pass** |

## Implemented

- `cs336_alignment/grpo.py` — full GRPO stack
- `tests/adapters.py` — wired to staff tests
- `cs336_alignment/modal_utils.py` — `SUNET_ID = "00000001"` for self-study Modal jobs

## Not implemented (optional / enrolled)

- Modal GRPO training run on GSM8K
- AlpacaEval / safety eval pipelines
- DPO training loop (full HH training not run; per-instance loss test passes)

## Verify

```bash
bash scripts/finalize_assignment.sh
```
