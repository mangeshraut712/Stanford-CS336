# CS336 Assignments

Monorepo layout for self-study CS336 implementations.  
See root [`PROJECT_PURPOSE.md`](../PROJECT_PURPOSE.md), [`MONOREPO.md`](../MONOREPO.md), and [`scripts/verify_all.sh`](../scripts/verify_all.sh).

| Folder | Course assignment | Status | Verify |
|--------|-------------------|--------|--------|
| [`assignment1-basics/`](assignment1-basics/) | Tokenizer, transformer, training | **Complete** | `pytest` + `verify_complete.sh` |
| [`assignment2-systems/`](assignment2-systems/) | Flash attention, DDP, FSDP | **Complete** | `uv run pytest -q` |
| [`assignment3-scaling/`](assignment3-scaling/) | Scaling laws, training API | **Complete** | `finalize_assignment.sh` |
| [`assignment4-data/`](assignment4-data/) | Data filtering, deduplication | **Complete** | `finalize_assignment.sh` |
| [`assignment5-alignment/`](assignment5-alignment/) | GRPO, SFT, DPO | **Complete** | `finalize_assignment.sh` |

Each assignment is a standalone `uv` project with its own `pyproject.toml` and tests.

## Quick verify

```bash
cd .. && bash scripts/verify_all.sh
```
