# CS336 Assignments

Monorepo layout for self-study CS336 implementations.  
See root [`PROJECT_PURPOSE.md`](../PROJECT_PURPOSE.md) and [`scripts/verify_all.sh`](../scripts/verify_all.sh).

| Folder | Course assignment | Status |
|--------|-------------------|--------|
| [`assignment1-basics/`](assignment1-basics/) | Tokenizer, transformer, training | **Complete** |
| [`assignment2-systems/`](assignment2-systems/) | Flash attention, DDP, FSDP | **Complete** |
| [`assignment3-scaling/`](assignment3-scaling/) | Scaling laws, training API | **Complete** (self-study) |

Each assignment is a standalone `uv` project with its own `pyproject.toml` and tests.

## Quick verify

```bash
cd .. && bash scripts/verify_all.sh
```
