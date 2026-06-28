# CS336 Spring 2026 Assignment 5: Alignment

> **Portfolio status:** Complete (self-study). See [`PROJECT_STATUS.md`](./PROJECT_STATUS.md), [`SOLUTION.md`](./SOLUTION.md), [`LEARNINGS.md`](./LEARNINGS.md). Verify: `bash scripts/finalize_assignment.sh` → **26/26 tests**.

For a full description of the assignment, see the assignment handout at
[cs336_spring2026_assignment5_alignment.pdf](./cs336_spring2026_assignment5_alignment.pdf)

We will include a supplemental (and completely optional) assignment on safety alignment, instruction tuning, and RLHF at [cs336_spring2026_assignment5_supplement_safety_rlhf.pdf](./cs336_spring2026_assignment5_supplement_safety_rlhf.pdf)

If you see any issues with the assignment handout or code, please feel free to
raise a GitHub issue or open a pull request with a fix.

## Setup

As in previous assignments, we use `uv` to manage dependencies.

1. Install all packages except `flash-attn`, then all packages (`flash-attn` is weird)
```
uv sync --no-install-package flash-attn
uv sync
```

2. Run all unit tests:

``` sh
bash scripts/finalize_assignment.sh
# or: uv run pytest tests/
```

Implementation lives in [`cs336_alignment/grpo.py`](./cs336_alignment/grpo.py) and [`tests/adapters.py`](./tests/adapters.py).

