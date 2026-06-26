# Assignment 1: Basics

**Author:** Mangesh Raut · [Course handout](./cs336_assignment1_basics.pdf) · [Stanford CS336](https://cs336.stanford.edu)

Our implementation of byte-level BPE, a GPT-style transformer, AdamW training, and end-to-end LM training on TinyStories (+ OWT subset on Mac).

## Quick start

```bash
cd assignments/assignment1-basics
uv run pytest -q                    # 47 passed, 1 xfail
bash scripts/package_solution.sh    # sync docs + build submission zip
```

Download datasets first — see [DATA.md](./DATA.md).

## Status

| Item | Status |
|------|--------|
| Unit tests | 47 passed, 1 xfail |
| TinyStories BPE + train | Done (val loss **2.07**) |
| OWT BPE 32k + tokenization | Done (1GB subset) |
| `owt_main` training | In progress (~2000 steps) — run `bash scripts/watch_pipeline.sh` |
| Writeup | [writeup.md](./writeup.md) — re-run `finalize_assignment.py` after `owt_main` |

## Layout

```
cs336_basics/     # tokenizer, model, optimizer, training utilities
train.py          # memmap training loop + generation
tests/adapters.py # connects tests → our implementation
scripts/          # BPE, tokenize, experiment pipelines
writeup.md        # Gradescope writeup (export to PDF)
SOLUTION.md       # detailed solution notes
LEARNINGS.md      # self-study takeaways
artifacts/        # small results only on GitHub (see artifacts/README.md)
```

## Pipelines

```bash
bash scripts/run_fast_ts.sh    # TinyStories only (~1 hr on Mac)
bash scripts/run_fast.sh       # + OWT 2GB subset (~3–4 hr)
bash scripts/run_all.sh        # full handout (very slow on Mac)
```

## Docs

- [SOLUTION.md](./SOLUTION.md) — what's implemented and how to reuse in Assignment 2
- [ASSIGNMENT_CHECKLIST.md](./ASSIGNMENT_CHECKLIST.md) — completion tracker
- [artifacts/README.md](./artifacts/README.md) — what's tracked vs local-only

Based on [stanford-cs336/assignment1-basics](https://github.com/stanford-cs336/assignment1-basics).
