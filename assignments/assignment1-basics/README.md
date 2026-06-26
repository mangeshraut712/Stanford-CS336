# Assignment 1: Basics

**Author:** Mangesh Raut · [Course handout](./cs336_assignment1_basics.pdf) · [Stanford CS336](https://cs336.stanford.edu)

Self-study implementation of byte-level BPE, a GPT-style transformer, AdamW training, and end-to-end LM training on TinyStories and OpenWebText (1GB Mac subset).

**Status: complete** for project purposes — see [PROJECT_STATUS.md](./PROJECT_STATUS.md).

## Quick start

```bash
cd assignments/assignment1-basics
uv sync
uv run pytest -q                    # 47 passed, 1 xfail
bash scripts/finalize_assignment.sh  # sync docs + writeup.pdf
```

Download datasets first — see [DATA.md](./DATA.md).

## Results (Mac, Apple Silicon)

| Run | Val loss | Notes |
|-----|----------|-------|
| `tinystories_main` | **2.07** | 2500 steps, batch 32, ~20M tokens |
| `owt_main` | **5.38** | 2000 steps, 32k vocab, 1GB OWT subset |

## Layout

```
cs336_basics/       # tokenizer, model, optimizer, training utilities
train.py            # memmap training loop + generation
tests/adapters.py   # connects tests → our implementation
scripts/            # BPE, tokenize, experiment pipelines
writeup.md          # learning writeup (Markdown)
writeup.pdf         # exported notes (regenerate via scripts/export_writeup_pdf.sh)
PROJECT_STATUS.md   # completion tracker
SOLUTION.md         # architecture & reuse notes for Assignment 2
LEARNINGS.md        # takeaways
artifacts/          # small tracked results (see artifacts/README.md)
```

## Pipelines

```bash
bash scripts/run_fast_ts.sh    # TinyStories only (~1 hr on Mac)
bash scripts/run_fast.sh       # + OWT 1GB subset (~3–4 hr)
bash scripts/run_all.sh          # full handout (very slow on Mac)
bash scripts/watch_pipeline.sh # check running pipeline
bash scripts/verify_complete.sh # exit 0 when all required artifacts exist
```

## Docs

| File | Purpose |
|------|---------|
| [PROJECT_STATUS.md](./PROJECT_STATUS.md) | What's done / optional |
| [SOLUTION.md](./SOLUTION.md) | Implementation map + Assignment 2 hooks |
| [LEARNINGS.md](./LEARNINGS.md) | Self-study notes |
| [writeup.md](./writeup.md) | Full writeup with metrics |

Based on [stanford-cs336/assignment1-basics](https://github.com/stanford-cs336/assignment1-basics).
