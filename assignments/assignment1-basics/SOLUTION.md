# CS336 Assignment 1 — Our Solution

**Author:** Mangesh Raut  
**Course:** Stanford CS336 — Language Modeling from Scratch (self-study)  
**Base repo:** [stanford-cs336/assignment1-basics](https://github.com/stanford-cs336/assignment1-basics)

This folder **is** our Assignment 1 solution: original implementation wired through `tests/adapters.py`, not copied from staff or student repos.

---

## What we built

| Component | Location | Status |
|-----------|----------|--------|
| BPE training (parallel pretokenization) | `cs336_basics/tokenizer.py` | Done |
| Tokenizer encode/decode/iterable | `cs336_basics/tokenizer.py` | Done |
| Transformer primitives (RoPE, RMSNorm, SwiGLU, attention) | `cs336_basics/model.py` | Done |
| Full LM + ablations | `cs336_basics/train_model.py` | Done |
| AdamW + cosine LR | `cs336_basics/optimizer.py` | Done |
| Memmap batches + checkpointing | `cs336_basics/data.py`, `checkpoint.py` | Done |
| End-to-end training + generation | `train.py` | Done |
| Experiment pipeline | `scripts/` | Done |

---

## Verify the solution

```bash
cd assignments/assignment1-basics
uv run pytest -q          # expect: 47 passed, 1 xfailed
```

The xfail is intentional: `test_encode_memory_usage` documents a non-streaming encode path.

---

## Results (Mac, Apple Silicon)

| Run | Metric |
|-----|--------|
| TinyStories BPE 10k | 10.3 min, longest token `" accomplishment"` |
| TinyStories tokenize (train) | 541M tokens, ~34 min |
| `tinystories_main` (2500 steps, batch 32) | **val loss 2.07**, ~36 min |
| OWT BPE 32k (1GB subset) | ~159 min |
| `owt_main` (2000 steps, batch 32) | **val loss 5.38**, ~7.7 hr wall (incl. sleep) |

Full tables: `artifacts/experiment_log.md`, `writeup.md` §9.

---

## Solution layout

```
cs336_basics/          # Core library (your implementation)
tests/adapters.py      # Bridges tests → cs336_basics
train.py               # Training loop
scripts/               # BPE, tokenize, experiments, pipelines
writeup.md             # Learning writeup (Markdown + PDF)
PROJECT_STATUS.md      # Completion tracker
LEARNINGS.md           # Self-study notes
artifacts/             # Tokenizers, metrics, samples (see artifacts/README.md)
```

---

## Sync docs

```bash
bash scripts/finalize_assignment.sh   # pytest + metrics + writeup.pdf
```

Status: `PROJECT_STATUS.md`

---

## Use with Assignment 2

When starting [assignment2-systems](https://github.com/stanford-cs336/assignment2-systems) in this monorepo:

```bash
cd assignments/assignment2-systems
# Copy or symlink your A1 package — see assignment2 README
cp -R ../assignment1-basics/cs336_basics ./cs336_basics
```

Assignment 2 ships a staff `cs336-basics` reference — **replace it with this folder** to use your own A1 solution.

---

## What’s optional / still running

- **OWT full corpus** — we use a **1GB** subset on Mac (`data/owt_train_1gb.txt`)
- **Leaderboard mod** — skipped on fast path; optional for full handout
- **327M tokens on B200** — course target; we used ~20M tokens on Mac

These do not block Assignment 2 or treating Assignment 1 as a complete learning project.
