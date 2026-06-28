# Assignment 3 — Goal & Local vs Hosted Workflow

## Goal of Assignment 3 (Scaling)

Learn **how language model performance scales** with:

- **Model size** (parameters \(N\))
- **Compute / data** (training FLOPs or tokens \(D\))

You use real (or reference) training runs to:

1. **Map isoflops curves** — at fixed compute \(C\), loss vs \(N\) is U-shaped; find compute-optimal \(N^*(C)\).
2. **Fit scaling laws** — power laws like Chinchilla: \(N^* \propto C^{0.5}\), loss \(\downarrow\) with more compute.
3. **Plan experiments** under a **GPU-hour budget** (~12 h on the course API).
4. **Predict final validation loss** for a large held-out config and submit via `save_final_submission()`.
5. **Write up** the math, plots, and decisions in the PDF handout.

This is **science + API orchestration**, not building a transformer from scratch.

---

## Do you need `A3_API_KEY`?

| Path | Need API key? | What you get |
|------|---------------|--------------|
| **Local analysis (self-study)** | No | Isoflops fits, scaling-law math, writeup drafts |
| **Local pytest (staff server)** | No | 7 tests with Postgres + `.env` |
| **Stanford hosted GPU API** | **Yes** | Real training runs on hyperturing |

### How to get `A3_API_KEY` (hosted API only)

Only if you are **enrolled in CS336**:

1. Your key is your **8-digit Stanford student ID** (e.g. `06123456`).
2. Course staff registers it on the server when the assignment opens.
3. Set it: `export A3_API_KEY=06123456`
4. Use `cs336_scaling/client.py` or `scripts/check_api.py` against  
   `http://hyperturing.stanford.edu:8000`

**Self-study without enrollment:** you do **not** have a valid hyperturing key. Use **local analysis** + reference data in `data/isoflops_curves.json`. Optional: run your own local FastAPI stack (see README “For non-students”) — still no GPU training without Modal/CUDA.

---

## Best path for you (local-first)

```bash
cd assignments/assignment3-scaling
brew services start postgresql@15          # if not running
cp .env.example .env                       # already done if .env exists
uv sync --extra server

# 1. Understand reference scaling data
uv run python scripts/analyze_isoflops.py
uv run python scripts/fit_scaling_laws.py

# 2. Verify staff API code (local tests)
uv run pytest -q

# 3. Later — if enrolled — hosted experiments
export A3_API_KEY=your_8_digit_id
uv run python scripts/check_api.py
uv run python scripts/submit_probe.py
```

---

## Key outputs so far

| File | Content |
|------|---------|
| `artifacts/isoflops_compute_optimal.json` | Best \((N, L)\) per compute budget |
| `artifacts/scaling_law_fits.json` | Power-law fits \(N \propto C^b\), \(L \propto C^d\) |

---

## Still to do (assignment deliverables)

- [ ] Your own experiment grid (if using hosted API)
- [ ] Refit scaling laws on **your** run results
- [ ] Final loss prediction + `save_final_submission()`
- [ ] PDF writeup (`cs336_assignment3_scaling.pdf` sections)
