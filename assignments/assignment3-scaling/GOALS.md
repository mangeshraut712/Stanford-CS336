# Assignment 3 — Goal & Workflow

## Goal

Learn how LM validation loss scales with model size \(N\) and compute \(C\) via isoflops curves, Chinchilla-style power laws, and final loss prediction.

## Paths

| Path | API key? | Output |
|------|----------|--------|
| Self-study (done) | No | Reference isoflops fits, writeup, prediction L≈7.17 |
| Local pytest | No | 7/7 pass with PostgreSQL |
| Hosted hyperturing | Yes (`A3_API_KEY`) | Personal training runs |

## Completed workflow

```bash
cd assignments/assignment3-scaling
bash scripts/finalize_assignment.sh
```

## Artifacts

- `artifacts/isoflops_compute_optimal.json`
- `artifacts/scaling_law_fits.json`
- `artifacts/isoflops_parabola_fits.json`
- `artifacts/final_submission_prediction.json`
- `artifacts/isoflops_curves.svg`
- `writeup.md`

## Optional (enrolled)

- [ ] Personal experiment grid on hyperturing
- [ ] Refit on your runs
- [ ] `predict_final_submission.py --submit`
- [ ] Gradescope PDF
