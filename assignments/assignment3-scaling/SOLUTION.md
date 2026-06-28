# CS336 Assignment 3 — Our Solution

**Author:** Mangesh Raut · **Course:** Stanford CS336 (self-study)

## Completed (self-study track)

| Deliverable | Status | Location |
|-------------|--------|----------|
| Isoflops analysis | Done | `artifacts/isoflops_compute_optimal.json` |
| Scaling-law fits | Done | `artifacts/scaling_law_fits.json` |
| Parabola fits + prediction | Done | `artifacts/final_submission_prediction.json` |
| Local pytest | **7/7** | PostgreSQL + `.env` |
| Writeup | Done | `writeup.md` |
| Hosted GPU runs | Optional | Needs `A3_API_KEY` |

## Reproduce

```bash
cd assignments/assignment3-scaling
bash scripts/finalize_assignment.sh
```

## Key results

- \(N^* \propto C^{0.469}\) (Chinchilla ≈ 0.5)
- Default config predicted loss: **7.169**

## Code added

- `cs336_scaling/analysis/scaling.py` — fits and prediction
- `scripts/run_analysis.py`, `predict_final_submission.py`, `verify_complete.sh`
