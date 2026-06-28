# CS336 Assignment 3 — Scaling Laws Writeup

**Author:** Mangesh Raut · **Track:** Self-study (reference isoflops + local API tests)

## 1. Overview

Assignment 3 studies how validation loss scales with model size \(N\) and training compute \(C\). Staff provides the JAX training stack and hosted API; student work is experiment design, isoflops analysis, scaling-law fits, and final loss prediction.

## 2. Scaling laws background

**Chinchilla (Hoffmann et al.):** compute-optimal training satisfies \(N^* \propto C^{0.5}\), \(D^* \propto C^{0.5}\), and \(L^* \propto C^{-0.05}\).

**IsoFLOPs:** at fixed \(C\), plot \(L\) vs \(N\) — U-shaped curve; minimum is \(N^*(C)\).

## 3. Reference data analysis

Source: `data/isoflops_curves.json` (72 points, 9 compute budgets from \(6\times10^{18}\) to \(3\times10^{21}\) FLOPs).

**Power-law fits on compute-optimal frontier:**

| Law | Fit | Reference |
|-----|-----|-----------|
| \(N^* \approx a C^b\) | \(a \approx 1.16\), **\(b \approx 0.469\)** | Chinchilla \(b \approx 0.5\) |
| \(L^* \approx c C^d\) | \(c \approx 132\), **\(d \approx -0.072\)** | Chinchilla \(d \approx -0.05\) |

**Per-budget parabolas** in \(\log N\) capture U-shapes for off-optimal predictions. Artifacts: `artifacts/scaling_law_fits.json`, `artifacts/isoflops_parabola_fits.json`, `artifacts/isoflops_curves.svg`.

## 4. Final loss prediction

Default client config (`default_training_config()`): \(N \approx 51.4\)M params, \(C \approx 6ND \approx 3.2\times10^{14}\) FLOPs.

| Method | Predicted loss |
|--------|----------------|
| Parabola fit (recommended) | **7.169** |
| Compute-optimal power law only | 11.77 (invalid — config not on frontier) |

Matches staff notebook placeholder (~7.2). See `artifacts/final_submission_prediction.json`.

```bash
export A3_API_KEY=06123456   # if enrolled
uv run python scripts/predict_final_submission.py --submit
```

## 5. Local verification

```bash
brew services start postgresql@15
uv sync --extra server
bash scripts/finalize_assignment.sh   # 7/7 pytest + artifacts
```

## 6. Limitations (self-study)

- No personal hyperturing runs without enrolled `A3_API_KEY`
- Small probe configs extrapolate below reference compute range
- Gradescope PDF optional; this `writeup.md` is the portfolio equivalent

## References

- Hoffmann et al., Chinchilla, 2022 · Kaplan et al., Scaling Laws, 2020 · `cs336_assignment3_scaling.pdf`
