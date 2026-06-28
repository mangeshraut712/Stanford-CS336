# CS336 Assignment 3 — Key Learnings (Self-Study)

## What this assignment is

- **Not** like A1/A2: staff provides JAX training + API; you run experiments and do scaling-law math.
- Hosted API: `http://hyperturing.stanford.edu:8000` with `A3_API_KEY` (8-digit student ID).
- Default budget: **12 GPU-hours** per account (`43200` seconds).

## Isoflops

- At fixed compute budget \(C\), train models of different sizes \(N\); loss vs \(N\) is U-shaped.
- **Compute-optimal** point: \(\arg\min_N L(N, C)\).
- Reference data: `data/isoflops_curves.json` (72 points, 9 compute budgets).

## TrainingConfig knobs

- Model size: `hidden_size`, `num_hidden_layers`, `num_attention_heads`, `head_dim`, etc.
- Training length: `total_train_tokens` (must divide `512 * train_batch_size`).
- `model_seed` must be unique per submitted config (duplicate configs rejected).
- Final submission: large run via `save_final_submission(config, predicted_final_loss)`.

## Local workflow (no GPU)

```bash
uv run python scripts/analyze_isoflops.py
```

## API workflow (needs enrolled key + network)

```bash
export A3_API_KEY=06123456
uv run python scripts/check_api.py
uv run python scripts/submit_probe.py   # small test job
```

## Deliverables (handout)

1. Scaling experiments + fits
2. Predicted final loss for validation run
3. PDF writeup with derivations
