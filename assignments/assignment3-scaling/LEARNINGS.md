# CS336 Assignment 3 — Key Learnings

- Staff provides JAX training + API; student work is scaling science, not model code.
- IsoFLOPs: U-shaped loss vs \(N\) at fixed \(C\); compute-optimal \(N^*(C)\).
- Reference fit: \(N^* \propto C^{0.469}\) — close to Chinchilla 0.5.
- Parabola fits per budget predict off-frontier configs; default config → **7.169**.
- Local stack: PostgreSQL + `.env` + `uv sync --extra server` → 7/7 pytest.
- Hosted API needs 8-digit `A3_API_KEY` (Stanford student ID).

```bash
bash scripts/finalize_assignment.sh
```
