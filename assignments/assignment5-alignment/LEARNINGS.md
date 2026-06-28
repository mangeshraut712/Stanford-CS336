# CS336 Assignment 5 — Key Learnings

- **GRPO** compares rollouts within a prompt group; mean baseline + std normalization stabilizes policy gradients.
- **Off-policy surrogates** use importance ratios; PPO-style clipping (`-min(A·r, A·clip(r))`) prevents destructive updates.
- **GSPO** clips at sequence level (mean log-ratio over response tokens), then broadcasts to all positions.
- **Constant loss normalization** scales by a fixed training constant; backprop uses full microbatch loss (grad accum still splits batches).
- **DPO** needs the Alpaca SFT template + EOS — raw `prompt + response` concat does not match staff tests.
- **SFT packing** concatenates formatted examples with EOS, then chunks into fixed `seq_length` with globally shifted labels.

```bash
bash scripts/finalize_assignment.sh
```
