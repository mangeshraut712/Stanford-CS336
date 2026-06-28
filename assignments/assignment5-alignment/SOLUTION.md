# Assignment 5 — Solution Summary

## GRPO core (`cs336_alignment/grpo.py`)

1. **Tokenization** — Concatenate encoded prompt + output; `input_ids = seq[:-1]`, `labels = seq[1:]`, `response_mask` marks response label positions (excluding pad).

2. **Log-probs** — Forward pass; gather `log_softmax(logits)` at `labels`; optional per-token entropy.

3. **Rewards** — Call staff-style reward fn per rollout; stack `reward` scalars; log means.

4. **Group advantages** — Reshape to `(n_groups, group_size)`; optional mean baseline; normalize by group std (GRPO), mean (MaxRL), or none (Dr.GRPO / RFT).

5. **Policy loss**
   - On-policy: `-A * log π`
   - Off-policy noclip: `-A * exp(log π - log π_old)`
   - GRPO clip: `-min(A·r, A·clip(r))` per token
   - GSPO: sequence-mean log-ratio, clip at sequence level, broadcast to tokens

6. **Aggregation** — `sequence`: mean over masked tokens per seq, then batch mean; `constant`: total masked sum / fixed constant.

7. **Train step** — Compute advantages once; microbatch forward/backward; for `constant` loss norm, backprop full micro loss (still divide reported loss sum for logging); clip grads; optimizer step.

## Supplement

- **SFT packing** — Alpaca-style template + EOS between examples; chunk into `seq_length` with global shifted labels.
- **Parsers** — MMLU: regex for A–D; GSM8K: last number in output.
- **DPO** — Standard `-log σ(β((log π_θ(y_w|x) - log π_θ(y_l|x)) - (log π_ref(y_w|x) - log π_ref(y_l|x))))` with response-token log-prob sum.
