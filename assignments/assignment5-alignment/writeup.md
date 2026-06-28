# Assignment 5 Writeup — Alignment (GRPO)

Self-study notes for the CS336 alignment assignment.

## What we built

A complete **Group Relative Policy Optimization (GRPO)** training step on CPU-friendly tiny GPT-2:

- Tokenize prompt/response pairs with a response mask for policy gradients.
- Score rollouts with a reward function; compute group-normalized advantages (GRPO, Dr.GRPO, MaxRL, RFT variants).
- Support on-policy and off-policy (importance sampling, token-level PPO clip, GSPO sequence clip).
- Aggregate per-token losses with sequence-level or constant normalization; gradient accumulation and clipping.

All **19 required** snapshot tests in `tests/test_grpo.py` pass on Mac without GPU.

## Key design choices

- **Advantage normalization:** Population std (`ddof=1`) within each prompt group matches staff snapshots.
- **Off-policy losses:** PPO-style `-min(A·r, A·clip(r))` without extra `log π` factors in the surrogate.
- **Constant loss norm:** Microbatch losses backprop at full scale (normalization constant already scales the objective); logged loss is the sum across microbatches.

## Optional supplement

- Packed instruction-tuning dataset (Llama-3 tokenizer fixture).
- MMLU / GSM8K output parsers for eval.
- DPO per-instance loss (Alpaca template + EOS; `tests/test_dpo.py` passes).

## Next steps (not run locally)

- Modal job: GRPO on GSM8K with vLLM rollouts (`cs336_alignment/vllm_utils.py`, `drgrpo_grader.py`).
- SFT → DPO pipeline on Alpaca-style data for safety / instruction following.
