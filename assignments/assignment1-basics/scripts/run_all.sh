#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

ARTIFACTS="$ROOT/artifacts"
DATA="$ROOT/data"
mkdir -p "$ARTIFACTS/bpe" "$ARTIFACTS/tokens" "$ARTIFACTS/experiments"

echo "=== §2.5 Train TinyStories BPE (10k) ==="
uv run python scripts/train_bpe.py \
  --input "$DATA/TinyStoriesV2-GPT4-train.txt" \
  --output-dir "$ARTIFACTS/bpe/tinystories_10k" \
  --vocab-size 10000 \
  --num-workers 4 \
  --name tinystories_10k

echo "=== §2.5 Train OWT BPE (32k) ==="
uv run python scripts/train_bpe.py \
  --input "$DATA/owt_train.txt" \
  --output-dir "$ARTIFACTS/bpe/owt_32k" \
  --vocab-size 32000 \
  --num-workers 1 \

echo "=== §2.7 Tokenizer experiments ==="
uv run python scripts/bpe_experiments.py

echo "=== §2.7(d) Tokenize datasets ==="
for split in train valid; do
  uv run python scripts/tokenize_corpus.py \
    --input "$DATA/TinyStoriesV2-GPT4-${split}.txt" \
    --output "$ARTIFACTS/tokens/tinystories_${split}.bin" \
    --tokenizer-dir "$ARTIFACTS/bpe/tinystories_10k"
done

for split in train valid; do
  uv run python scripts/tokenize_corpus.py \
    --input "$DATA/owt_${split}.txt" \
    --output "$ARTIFACTS/tokens/owt_${split}.bin" \
    --tokenizer-dir "$ARTIFACTS/bpe/owt_32k"
done

echo "=== §7.2 TinyStories training (low-resource Mac-friendly) ==="
uv run python train.py \
  --train-tokens "$ARTIFACTS/tokens/tinystories_train.bin" \
  --val-tokens "$ARTIFACTS/tokens/tinystories_valid.bin" \
  --tokenizer-dir "$ARTIFACTS/bpe/tinystories_10k" \
  --output-dir "$ARTIFACTS/experiments/tinystories_main" \
  --run-name tinystories_main \
  --batch-size 32 \
  --max-steps 5000 \
  --eval-every 500 \
  --checkpoint-every 2500

echo "=== §7.4 OpenWebText training ==="
uv run python train.py \
  --train-tokens "$ARTIFACTS/tokens/owt_train.bin" \
  --val-tokens "$ARTIFACTS/tokens/owt_valid.bin" \
  --tokenizer-dir "$ARTIFACTS/bpe/owt_32k" \
  --output-dir "$ARTIFACTS/experiments/owt_main" \
  --run-name owt_main \
  --vocab-size 32000 \
  --batch-size 16 \
  --max-steps 5000 \
  --eval-every 500 \
  --checkpoint-every 2500

echo "=== §7.5 Leaderboard modification: tuned LR + larger batch ==="
uv run python train.py \
  --train-tokens "$ARTIFACTS/tokens/owt_train.bin" \
  --val-tokens "$ARTIFACTS/tokens/owt_valid.bin" \
  --tokenizer-dir "$ARTIFACTS/bpe/owt_32k" \
  --output-dir "$ARTIFACTS/experiments/leaderboard_mod" \
  --run-name leaderboard_mod \
  --vocab-size 32000 \
  --batch-size 32 \
  --max-steps 8000 \
  --max-lr 6e-4 \
  --warmup-steps 400 \
  --eval-every 400

echo "=== Submission zip ==="
bash make_submission.sh

echo "Done. See artifacts/ and writeup.md"
