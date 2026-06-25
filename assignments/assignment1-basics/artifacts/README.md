# Artifacts

Tracked in git (small JSON/text):
- `bpe/tinystories_10k/` — trained tokenizer (vocab, merges, stats)
- `experiments/*/run_summary.json` — training metrics
- `experiments/*/generated_sample.txt` — generation samples
- `tokens/*.meta.json` — tokenization metadata (not the `.bin` files)

**Local only** (gitignored):
- `tokens/*.bin` — memmap token arrays (GB-scale)
- `experiments/*/final_model/`, `latest_model/` — checkpoints (`.pt`)
- `pipeline*.log`, `pipeline.pid`
- `bpe/owt_32k/` until trained locally

Regenerate with `bash scripts/run_all.sh` or `bash scripts/run_fast_ts.sh`.
