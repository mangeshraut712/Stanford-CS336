# Assignment 4 — Goal & Workflow

## Goal

Build a **data processing pipeline** for LLM pretraining: extract text from HTML, filter by language/quality/toxicity, mask PII, deduplicate, and tokenize Common Crawl + wiki data for training.

## Components implemented

| Component | Module |
|-----------|--------|
| HTML → text | `cs336_data/extract.py` (resiliparse) |
| Language ID | `cs336_data/langid.py` (fastText LID) |
| PII masking | `cs336_data/pii.py` |
| NSFW / toxic classifiers | `cs336_data/toxicity.py` (Dolma fastText) |
| Quality classifier + Gopher rules | `cs336_data/quality.py` |
| Line + MinHash dedup | `cs336_data/deduplication.py` |
| English WET filter | `cs336_data/wet_files.py` + `make_is_english()` |

## Local workflow (self-study)

```bash
cd assignments/assignment4-data
uv sync
uv run python scripts/download_data.py --offline-only   # ~2GB classifiers + sample CC
bash scripts/finalize_assignment.sh                     # 21/21 pytest
```

## Optional pipeline (self-study)

```bash
bash scripts/run_optional_pipeline.sh
```

This runs:
1. **21/21 pytest**
2. **Sample WET processing** — filters 19k records from `local-shared-data/CC/example.warc.wet.gz`
3. **GPT-2 tokenization** → `artifacts/tokens/smoke_train.bin`
4. **Smoke train** — tiny 2-layer model, 10 steps on MPS/CPU via staff `train_from_config`

Additional optional steps:
```bash
uv run python scripts/download_wet_smoke.py   # 4 raw WET files → English-filtered chunk
uv run modal run scripts/train.py --train-bin /path/to/your_data.bin   # 8×B200 full train
```

## Deliverables

- [x] All unit tests (21/21)
- [x] `writeup.md`
- [x] Optional smoke pipeline (`run_optional_pipeline.sh`)
- [x] English WET smoke download (`download_wet_smoke.py`, 4 files)
- [ ] Full Modal 8×B200 training run (optional)
- [ ] Gradescope zip (optional)
