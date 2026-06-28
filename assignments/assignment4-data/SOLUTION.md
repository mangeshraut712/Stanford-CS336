# CS336 Assignment 4 — Our Solution

**Author:** Mangesh Raut · **Course:** Stanford CS336 (self-study)

## Completed

| Component | Tests | Implementation |
|-----------|-------|----------------|
| HTML extraction | 1 | resiliparse `HTMLTree` + `extract_plain_text` |
| Language ID | 2 | fastText `lid.176.bin` |
| PII masking | 5 | regex emails, phones, IPs |
| NSFW / toxic | 2 | Dolma Jigsaw fastText models |
| Quality + Gopher | 8 | trained fastText + heuristic rules |
| Deduplication | 3 | cross-file line dedup + MinHash LSH |

**Total: 21/21 pytest pass**

## Reproduce

```bash
cd assignments/assignment4-data
uv sync
bash scripts/finalize_assignment.sh
```

## Design notes

- **Exact line dedup:** drop lines appearing in more than one file (not keep-first).
- **MinHash:** word n-gram shingles, mmh3 signatures, LSH banding, union-find clustering.
- **Quality classifier:** fastText supervised model trained on fixture cc/wiki samples (+ augmentations), cached at `cs336_data/models/quality_cc_wiki.bin`.
- **is_english:** fastText LID with probability ≥ 0.7.

## Optional next steps

- Run full Modal pipeline for wiki URL extraction + English WET download
- Tokenize filtered corpus and train with staff `scripts/train.py`
