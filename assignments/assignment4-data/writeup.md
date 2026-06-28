# CS336 Assignment 4 — Data Pipeline Writeup

**Author:** Mangesh Raut · **Track:** Self-study (local tests + offline assets)

## 1. Overview

Assignment 4 builds tooling to turn raw web data (Common Crawl WARC/WET, wiki dumps) into clean pretraining corpora. Components: extraction, filtering, PII removal, quality/toxicity classifiers, and deduplication.

## 2. Implemented components

### 2.1 HTML extraction

Uses **resiliparse**: parse HTML to `HTMLTree`, extract plain text with `main_content=False, links=False`. Matches staff fixture on Moby Dick HTML.

### 2.2 Language identification

fastText `lid.176.bin` — returns ISO codes (`en`, `zh`) with confidence. Used for `is_english()` in WET filtering (threshold 0.7).

### 2.3 PII masking

Regex-based replacement with placeholders:
- `|||EMAIL_ADDRESS|||`
- `|||PHONE_NUMBER|||` (multiple US formats)
- `|||IP_ADDRESS|||`

Existing placeholders are not double-counted.

### 2.4 Toxicity classifiers

Pretrained Dolma Jigsaw fastText models for NSFW and hate speech (`non-nsfw`/`nsfw`, `non-toxic`/`toxic`).

### 2.5 Quality

- **Gopher heuristics:** 50–100k non-symbol words, avg word length 3–10, ≤30% ellipsis lines, ≥80% alphabetic words.
- **Quality classifier:** fastText supervised model (`cc` vs `wiki`) trained on fixture samples + augmentations.

### 2.6 Deduplication

| Method | Rule |
|--------|------|
| Exact line | Remove lines appearing in **more than one file** |
| MinHash | Word n-gram shingles, mmh3 minhash, LSH bands, Jaccard ≥ threshold, keep earliest doc per cluster |

## 3. Verification

```bash
uv sync
uv run python scripts/download_data.py --offline-only
bash scripts/finalize_assignment.sh   # 21/21 pass
```

## 4. Limitations (self-study)

- No full Modal CC download or GPU training run
- Quality classifier trained on small fixture set (sufficient for unit tests)
- `local-shared-data/` (~2GB) not committed to git

## References

- DeepMind Gopher paper (quality heuristics)
- CS336 Assignment 4 handout: `cs336_assignment4_data.pdf`
- Dolma / AllenAI Jigsaw fastText classifiers
