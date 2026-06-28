# CS336 Assignment 4 — Key Learnings

- Data quality matters as much as model architecture for pretraining.
- **resiliparse** extracts clean text from HTML better than naive regex stripping.
- **fastText** handles language ID and Dolma-style toxicity classifiers efficiently on CPU.
- **Gopher rules** are cheap heuristics (word count, avg length, ellipsis ratio) before expensive training.
- **Exact line dedup** removes boilerplate navigation lines repeated across crawled pages.
- **MinHash + LSH** scales fuzzy document dedup (e.g. duplicate MIT licenses with formatting drift).
- Offline assets (~2GB) live in `local-shared-data/`; verify NSFW model download completes (~992MB).

```bash
bash scripts/finalize_assignment.sh
```
