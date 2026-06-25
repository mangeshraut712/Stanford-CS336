# Local datasets (not in git)

Download into this directory:

```bash
mkdir -p data && cd data

# TinyStories
curl -LO https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
curl -LO https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt

# OpenWebText sample (~11GB train)
curl -LO https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz
gunzip owt_train.txt.gz
curl -LO https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz
gunzip owt_valid.txt.gz
```

**Mac fast path:** `head -c 2147483648 data/owt_train.txt > data/owt_train_2gb.txt` for OWT BPE on a 2GB subset.

| File | Approx. size |
|------|----------------|
| `TinyStoriesV2-GPT4-train.txt` | 2.1 GB |
| `TinyStoriesV2-GPT4-valid.txt` | 22 MB |
| `owt_train.txt` | 11 GB |
| `owt_valid.txt` | 277 MB |
