# CS336 Spring 2025 Assignment 1: Basics

For a full description of the assignment, see the assignment handout at
[cs336_assignment1_basics.pdf](./cs336_assignment1_basics.pdf)

If you see any issues with the assignment handout or code, please feel free to
raise a GitHub issue or open a pull request with a fix.

## Setup

### Environment
We manage our environments with `uv` to ensure reproducibility, portability, and ease of use.
Install `uv` [here](https://github.com/astral-sh/uv#installation) (recommended), or run `pip install uv`/`brew install uv`.
We recommend reading a bit about managing projects in `uv` [here](https://docs.astral.sh/uv/guides/projects/#managing-dependencies) (you will not regret it!).

You can now run any code in the repo using
```sh
uv run <python_file_path>
```
and the environment will be automatically solved and activated when necessary.

### Run unit tests


```sh
uv run pytest
```

Initially, all tests should fail with `NotImplementedError`s.
To connect your implementation to the tests, complete the
functions in [./tests/adapters.py](./tests/adapters.py).

### Train BPE tokenizers, tokenize data, and train a model

```sh
# Full pipeline (BPE + tokenization + training + submission zip)
bash scripts/run_all.sh

# Unattended overnight run (keeps Mac awake; resumes from existing artifacts)
bash scripts/run_overnight.sh
tail -f artifacts/pipeline.log

# Or individual steps:
uv run python scripts/train_bpe.py --input data/TinyStoriesV2-GPT4-train.txt \
  --output-dir artifacts/bpe/tinystories_10k --vocab-size 10000 --num-workers 4
uv run python scripts/tokenize_corpus.py --input data/TinyStoriesV2-GPT4-train.txt \
  --output artifacts/tokens/tinystories_train.bin --tokenizer-dir artifacts/bpe/tinystories_10k
uv run python train.py --train-tokens artifacts/tokens/tinystories_train.bin \
  --val-tokens artifacts/tokens/tinystories_valid.bin \
  --tokenizer-dir artifacts/bpe/tinystories_10k \
  --output-dir artifacts/experiments/tinystories_main
```

See [writeup.md](./writeup.md) for experiment documentation.

## Our solution

This repo contains **our own Assignment 1 implementation** (not staff code). See [SOLUTION.md](./SOLUTION.md) for structure, results, and how to package for submission or reuse in Assignment 2.

```bash
bash scripts/package_solution.sh   # pytest + sync docs + submission zip
```

### Download data
Download the TinyStories data and a subsample of OpenWebText

``` sh
mkdir -p data
cd data

wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt

wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz
gunzip owt_train.txt.gz
wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz
gunzip owt_valid.txt.gz

cd ..
```

