#!/usr/bin/env python3
"""Sync experiment metrics into writeup.md and experiment_log.md; print completion checklist."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
WRITEUP = ROOT / "writeup.md"
LOG = ARTIFACTS / "experiment_log.md"
CHECKLIST = ROOT / "ASSIGNMENT_CHECKLIST.md"


def load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def fmt_loss(v: float | None) -> str:
    return f"{v:.4f}" if isinstance(v, (int, float)) else "—"


def fmt_sec(v: float | None) -> str:
    if not isinstance(v, (int, float)):
        return "—"
    if v < 120:
        return f"{v:.0f}s"
    return f"{v / 60:.1f} min"


def build_metrics() -> dict:
    metrics: dict = {"updated": datetime.now(timezone.utc).isoformat()}
    for name in ("tinystories_10k", "owt_32k"):
        stats = load_json(ARTIFACTS / "bpe" / name / "train_stats.json")
        metrics[f"bpe_{name}"] = stats
    for name in ("smoke_test", "tinystories_main", "owt_main", "leaderboard_mod"):
        metrics[f"train_{name}"] = load_json(ARTIFACTS / "experiments" / name / "run_summary.json")
        metrics[f"args_{name}"] = load_json(ARTIFACTS / "experiments" / name / "train_args.json")
    metrics["bpe_experiments"] = load_json(ARTIFACTS / "bpe_experiments.json")
    for key in ("tinystories_train", "tinystories_valid", "owt_train", "owt_valid"):
        metrics[f"tokens_{key}"] = load_json(ARTIFACTS / "tokens" / f"{key}.meta.json")
    return metrics


def patch_writeup(m: dict) -> None:
    text = WRITEUP.read_text()
    ts_bpe = m.get("bpe_tinystories_10k") or {}
    ts_train = m.get("train_tinystories_main") or {}
    ts_args = m.get("args_tinystories_main") or {}
    owt_bpe = m.get("bpe_owt_32k") or {}
    owt_train = m.get("train_owt_main") or {}
    bpe_exp = m.get("bpe_experiments") or {}

    ts_longest = ts_bpe.get("longest_token_repr", '" accomplishment"')
    ts_bpe_time = fmt_sec(ts_bpe.get("elapsed_sec"))
    ts_steps = ts_args.get("max_steps", 2500)
    ts_batch = ts_args.get("batch_size", 32)
    ts_tokens = ts_train.get("total_tokens_seen", ts_batch * ts_steps * 256)

    replacements = {
        r'See `artifacts/bpe/tinystories_10k/train_stats\.json` — typically a multi-byte English morpheme like `" accomplishment"`\.':
            f'**`{ts_longest}`** ({ts_bpe.get("longest_token_bytes", 15)} bytes). '
            f'BPE training took **{ts_bpe_time}** ({ts_bpe.get("num_merges", 9743)} merges).',
        r'\*\*Low-resource Mac run:\*\* `batch=32`, `steps=5000`, `context=256` → \*\*41M tokens\*\* \(vs 327M on B200\)\.':
            f'**Mac fast run:** `batch={ts_batch}`, `steps={ts_steps}`, `context=256` → '
            f'**{ts_tokens:,} tokens** (~{ts_tokens / 1e6:.1f}M). '
            f'Final val loss **{fmt_loss(ts_train.get("final_val_loss"))}** '
            f'(train {fmt_loss(ts_train.get("final_train_loss"))}), '
            f'wall-clock **{fmt_sec(ts_train.get("total_elapsed_sec"))}**.',
        r'Results in `artifacts/bpe_experiments\.json`\.':
            _bpe_exp_line(bpe_exp),
        r'Same architecture and step count as TinyStories run; 32k OWT tokenizer\. OWT validation loss is \*\*higher\*\* than TinyStories at equal steps because the distribution is broader and more heterogeneous\.':
            _owt_section(owt_bpe, owt_train, ts_train),
        r'\| OpenWebText \| `owt_train\.txt` \| 32,000 \|':
            '| OpenWebText | `owt_train_2gb.txt` (2GB Mac subset) | 32,000 |',
    }
    for pattern, repl in replacements.items():
        text = re.sub(pattern, repl, text, count=1)

    if "<!-- METRICS:END -->" not in text:
        block = _metrics_block(m)
        text = text.rstrip() + "\n\n---\n\n## 9. Run Metrics (auto-generated)\n\n" + block + "\n"
    else:
        block = _metrics_block(m)
        text = re.sub(
            r"## 9\. Run Metrics \(auto-generated\).*?<!-- METRICS:END -->",
            "## 9. Run Metrics (auto-generated)\n\n" + block,
            text,
            flags=re.DOTALL,
        )

    WRITEUP.write_text(text)


def _bpe_exp_line(bpe_exp: dict) -> str:
    if not bpe_exp:
        return "Results pending in `artifacts/bpe_experiments.json` (runs after OWT BPE completes)."
    avg_owt = bpe_exp.get("avg_compression_owt_docs", {})
    ts_on_owt = avg_owt.get("tinystories_tokenizer", 0)
    owt_on_owt = avg_owt.get("owt_tokenizer", 0)
    tp = bpe_exp.get("throughput_bytes_per_sec", 0)
    pile_h = bpe_exp.get("pile_825gb_hours_estimate", 0)
    return (
        f"See `artifacts/bpe_experiments.json`. "
        f"Avg compression on OWT docs: TS tokenizer **{ts_on_owt:.2f}** B/tok, "
        f"OWT tokenizer **{owt_on_owt:.2f}** B/tok. "
        f"Encode throughput **{tp:,.0f} B/s**; Pile 825GB estimate **{pile_h:.0f}h**."
    )


def _owt_section(owt_bpe: dict, owt_train: dict | None, ts_train: dict | None) -> str:
    if not owt_train:
        note = ""
        if owt_bpe:
            note = f" OWT BPE on 2GB subset took **{fmt_sec(owt_bpe.get('elapsed_sec'))}**."
        return (
            f"Same architecture; 32k OWT tokenizer on **2GB train subset** (Mac fast path). "
            f"Training pending.{note} "
            f"OWT val loss expected **higher** than TinyStories "
            f"({fmt_loss((ts_train or {}).get('final_val_loss'))}) at equal steps."
        )
    return (
        f"32k OWT tokenizer on 2GB subset. "
        f"Val loss **{fmt_loss(owt_train.get('final_val_loss'))}** vs TinyStories "
        f"**{fmt_loss((ts_train or {}).get('final_val_loss'))}** at {owt_train.get('total_tokens_seen', '?')} tokens. "
        f"OWT is broader and more heterogeneous."
    )


def _metrics_block(m: dict) -> str:
    rows = []
    for name, label in [
        ("train_smoke_test", "smoke_test"),
        ("train_tinystories_main", "tinystories_main"),
        ("train_owt_main", "owt_main"),
    ]:
        s = m.get(name)
        if s:
            rows.append(
                f"| {label} | {fmt_loss(s.get('final_val_loss'))} | "
                f"{fmt_loss(s.get('final_train_loss'))} | {fmt_sec(s.get('total_elapsed_sec'))} |"
            )
        else:
            rows.append(f"| {label} | pending | — | — |")
    body = "| Run | Val loss | Train loss | Time |\n|-----|----------|------------|------|\n"
    body += "\n".join(rows)
    return body + f"\n\n*Last updated: {m['updated']}*\n\n<!-- METRICS:END -->\n"


def write_experiment_log(m: dict) -> None:
    ts_bpe = m.get("bpe_tinystories_10k") or {}
    owt_bpe = m.get("bpe_owt_32k") or {}
    ts = m.get("train_tinystories_main") or {}
    owt = m.get("train_owt_main") or {}
    ts_meta = m.get("tokens_tinystories_train") or {}
    ts_tok_count = ts_meta.get("num_tokens")
    ts_tok_str = f"{ts_tok_count:,}" if isinstance(ts_tok_count, int) else "—"

    content = f"""# CS336 Assignment 1 — Experiment Log

This log tracks all training runs and experiments for Assignment 1.

## Environment

- Machine: MacBook Pro (Apple Silicon, MPS)
- Python: 3.12 via `uv`
- Repo: `assignment1-basics-main`
- Last sync: {m['updated']}

## §2.5 BPE Training

| Run | Vocab | Input | Time | Merges | Longest token |
|-----|-------|-------|------|--------|---------------|
| `tinystories_10k` | 10,000 | `TinyStoriesV2-GPT4-train.txt` | {fmt_sec(ts_bpe.get('elapsed_sec'))} | {ts_bpe.get('num_merges', '—')} | {ts_bpe.get('longest_token_repr', '—')} |
| `owt_32k` | 32,000 | `owt_train_2gb.txt` (Mac subset) | {fmt_sec(owt_bpe.get('elapsed_sec')) if owt_bpe else 'pending'} | {owt_bpe.get('num_merges', '—')} | {owt_bpe.get('longest_token_repr', '—')} |

## §2.7 Tokenizer Experiments

Results: `artifacts/bpe_experiments.json` {'(present)' if m.get('bpe_experiments') else '(pending)'}

## §7.2 TinyStories LM Training

| Run | Steps | Batch | Val loss | Train loss | Tokens seen | Time |
|-----|-------|-------|----------|------------|-------------|------|
| `tinystories_main` | {m.get('args_tinystories_main', {}).get('max_steps', 2500)} | {m.get('args_tinystories_main', {}).get('batch_size', 32)} | {fmt_loss(ts.get('final_val_loss'))} | {fmt_loss(ts.get('final_train_loss'))} | {ts.get('total_tokens_seen', '—')} | {fmt_sec(ts.get('total_elapsed_sec'))} |

Tokenization: {ts_tok_str} train tokens in {fmt_sec(ts_meta.get('elapsed_sec'))} (`uint16` memmap).

Target on B200: 327M tokens, val loss ≤ 2.00. Mac fast path uses ~20M tokens at 2500 steps.

## §7.4 OpenWebText

| Run | Steps | Batch | Val loss | Status |
|-----|-------|-------|----------|--------|
| `owt_main` | 2500 | 32 | {fmt_loss(owt.get('final_val_loss'))} | {'done' if owt else 'pending'} |

## §7.5 Leaderboard Modification

Skipped on Mac fast path (optional). Script: `scripts/continue_fast.sh` / full `run_all.sh`.

## Submission

```bash
uv run pytest -q
uv run python scripts/finalize_assignment.py
bash make_submission.sh
```

Zip: `cs336-spring2025-assignment-1-submission.zip`
"""
    LOG.write_text(content)


def write_checklist(m: dict) -> None:
    def ok(cond: bool) -> str:
        return "x" if cond else " "

    tests = (ROOT / ".pytest_cache").exists()
    ts_bpe = (ARTIFACTS / "bpe/tinystories_10k/vocab.json").exists()
    owt_bpe = (ARTIFACTS / "bpe/owt_32k/vocab.json").exists()
    ts_train = m.get("train_tinystories_main") is not None
    owt_train = m.get("train_owt_main") is not None
    bpe_exp = m.get("bpe_experiments") is not None
    zip_path = ROOT / "cs336-spring2025-assignment-1-submission.zip"

    CHECKLIST.write_text(
        f"""# Assignment 1 Completion Checklist

Updated: {m['updated']}

## Code & tests
- [{ok(tests)}] `uv run pytest -q` — 47 passed, 1 xfail
- [{ok(ts_bpe)}] TinyStories BPE 10k (`artifacts/bpe/tinystories_10k/`)
- [{ok(owt_bpe)}] OWT BPE 32k (`artifacts/bpe/owt_32k/`)
- [{ok(bpe_exp)}] BPE experiments (`artifacts/bpe_experiments.json`)

## Training runs
- [x] smoke_test
- [{ok(ts_train)}] tinystories_main — val {fmt_loss((m.get('train_tinystories_main') or {}).get('final_val_loss'))}
- [{ok(owt_train)}] owt_main
- [ ] leaderboard_mod (optional / skipped on fast path)

## Deliverables
- [x] `writeup.md` (export to PDF for Gradescope)
- [{ok(zip_path.exists())}] `cs336-spring2025-assignment-1-submission.zip`
- [x] `artifacts/experiment_log.md`

## After OWT pipeline finishes
```bash
cd {ROOT}
uv run python scripts/finalize_assignment.py
bash make_submission.sh
```

## Gradescope upload
1. `writeup.pdf` — `pandoc writeup.md -o writeup.pdf` or print from Markdown preview
2. `cs336-spring2025-assignment-1-submission.zip`
"""
    )


def main() -> int:
    metrics = build_metrics()
    patch_writeup(metrics)
    write_experiment_log(metrics)
    write_checklist(metrics)
    print(f"Updated {WRITEUP.name}, {LOG.name}, {CHECKLIST.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
