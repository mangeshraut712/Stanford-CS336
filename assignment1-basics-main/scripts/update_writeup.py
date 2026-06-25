#!/usr/bin/env python3
"""Append experiment metrics from run_summary.json into artifacts/experiment_log.md."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
LOG = ARTIFACTS / "experiment_log.md"


def load_summary(name: str) -> dict | None:
    path = ARTIFACTS / "experiments" / name / "run_summary.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def main() -> None:
    runs = ["smoke_test", "tinystories_main", "owt_main", "leaderboard_mod"]
    lines = [
        f"\n## Auto-update {datetime.now(timezone.utc).isoformat()}\n",
    ]
    for name in runs:
        summary = load_summary(name)
        if summary is None:
            lines.append(f"- **{name}**: not run yet\n")
            continue
        val = summary.get("final_val_loss")
        train = summary.get("final_train_loss")
        tokens = summary.get("total_tokens_seen")
        elapsed = summary.get("total_elapsed_sec")
        v_str = f"{val:.4f}" if isinstance(val, (int, float)) else str(val)
        t_str = f"{train:.4f}" if isinstance(train, (int, float)) else str(train)
        lines.append(
            f"- **{name}**: val_loss={v_str}, train_loss={t_str}, "
            f"tokens={tokens}, elapsed={elapsed}\n"
        )
    bpe_exp = ARTIFACTS / "bpe_experiments.json"
    if bpe_exp.exists():
        with open(bpe_exp) as f:
            data = json.load(f)
        lines.append(
            f"- **bpe_experiments**: throughput={data.get('throughput_bytes_per_sec', 0):.0f} B/s, "
            f"pile_est={data.get('pile_825gb_hours_estimate', 0):.1f}h\n"
        )
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG, "a") as f:
        f.writelines(lines)
    print("".join(lines))
