#!/usr/bin/env python3
"""Probe the hosted Assignment 3 training API (requires A3_API_KEY)."""

from __future__ import annotations

import os
import sys

import requests

API_BASE = "http://hyperturing.stanford.edu:8000"


def main() -> None:
    key = os.getenv("A3_API_KEY", "")
    if len(key) != 8 or not key.isdigit():
        print("Set A3_API_KEY to your 8-digit student ID, e.g. export A3_API_KEY=06123456")
        sys.exit(1)

    try:
        r = requests.get(f"{API_BASE}/docs", timeout=15)
        print(f"API docs: HTTP {r.status_code}")
    except requests.RequestException as exc:
        print(f"API unreachable: {exc}")
        sys.exit(1)

    from cs336_scaling.client import get_budget, list_experiments

    budget = get_budget()
    exps = list_experiments()
    print(f"Budget: {budget.used_seconds:.0f}s used, {budget.remaining_seconds:.0f}s remaining")
    print(f"Experiments on account: {len(exps)}")


if __name__ == "__main__":
    main()
