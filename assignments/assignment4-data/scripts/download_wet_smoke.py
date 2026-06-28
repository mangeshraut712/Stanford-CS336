#!/usr/bin/env python3
"""Download a small English WET smoke sample (4 raw WET files -> 1 filtered chunk)."""

from __future__ import annotations

from cs336_data.wet_files import EnglishWetFiles


def main() -> None:
    wet = EnglishWetFiles(n_files=4, group_size=4)
    paths = wet.load_or_create()
    print(f"Created {len(paths)} English WET chunk(s):")
    for p in paths:
        print(f"  {p}")


if __name__ == "__main__":
    main()
