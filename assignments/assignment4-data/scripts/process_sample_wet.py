#!/usr/bin/env python3
"""Process the offline sample WET file through the filtering pipeline."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

from warcio.archiveiterator import ArchiveIterator

from cs336_data.common import get_shared_assets_path
from cs336_data.pipeline import filter_document

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts" / "processed"


def main() -> None:
    wet_path = get_shared_assets_path() / "CC" / "example.warc.wet.gz"
    if not wet_path.exists():
        raise FileNotFoundError(f"Missing {wet_path}; run scripts/download_data.py --offline-only")

    ART.mkdir(parents=True, exist_ok=True)
    out_path = ART / "sample_corpus.txt"
    stats = {"records": 0, "kept": 0, "chars": 0}

    with gzip.open(wet_path, "rb") as stream, out_path.open("w") as out:
        for rec in ArchiveIterator(stream):
            if rec.rec_type != "conversion":
                continue
            stats["records"] += 1
            text = rec.content_stream().read().decode("utf-8", errors="replace")
            cleaned = filter_document(text)
            if cleaned:
                stats["kept"] += 1
                stats["chars"] += len(cleaned)
                out.write(cleaned)
                out.write("\n\n")

    summary_path = ART / "sample_corpus_stats.json"
    summary_path.write_text(json.dumps(stats, indent=2))
    print(f"Processed {stats['records']} WET records -> kept {stats['kept']} ({stats['chars']:,} chars)")
    print(f"Wrote {out_path}")
    print(f"Stats {summary_path}")


if __name__ == "__main__":
    main()
