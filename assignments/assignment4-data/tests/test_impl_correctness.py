"""Correctness checks that the staff fixtures do not cover."""

from __future__ import annotations

from cs336_data.deduplication import _band_ranges, exact_line_deduplication, minhash_deduplication
from cs336_data.pii import mask_emails, mask_ips, mask_phone_numbers
from cs336_data.quality import gopher_quality_filter


def test_ipv4_octets_must_be_in_range() -> None:
    masked, count = mask_ips("see 10.0.0.255 and 999.999.999.999 and 1.2.3.256.")
    assert masked == "see |||IP_ADDRESS||| and 999.999.999.999 and 1.2.3.256."
    assert count == 1


def test_phone_and_email_counts_do_not_swallow_neighbors() -> None:
    text = "Call 283-182-3829 or 28318238299 and write a@b.co today."
    masked, count = mask_phone_numbers(text)
    assert count == 1
    assert "28318238299" in masked
    emails, email_count = mask_emails("write a@b.co and not a@b.")
    assert email_count == 1
    assert emails == "write |||EMAIL_ADDRESS||| and not a@b."


def test_gopher_counts_unicode_ellipsis() -> None:
    lines = [" ".join(["alpha"] * 60)]
    lines.extend(["tail …"] * 4)
    lines.extend(["tail end"] * 5)
    assert gopher_quality_filter("\n".join(lines)) is False


def test_exact_dedup_removes_cross_file_lines_and_keeps_within_file_repeats(tmp_path) -> None:
    first = tmp_path / "a.txt"
    second = tmp_path / "b.txt"
    first.write_text("shared\nshared\nonly-a\ntwice\ntwice\n")
    second.write_text("shared\nonly-b\n")
    output = tmp_path / "out"
    exact_line_deduplication([first, second], output)
    assert (output / "a.txt").read_text() == "only-a\ntwice\ntwice\n"
    assert (output / "b.txt").read_text() == "only-b\n"


def test_band_ranges_use_every_hash() -> None:
    assert _band_ranges(5, 2) == [(0, 3), (3, 5)]
    assert _band_ranges(100, 10) == [(index * 10, (index + 1) * 10) for index in range(10)]


def test_minhash_removes_exact_duplicates_when_hash_count_is_not_divisible(tmp_path) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.txt").write_text("alpha beta gamma delta epsilon zeta eta theta\n")
    (docs / "b.txt").write_text("alpha beta gamma delta epsilon zeta eta theta\n")
    (docs / "c.txt").write_text("one two three four five six seven eight nine ten\n")
    output = tmp_path / "out"
    minhash_deduplication(
        input_files=sorted(docs.glob("*.txt")),
        num_hashes=5,
        num_bands=2,
        ngrams=3,
        jaccard_threshold=0.8,
        output_directory=output,
    )
    names = sorted(path.name for path in output.iterdir())
    assert names == ["a.txt", "c.txt"]
