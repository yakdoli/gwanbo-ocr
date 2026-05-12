from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from gwanbo_ocr.sampling import build_sample_suite, critical_tokens_for_entry, sample_entries


def test_sample_entries_are_deterministic_and_stratified() -> None:
    entries = [
        _entry("a", "searchThema", "2001"),
        _entry("b", "searchThema", "2001"),
        _entry("c", "pety", "1999"),
        _entry("d", "pety", "1999"),
        _entry("missing", "pety", "2000", exists=False),
    ]

    first = sample_entries(entries, size=2, seed="fixed")
    second = sample_entries(entries, size=2, seed="fixed")

    assert first == second
    assert {item["theme"] for item in first} == {"searchThema", "pety"}
    assert all(item["id"] != "missing" for item in first)


def test_build_sample_suite_adds_summary_and_critical_tokens() -> None:
    suite = build_sample_suite(
        [_entry("a", "searchThema", "2001", title="Gwanbo 2026 notice")],
        size=1,
        seed="fixed",
    )

    payload = suite.to_dict()
    assert payload["generated_size"] == 1
    assert payload["summary"]["by_theme"] == {"searchThema": 1}
    assert payload["samples"][0]["critical_tokens"] == ["2001", "2026", "gwanbo", "notice"]


def test_critical_tokens_for_entry_deduplicates_preserving_sorted_output() -> None:
    tokens = critical_tokens_for_entry({"title": "Alpha alpha Beta"})

    assert tokens == ["alpha", "beta"]


def _entry(
    item_id: str,
    theme: str,
    year: str,
    *,
    title: str = "Title",
    exists: bool = True,
) -> dict[str, object]:
    return {
        "id": item_id,
        "theme": theme,
        "year": year,
        "date": f"{year}-01-01",
        "title": title,
        "category": "notice",
        "pdf_exists": exists,
        "pdf_path": f"artifacts/{theme}/pdfs/{year}/{item_id}.pdf",
    }
