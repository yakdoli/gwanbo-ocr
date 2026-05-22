from __future__ import annotations

# pyright: reportMissingImports=false
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from gwanbo_ocr.pdf.font_analysis import (  # noqa: E402
    FontStats,
    ResolutionTier,
    classify_font_tier,
    extract_page_font_stats,
    extract_pdf_font_stats,
)


def _font_stats(p10_size: float, *, has_native_text: bool = True) -> FontStats:
    return FontStats(
        min_size=p10_size,
        p10_size=p10_size,
        median_size=p10_size,
        mean_size=p10_size,
        max_size=p10_size,
        char_count=100 if has_native_text else 0,
        has_native_text=has_native_text,
    )


def test_extract_page_font_stats_basic() -> None:
    page = MagicMock()
    page.chars = [{"size": 6}, {"size": 8.0}, {"size": 10}, {"size": 12.0}]

    stats = extract_page_font_stats(page)

    assert stats.min_size == 6.0
    assert stats.p10_size == 6.6
    assert stats.median_size == 9.0
    assert stats.mean_size == 9.0
    assert stats.max_size == 12.0
    assert stats.char_count == 4
    assert stats.has_native_text is True


def test_extract_page_font_stats_no_chars() -> None:
    page = MagicMock()
    page.chars = []

    stats = extract_page_font_stats(page)

    assert stats.min_size == 0.0
    assert stats.p10_size == 0.0
    assert stats.median_size == 0.0
    assert stats.mean_size == 0.0
    assert stats.max_size == 0.0
    assert stats.char_count == 0
    assert stats.has_native_text is False


def test_extract_page_font_stats_single_char() -> None:
    page = MagicMock()
    page.chars = [{"size": 9.5}]

    stats = extract_page_font_stats(page)

    assert stats.min_size == 9.5
    assert stats.p10_size == 9.5
    assert stats.median_size == 9.5
    assert stats.mean_size == 9.5
    assert stats.max_size == 9.5
    assert stats.char_count == 1
    assert stats.has_native_text is True


def test_classify_font_tier_high() -> None:
    assert classify_font_tier(_font_stats(6.9)) == ResolutionTier.HIGH


def test_classify_font_tier_standard() -> None:
    assert classify_font_tier(_font_stats(8.0)) == ResolutionTier.STANDARD


def test_classify_font_tier_low() -> None:
    assert classify_font_tier(_font_stats(10.0)) == ResolutionTier.LOW


def test_classify_font_tier_no_text() -> None:
    assert classify_font_tier(_font_stats(8.0, has_native_text=False)) == ResolutionTier.STANDARD


def test_extract_pdf_font_stats() -> None:
    page_one = MagicMock()
    page_one.chars = [{"size": 6.0}, {"size": 8.0}]
    page_two = MagicMock()
    page_two.chars = [{"size": 12.0}]

    fake_pdf = MagicMock()
    fake_pdf.pages = [page_one, page_two]
    fake_context = MagicMock()
    fake_context.__enter__.return_value = fake_pdf
    fake_context.__exit__.return_value = None
    fake_pdfplumber = MagicMock()
    fake_pdfplumber.open.return_value = fake_context

    with patch("gwanbo_ocr.pdf.font_analysis.pdfplumber", fake_pdfplumber):
        stats = extract_pdf_font_stats("dummy.pdf")

    assert set(stats) == {0, 1}
    assert stats[0].char_count == 2
    assert stats[0].min_size == 6.0
    assert stats[0].max_size == 8.0
    assert stats[1].char_count == 1
    assert stats[1].p10_size == 12.0
    fake_pdfplumber.open.assert_called_once_with("dummy.pdf")
