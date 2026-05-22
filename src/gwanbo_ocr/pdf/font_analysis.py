"""Font size extraction helpers for PDF rendering resolution tier selection."""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

try:
    import pdfplumber  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - default in the kata environment.
    pdfplumber = None  # type: ignore[assignment]


class ResolutionTier(StrEnum):
    """Rendering resolution tier inferred from small-font prevalence."""

    HIGH = "high"
    STANDARD = "standard"
    LOW = "low"


@dataclass(frozen=True)
class FontStats:
    """Per-page font size statistics derived from pdfplumber character metadata."""

    min_size: float
    p10_size: float
    median_size: float
    mean_size: float
    max_size: float
    char_count: int
    has_native_text: bool


EMPTY_FONT_STATS = FontStats(
    min_size=0.0,
    p10_size=0.0,
    median_size=0.0,
    mean_size=0.0,
    max_size=0.0,
    char_count=0,
    has_native_text=False,
)


def extract_page_font_stats(page: Any) -> FontStats:
    """Extract font size statistics from a pdfplumber page's character list."""
    sizes = sorted(_char_font_sizes(page))
    if not sizes:
        return EMPTY_FONT_STATS

    return FontStats(
        min_size=sizes[0],
        p10_size=_percentile(sizes, 10.0),
        median_size=statistics.median(sizes),
        mean_size=statistics.fmean(sizes),
        max_size=sizes[-1],
        char_count=len(sizes),
        has_native_text=True,
    )


def extract_pdf_font_stats(
    pdf_path: Path | str, max_pages: int | None = None
) -> dict[int, FontStats]:
    """Extract per-page font statistics for a PDF using pdfplumber when available."""
    if pdfplumber is None:
        return {}

    path = Path(pdf_path)
    stats: dict[int, FontStats] = {}
    try:
        with pdfplumber.open(str(path)) as pdf:
            page_count = len(pdf.pages)
            pages_to_scan = page_count if max_pages is None else min(page_count, max_pages)
            for page_index, page in enumerate(pdf.pages[:pages_to_scan]):
                stats[page_index] = extract_page_font_stats(page)
    except Exception:  # noqa: BLE001
        return {}
    return stats


def classify_font_tier(font_stats: FontStats) -> ResolutionTier:
    """Classify rendering resolution tier from p10 font size."""
    if font_stats.p10_size < 7.0:
        return ResolutionTier.HIGH
    if font_stats.p10_size < 10.0:
        return ResolutionTier.STANDARD
    return ResolutionTier.LOW


def _char_font_sizes(page: Any) -> list[float]:
    sizes: list[float] = []
    for char in getattr(page, "chars", []) or []:
        if not isinstance(char, dict):
            continue
        size = char.get("size")
        if isinstance(size, int | float):
            sizes.append(float(size))
    return sizes


def _percentile(sorted_data: list[float], p: float) -> float:
    if not sorted_data:
        return 0.0
    k = (len(sorted_data) - 1) * (p / 100.0)
    f = int(k)
    c = f + 1
    if c >= len(sorted_data):
        return sorted_data[f]
    return sorted_data[f] + (k - f) * (sorted_data[c] - sorted_data[f])


__all__ = [
    "FontStats",
    "ResolutionTier",
    "classify_font_tier",
    "extract_page_font_stats",
    "extract_pdf_font_stats",
]
