"""Dual-strategy PDF page resolution tier selection.

Text PDFs with a native text layer use font statistics from font analysis to
choose a rendering tier. Image or scanned PDFs use a quick low-resolution OCR
probe supplied by the caller to estimate text density and page complexity. This
module only scores already-collected signals; it does not render pages or run
OCR itself.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from gwanbo_ocr.pdf.font_analysis import FontStats, ResolutionTier, classify_font_tier

DENSE_SMALL_PAGE_AREA_THRESHOLD = 500_000
DENSE_SMALL_PAGE_TEXT_THRESHOLD = 2_000
STANDARD_TEXT_THRESHOLD = 500
STANDARD_AVG_LINE_LENGTH_THRESHOLD = 40
MANY_LINES_PER_PIXEL_THRESHOLD = 0.035


RESOLUTION_TIERS: dict[ResolutionTier, dict[str, Any]] = {
    ResolutionTier.HIGH: {
        "max_long_edge": 2400,
        "dpi": 300,
        "recommended_model": "chandra_ocr_2_lemonade_vllm",
        "description": "Small text (footnotes, annotations)",
    },
    ResolutionTier.STANDARD: {
        "max_long_edge": 1540,
        "dpi": 200,
        "recommended_model": "lightonocr2_1b_lemonade_vllm",
        "description": "Normal body text",
    },
    ResolutionTier.LOW: {
        "max_long_edge": 1088,
        "dpi": 150,
        "recommended_model": "bizonai_ocr_lemonade_vllm",
        "description": "Large text (headers, titles)",
    },
}


@dataclass(frozen=True)
class PageResolution:
    page_index: int
    tier: ResolutionTier
    method: str
    max_long_edge: int
    dpi: int
    recommended_model: str
    confidence: float
    reason: str


@dataclass
class ResolutionPlan:
    pdf_path: str
    pages: list[PageResolution]
    default_tier: ResolutionTier
    generated_at: str


def determine_resolution_tier_font(font_stats: FontStats) -> ResolutionTier:
    """Determine a page resolution tier from native PDF font statistics."""
    if not font_stats.has_native_text:
        return ResolutionTier.STANDARD
    return classify_font_tier(font_stats)


def determine_resolution_tier_ocr_probe(
    ocr_text: str, image_dimensions: tuple[int, int]
) -> ResolutionTier:
    """Determine a page resolution tier from a low-resolution OCR probe.

    The thresholds are initial tunable constants. Dense text in a small rendered
    probe suggests small glyphs and therefore a higher resolution OCR pass.
    """
    width, height = image_dimensions
    image_area = max(width, 0) * max(height, 0)
    text_length = len(ocr_text)
    lines = text_lines(ocr_text)
    avg_line_length = average_line_length(lines)
    line_density = ratio(len(lines), max(height, 1))

    if (
        image_area < DENSE_SMALL_PAGE_AREA_THRESHOLD
        and text_length > DENSE_SMALL_PAGE_TEXT_THRESHOLD
    ):
        return ResolutionTier.HIGH
    if line_density > MANY_LINES_PER_PIXEL_THRESHOLD and text_length > STANDARD_TEXT_THRESHOLD:
        return ResolutionTier.HIGH
    if (
        text_length > STANDARD_TEXT_THRESHOLD
        and avg_line_length > STANDARD_AVG_LINE_LENGTH_THRESHOLD
    ):
        return ResolutionTier.STANDARD
    return ResolutionTier.LOW


def build_resolution_plan(
    pdf_path: Path | str,
    font_stats_per_page: Mapping[int, FontStats] | Sequence[FontStats] | None = None,
    ocr_probe_results: Mapping[int, Any] | Sequence[Any] | None = None,
) -> ResolutionPlan:
    """Build per-page resolution assignments from font stats or OCR probe results."""
    normalized_font_stats = normalize_page_mapping(font_stats_per_page)
    normalized_ocr_results = normalize_page_mapping(ocr_probe_results)
    page_indices = sorted(set(normalized_font_stats) | set(normalized_ocr_results))
    pages: list[PageResolution] = []

    for page_index in page_indices:
        font_stats = normalized_font_stats.get(page_index)
        if isinstance(font_stats, FontStats):
            pages.append(page_resolution_from_font(page_index, font_stats))
            continue

        if page_index in normalized_ocr_results:
            ocr_text, image_dimensions = normalize_ocr_probe(normalized_ocr_results[page_index])
            pages.append(page_resolution_from_ocr_probe(page_index, ocr_text, image_dimensions))
            continue

        pages.append(page_resolution_default(page_index))

    return ResolutionPlan(
        pdf_path=str(Path(pdf_path)),
        pages=pages,
        default_tier=ResolutionTier.STANDARD,
        generated_at=datetime.now().isoformat(),
    )


def page_resolution_from_font(page_index: int, font_stats: FontStats) -> PageResolution:
    tier = determine_resolution_tier_font(font_stats)
    config = RESOLUTION_TIERS[tier]
    return PageResolution(
        page_index=page_index,
        tier=tier,
        method="font_analysis",
        max_long_edge=int(config["max_long_edge"]),
        dpi=int(config["dpi"]),
        recommended_model=str(config["recommended_model"]),
        confidence=font_confidence(font_stats, tier),
        reason=font_reason(font_stats, tier),
    )


def page_resolution_from_ocr_probe(
    page_index: int, ocr_text: str, image_dimensions: tuple[int, int]
) -> PageResolution:
    tier = determine_resolution_tier_ocr_probe(ocr_text, image_dimensions)
    config = RESOLUTION_TIERS[tier]
    lines = text_lines(ocr_text)
    return PageResolution(
        page_index=page_index,
        tier=tier,
        method="ocr_probe",
        max_long_edge=int(config["max_long_edge"]),
        dpi=int(config["dpi"]),
        recommended_model=str(config["recommended_model"]),
        confidence=ocr_probe_confidence(ocr_text, image_dimensions),
        reason=ocr_probe_reason(ocr_text, image_dimensions, lines, tier),
    )


def page_resolution_default(page_index: int) -> PageResolution:
    tier = ResolutionTier.STANDARD
    config = RESOLUTION_TIERS[tier]
    return PageResolution(
        page_index=page_index,
        tier=tier,
        method="font_analysis",
        max_long_edge=int(config["max_long_edge"]),
        dpi=int(config["dpi"]),
        recommended_model=str(config["recommended_model"]),
        confidence=0.0,
        reason="No font stats or OCR probe result was available; using default tier",
    )


def font_confidence(font_stats: FontStats, tier: ResolutionTier) -> float:
    if not font_stats.has_native_text:
        return 0.2
    if font_stats.char_count < 20:
        return 0.45
    if tier == ResolutionTier.HIGH and font_stats.p10_size < 6.5:
        return 0.9
    if tier == ResolutionTier.STANDARD and 7.5 <= font_stats.p10_size < 9.5:
        return 0.85
    if tier == ResolutionTier.LOW and font_stats.p10_size >= 11.0:
        return 0.85
    return 0.7


def font_reason(font_stats: FontStats, tier: ResolutionTier) -> str:
    if not font_stats.has_native_text:
        return "No native text was detected in font stats; using standard fallback tier"
    return (
        f"Font analysis selected {tier.value} from p10 font size "
        f"{font_stats.p10_size:.2f} across {font_stats.char_count} characters"
    )


def ocr_probe_confidence(ocr_text: str, image_dimensions: tuple[int, int]) -> float:
    width, height = image_dimensions
    if not ocr_text.strip() or width <= 0 or height <= 0:
        return 0.25
    if len(ocr_text) > DENSE_SMALL_PAGE_TEXT_THRESHOLD:
        return 0.8
    if len(ocr_text) > STANDARD_TEXT_THRESHOLD:
        return 0.65
    return 0.45


def ocr_probe_reason(
    ocr_text: str,
    image_dimensions: tuple[int, int],
    lines: list[str],
    tier: ResolutionTier,
) -> str:
    width, height = image_dimensions
    image_area = max(width, 0) * max(height, 0)
    return (
        f"OCR probe selected {tier.value} from {len(ocr_text)} characters, "
        f"{len(lines)} lines, average line length {average_line_length(lines):.1f}, "
        f"and image area {image_area} pixels"
    )


def normalize_page_mapping(values: Mapping[int, Any] | Sequence[Any] | None) -> dict[int, Any]:
    if values is None:
        return {}
    if isinstance(values, Mapping):
        return {int(page_index): value for page_index, value in values.items()}
    return {page_index: value for page_index, value in enumerate(values)}


def normalize_ocr_probe(value: Any) -> tuple[str, tuple[int, int]]:
    if isinstance(value, str):
        return value, (0, 0)
    if isinstance(value, Mapping):
        text = str(value.get("ocr_text") or value.get("text") or "")
        dimensions = value.get("image_dimensions") or value.get("dimensions") or (0, 0)
        return text, normalize_dimensions(dimensions)
    if isinstance(value, tuple) and len(value) == 2:
        text, dimensions = value
        return str(text), normalize_dimensions(dimensions)
    return str(value or ""), (0, 0)


def normalize_dimensions(value: Any) -> tuple[int, int]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes) or len(value) < 2:
        return (0, 0)
    width, height = value[0], value[1]
    if not isinstance(width, int | float) or not isinstance(height, int | float):
        return (0, 0)
    return (int(width), int(height))


def text_lines(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]


def average_line_length(lines: list[str]) -> float:
    if not lines:
        return 0.0
    return sum(len(line) for line in lines) / len(lines)


def ratio(numerator: int | float, denominator: int | float) -> float:
    if not denominator:
        return 0.0
    return numerator / denominator


__all__ = [
    "RESOLUTION_TIERS",
    "FontStats",
    "PageResolution",
    "ResolutionPlan",
    "ResolutionTier",
    "build_resolution_plan",
    "determine_resolution_tier_font",
    "determine_resolution_tier_ocr_probe",
]
