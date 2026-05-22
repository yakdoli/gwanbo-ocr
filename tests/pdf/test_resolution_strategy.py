from __future__ import annotations

# pyright: reportMissingImports=false
import sys
from pathlib import Path
from typing import cast

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from gwanbo_ocr.pdf.font_analysis import FontStats, ResolutionTier  # noqa: E402
from gwanbo_ocr.pdf.resolution_strategy import (  # noqa: E402
    RESOLUTION_TIERS,
    build_resolution_plan,
    determine_resolution_tier_font,
    determine_resolution_tier_ocr_probe,
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


def test_resolution_tiers_constants() -> None:
    assert set(RESOLUTION_TIERS) == {
        ResolutionTier.HIGH,
        ResolutionTier.STANDARD,
        ResolutionTier.LOW,
    }
    for tier in (ResolutionTier.HIGH, ResolutionTier.STANDARD, ResolutionTier.LOW):
        assert set(RESOLUTION_TIERS[tier]) == {
            "max_long_edge",
            "dpi",
            "recommended_model",
            "description",
        }


def test_determine_tier_font_high() -> None:
    assert determine_resolution_tier_font(_font_stats(6.0)) == ResolutionTier.HIGH


def test_determine_tier_font_standard() -> None:
    assert determine_resolution_tier_font(_font_stats(8.5)) == ResolutionTier.STANDARD


def test_determine_tier_ocr_probe_dense_text() -> None:
    dense_text = "\n".join(
        f"Dense OCR line {line_number:03d} with enough words for small glyphs."
        for line_number in range(60)
    )

    tier = determine_resolution_tier_ocr_probe(dense_text, (600, 700))

    assert tier == ResolutionTier.HIGH


def test_determine_tier_ocr_probe_sparse_text() -> None:
    sparse_text = "Title\nShort note"

    tier = determine_resolution_tier_ocr_probe(sparse_text, (1600, 2200))

    assert tier == ResolutionTier.LOW


def test_build_resolution_plan_with_font_stats() -> None:
    plan = build_resolution_plan(
        "document.pdf",
        font_stats_per_page={
            0: _font_stats(6.5),
            1: _font_stats(8.0),
            2: _font_stats(11.0),
        },
    )

    assert plan.pdf_path == "document.pdf"
    assert plan.default_tier == ResolutionTier.STANDARD
    assert [page.page_index for page in plan.pages] == [0, 1, 2]
    assert [page.tier for page in plan.pages] == [
        ResolutionTier.HIGH,
        ResolutionTier.STANDARD,
        ResolutionTier.LOW,
    ]
    assert all(page.method == "font_analysis" for page in plan.pages)
    assert plan.pages[0].dpi == RESOLUTION_TIERS[ResolutionTier.HIGH]["dpi"]
    assert plan.pages[1].max_long_edge == RESOLUTION_TIERS[ResolutionTier.STANDARD]["max_long_edge"]
    assert (
        plan.pages[2].recommended_model == RESOLUTION_TIERS[ResolutionTier.LOW]["recommended_model"]
    )


def test_build_resolution_plan_mixed() -> None:
    plan = build_resolution_plan(
        "mixed.pdf",
        font_stats_per_page=cast("dict[int, FontStats]", {0: _font_stats(6.0), 1: None}),
        ocr_probe_results={
            2: ("Heading\nSubheading", (1200, 1600)),
        },
    )

    assert [page.page_index for page in plan.pages] == [0, 1, 2]
    assert [page.tier for page in plan.pages] == [
        ResolutionTier.HIGH,
        ResolutionTier.STANDARD,
        ResolutionTier.LOW,
    ]
    assert [page.method for page in plan.pages] == ["font_analysis", "font_analysis", "ocr_probe"]


def test_build_resolution_plan_no_data() -> None:
    plan = build_resolution_plan(
        "no-data.pdf",
        font_stats_per_page=[_font_stats(0.0, has_native_text=False) for _ in range(3)],
    )

    assert [page.page_index for page in plan.pages] == [0, 1, 2]
    assert [page.tier for page in plan.pages] == [
        ResolutionTier.STANDARD,
        ResolutionTier.STANDARD,
        ResolutionTier.STANDARD,
    ]
    assert all(page.confidence == 0.2 for page in plan.pages)
