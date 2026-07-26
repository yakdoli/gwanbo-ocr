from __future__ import annotations

from gwanbo_ocr.pdf.correction import CorrectionKind, correct_ocr_text


def test_correct_ocr_text_normalizes_safely_and_preserves_protected_values() -> None:
    raw_text = "관보\r\n재산  1,234,567원\u200b 2026-07-23"

    result = correct_ocr_text(raw_text)

    assert result.raw_text == raw_text
    assert result.corrected_text == "관보\n재산 1,234,567원 2026-07-23"
    assert result.protected_values == ("1,234,567", "2026-07-23")
    assert {change.kind for change in result.changes} == {
        CorrectionKind.UNICODE_NFC,
        CorrectionKind.LINE_ENDINGS,
        CorrectionKind.ZERO_WIDTH,
        CorrectionKind.HORIZONTAL_SPACE,
    }
    assert result.requires_review is False


def test_correct_ocr_text_leaves_clean_text_unchanged() -> None:
    raw_text = "대한민국 관보 제123호"

    result = correct_ocr_text(raw_text)

    assert result.corrected_text == raw_text
    assert result.changes == ()
    assert result.protected_values == ("123",)
