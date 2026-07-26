from __future__ import annotations

import json
from pathlib import Path

from gwanbo_ocr.pdf.correct import correct_manifest
from gwanbo_ocr.pdf.vlm_correction import (
    OcrSpan,
    VlmReviewRequest,
    review_ocr_text,
)


class FakeReviewBackend:
    def __init__(self, response: str) -> None:
        self.response = response
        self.requests: list[VlmReviewRequest] = []

    def complete(self, request: VlmReviewRequest) -> str:
        self.requests.append(request)
        return self.response


def test_vlm_review_applies_high_confidence_low_confidence_span() -> None:
    backend = FakeReviewBackend(
        json.dumps(
            {
                "verdict": "revise",
                "suggestions": [
                    {
                        "span_id": "s1",
                        "original_text": "공 고",
                        "corrected_text": "공고",
                        "confidence": 0.98,
                        "reason": "word spacing",
                    }
                ],
            }
        )
    )
    span = OcrSpan(span_id="s1", text="공 고", confidence=0.4, start=3, end=6)

    result = review_ocr_text("관보 공 고", (span,), backend)

    assert result.corrected_text == "관보 공고"
    assert result.applied_span_ids == ("s1",)
    assert result.requires_review is False
    assert backend.requests[0].spans == (span,)


def test_vlm_review_rejects_protected_numeric_drift() -> None:
    backend = FakeReviewBackend(
        json.dumps(
            {
                "verdict": "revise",
                "suggestions": [
                    {
                        "span_id": "amount",
                        "original_text": "1,234원",
                        "corrected_text": "1,284원",
                        "confidence": 0.99,
                        "reason": "visual guess",
                    }
                ],
            }
        )
    )
    span = OcrSpan(
        span_id="amount",
        text="1,234원",
        confidence=0.2,
        start=3,
        end=9,
        page_index=0,
        bbox=(0.0, 0.0, 10.0, 10.0),
        image_path=Path("crop.png"),
    )

    result = review_ocr_text("금액 1,234원", (span,), backend)

    assert result.corrected_text == "금액 1,234원"
    assert result.applied_span_ids == ()
    assert result.requires_review is True
    assert "protected_value_drift:amount" in result.issues


def test_vlm_review_escalates_malformed_response() -> None:
    span = OcrSpan(span_id="s1", text="공 고", confidence=0.4, start=3, end=6)

    result = review_ocr_text("관보 공 고", (span,), FakeReviewBackend("not-json"))

    assert result.corrected_text == "관보 공 고"
    assert result.requires_review is True
    assert result.issues == ("invalid_vlm_response",)


def test_vlm_review_skips_high_confidence_spans() -> None:
    backend = FakeReviewBackend("should not be called")
    span = OcrSpan(span_id="s1", text="공고", confidence=0.99, start=3, end=5)

    result = review_ocr_text("관보 공고", (span,), backend)

    assert result.corrected_text == "관보 공고"
    assert result.requires_review is False
    assert backend.requests == []


def test_vlm_review_blocks_overlapping_spans_before_backend_call() -> None:
    backend = FakeReviewBackend(
        json.dumps(
            {
                "verdict": "revise",
                "suggestions": [
                    {
                        "span_id": "a",
                        "original_text": "관보 공",
                        "corrected_text": "관보공",
                        "confidence": 0.99,
                    },
                    {
                        "span_id": "b",
                        "original_text": " 공고",
                        "corrected_text": "공고",
                        "confidence": 0.99,
                    },
                ],
            }
        )
    )
    spans = (
        OcrSpan(span_id="a", text="관보 공", confidence=0.2, start=0, end=4),
        OcrSpan(span_id="b", text=" 공고", confidence=0.2, start=2, end=5),
    )

    result = review_ocr_text("관보 공고", spans, backend)

    assert (
        result.corrected_text,
        result.applied_span_ids,
        result.requires_review,
        backend.requests,
    ) == ("관보 공고", (), True, [])


def test_correction_manifest_applies_gated_vlm_review(tmp_path: Path) -> None:
    input_path = tmp_path / "ocr.jsonl"
    input_path.write_text(
        json.dumps(
            {
                "document_id": "doc-1",
                "raw_text": "관보 공 고",
                "low_confidence_spans": [
                    {"span_id": "s1", "text": "공 고", "confidence": 0.4, "start": 3, "end": 6}
                ],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    output_path = tmp_path / "corrected.jsonl"
    backend = FakeReviewBackend(
        json.dumps(
            {
                "verdict": "revise",
                "suggestions": [
                    {
                        "span_id": "s1",
                        "original_text": "공 고",
                        "corrected_text": "공고",
                        "confidence": 0.98,
                    }
                ],
            }
        )
    )

    summary = correct_manifest(input_path, output_path, backend)

    row = json.loads(output_path.read_text(encoding="utf-8"))
    assert summary.processed == 1
    assert row["raw_text"] == "관보 공 고"
    assert row["ocr_correction"]["corrected_text"] == "관보 공고"
    assert row["ocr_correction"]["applied_span_ids"] == ["s1"]
