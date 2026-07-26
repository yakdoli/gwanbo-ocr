from __future__ import annotations

import json

from gwanbo_ocr.pdf.vlm_correction import OcrSpan, VlmReviewRequest, review_ocr_text


class FakeReviewBackend:
    def __init__(self, response: str) -> None:
        self.response = response
        self.requests: list[VlmReviewRequest] = []

    def complete(self, request: VlmReviewRequest) -> str:
        self.requests.append(request)
        return self.response


def test_vlm_review_rejects_duplicate_suggestions_for_one_span() -> None:
    backend = FakeReviewBackend(
        json.dumps(
            {
                "verdict": "revise",
                "suggestions": [
                    {
                        "span_id": "s",
                        "original_text": "a",
                        "corrected_text": "X",
                        "confidence": 0.99,
                    },
                    {
                        "span_id": "s",
                        "original_text": "a",
                        "corrected_text": "Y",
                        "confidence": 0.99,
                    },
                ],
            }
        )
    )
    span = OcrSpan(span_id="s", text="a", confidence=0.2, start=0, end=1)

    result = review_ocr_text("abc", (span,), backend)

    assert (
        result.corrected_text,
        result.applied_span_ids,
        result.issues,
        result.requires_review,
    ) == ("abc", (), ("duplicate_suggestion:s",), True)


def test_vlm_review_blocks_offsets_shifted_by_normalization() -> None:
    backend = FakeReviewBackend(
        json.dumps(
            {
                "verdict": "revise",
                "suggestions": [
                    {
                        "span_id": "s",
                        "original_text": "aa",
                        "corrected_text": "ZZ",
                        "confidence": 0.99,
                    }
                ],
            }
        )
    )
    span = OcrSpan(span_id="s", text="aa", confidence=0.2, start=3, end=5)

    result = review_ocr_text("x  aaaa", (span,), backend)

    assert (
        result.corrected_text,
        result.applied_span_ids,
        result.issues,
        result.requires_review,
        backend.requests,
    ) == ("x aaaa", (), ("unstable_span_offsets:s",), True, [])


def test_vlm_review_rejects_duplicate_input_ids_before_filtering() -> None:
    backend = FakeReviewBackend("should not be called")
    spans = (
        OcrSpan(span_id="dup", text="a", confidence=0.2, start=0, end=1),
        OcrSpan(span_id="dup", text="wrong", confidence=0.2, start=1, end=2),
    )

    result = review_ocr_text("abc", spans, backend)

    assert (result.applied_span_ids, result.issues, result.requires_review, backend.requests) == (
        (),
        ("duplicate_span:dup",),
        True,
        [],
    )
