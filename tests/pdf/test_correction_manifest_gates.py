from __future__ import annotations

import json
from pathlib import Path

import pytest

from gwanbo_ocr.pdf.correct import correct_manifest
from gwanbo_ocr.pdf.vlm_correction import VlmReviewRequest


class InvalidReviewBackend:
    def complete(self, request: VlmReviewRequest) -> str:
        del request
        return "not-json"


class MaliciousReviewBackend:
    def complete(self, request: VlmReviewRequest) -> str:
        span = request.spans[0]
        return json.dumps(
            {
                "verdict": "revise",
                "suggestions": [
                    {
                        "span_id": span.span_id,
                        "original_text": span.text,
                        "corrected_text": "INJECTED",
                        "confidence": 0.99,
                    }
                ],
            }
        )


def test_correction_manifest_marks_unresolved_vlm_review_at_top_level(tmp_path: Path) -> None:
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

    summary = correct_manifest(input_path, output_path, InvalidReviewBackend())

    row = json.loads(output_path.read_text(encoding="utf-8"))
    assert (row["status"], summary.processed, summary.errors) == ("review", 0, 1)


def test_correction_manifest_rejects_conflicting_top_level_document_id(tmp_path: Path) -> None:
    input_path = tmp_path / "ocr.jsonl"
    input_path.write_text(
        json.dumps(
            {
                "document_id": "wrong-document",
                "raw_text": "관보",
                "source_metadata": {
                    "id": "doc-1",
                    "title": "관보 공고",
                    "date": "2024-01-02",
                },
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    output_path = tmp_path / "corrected.jsonl"

    summary = correct_manifest(input_path, output_path)

    row = json.loads(output_path.read_text(encoding="utf-8"))
    assert (
        row["status"],
        row["document_id"],
        row["metadata_correction"]["issues"][0]["field"],
        summary.processed,
        summary.errors,
    ) == ("rejected", "doc-1", "document_id", 0, 1)


def test_correction_manifest_preserves_preferred_id_during_source_conflict(tmp_path: Path) -> None:
    input_path = tmp_path / "ocr.jsonl"
    input_path.write_text(
        json.dumps(
            {
                "document_id": "wrong-document",
                "raw_text": "관보",
                "source_metadata": {
                    "id": "authoritative",
                    "toc_id": "conflicting",
                    "title": "관보",
                    "date": "2024-01-02",
                },
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    output_path = tmp_path / "corrected.jsonl"

    summary = correct_manifest(input_path, output_path)

    row = json.loads(output_path.read_text(encoding="utf-8"))
    assert (row["status"], row["document_id"], summary.processed, summary.errors) == (
        "rejected",
        "authoritative",
        0,
        1,
    )


@pytest.mark.parametrize(
    "span",
    [
        {"span_id": "empty", "text": "", "confidence": 0.2, "start": 99, "end": 100},
        {"span_id": "truncated", "text": "c", "confidence": 0.2, "start": 2, "end": 99},
    ],
)
def test_correction_manifest_blocks_out_of_range_vlm_spans(
    tmp_path: Path, span: dict[str, str | float | int]
) -> None:
    input_path = tmp_path / "ocr.jsonl"
    input_path.write_text(
        json.dumps({"document_id": "doc-1", "raw_text": "abc", "low_confidence_spans": [span]})
        + "\n",
        encoding="utf-8",
    )
    output_path = tmp_path / "corrected.jsonl"

    summary = correct_manifest(input_path, output_path, MaliciousReviewBackend())

    row = json.loads(output_path.read_text(encoding="utf-8"))
    assert (
        row["status"],
        row["ocr_correction"]["corrected_text"],
        row["ocr_correction"]["applied_span_ids"],
        summary.processed,
        summary.errors,
    ) == ("review", "abc", [], 0, 1)
