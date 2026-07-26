from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from gwanbo_ocr.pdf.correct import OcrInputRow
from gwanbo_ocr.pdf.io import iter_jsonl_records
from gwanbo_ocr.pdf.vlm_correction import VlmReviewResponse

MAX_OCR_TEXT_CHARS = 2 * 1024 * 1024
MAX_VLM_SPANS = 1_000
MAX_VLM_SUGGESTIONS = 1_000


def test_jsonl_reader_rejects_oversized_line_and_continues(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_bytes(b'{"text":"' + b"x" * 100 + b'"}\n{"ok":true}\n')

    rows = list(iter_jsonl_records(manifest, max_line_bytes=32))

    assert (
        rows[0][0],
        rows[0][1]["status"],
        rows[0][1]["error"],
        rows[1],
    ) == (1, "error", "jsonl_line_too_large", (2, {"ok": True}))


def test_ocr_input_rejects_oversized_text() -> None:
    with pytest.raises(ValidationError):
        OcrInputRow.model_validate({"raw_text": "x" * (MAX_OCR_TEXT_CHARS + 1)})


def test_ocr_input_rejects_excessive_spans() -> None:
    span = {"span_id": "s", "text": "x", "confidence": 0.2, "start": 0, "end": 1}

    with pytest.raises(ValidationError):
        OcrInputRow.model_validate(
            {"raw_text": "x", "low_confidence_spans": [span] * (MAX_VLM_SPANS + 1)}
        )


def test_vlm_response_rejects_excessive_suggestions() -> None:
    suggestion = {
        "span_id": "s",
        "original_text": "x",
        "corrected_text": "y",
        "confidence": 0.99,
    }
    payload = json.dumps(
        {"verdict": "revise", "suggestions": [suggestion] * (MAX_VLM_SUGGESTIONS + 1)}
    )

    with pytest.raises(ValidationError):
        VlmReviewResponse.model_validate_json(payload)
