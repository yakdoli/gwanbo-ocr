from __future__ import annotations

import json
from pathlib import Path

import pytest

from gwanbo_ocr.pdf.openai_review import OpenAICompatibleReviewBackend, VlmResponseError
from gwanbo_ocr.pdf.vlm_correction import OcrSpan, VlmReviewRequest


class SuccessfulSender:
    def post(self, url: str, headers: dict[str, str], body: bytes, timeout: float) -> bytes:
        del url, headers, body, timeout
        content = json.dumps({"verdict": "accept", "suggestions": []})
        return json.dumps({"choices": [{"message": {"content": content}}]}).encode()


def test_openai_backend_rejects_oversized_ocr_text() -> None:
    backend = OpenAICompatibleReviewBackend(
        base_url="http://localhost:8000/v1", model="vlm", sender=SuccessfulSender()
    )

    with pytest.raises(VlmResponseError, match="text"):
        backend.complete(VlmReviewRequest("x" * (2 * 1024 * 1024 + 1), ()))


def test_openai_backend_rejects_excessive_crop_count(tmp_path: Path) -> None:
    spans = []
    for index in range(17):
        crop = tmp_path / f"{index}.png"
        crop.write_bytes(b"x")
        spans.append(
            OcrSpan(
                span_id=str(index),
                text="x",
                confidence=0.2,
                start=0,
                end=1,
                image_path=crop,
            )
        )
    backend = OpenAICompatibleReviewBackend(
        base_url="http://localhost:8000/v1",
        model="vlm",
        crop_root=tmp_path,
        sender=SuccessfulSender(),
    )

    with pytest.raises(VlmResponseError, match="crop"):
        backend.complete(VlmReviewRequest("x", tuple(spans)))


def test_openai_backend_rejects_aggregate_crop_payload(tmp_path: Path) -> None:
    spans = []
    for index in range(13):
        crop = tmp_path / f"{index}.png"
        crop.write_bytes(b"x" * 1024 * 1024)
        spans.append(
            OcrSpan(
                span_id=str(index),
                text="x",
                confidence=0.2,
                start=0,
                end=1,
                image_path=crop,
            )
        )
    backend = OpenAICompatibleReviewBackend(
        base_url="http://localhost:8000/v1",
        model="vlm",
        crop_root=tmp_path,
        sender=SuccessfulSender(),
    )

    with pytest.raises(VlmResponseError, match="request"):
        backend.complete(VlmReviewRequest("x", tuple(spans)))
