from __future__ import annotations

import json
from pathlib import Path

import pytest

from gwanbo_ocr.pdf.io import UnsafePathError
from gwanbo_ocr.pdf.openai_review import (
    ImageResourceLimitError,
    OpenAICompatibleReviewBackend,
    VlmResponseError,
)
from gwanbo_ocr.pdf.vlm_correction import OcrSpan, VlmReviewRequest


class FakeJsonSender:
    def __init__(self, response: bytes) -> None:
        self.response = response
        self.url = ""
        self.headers: dict[str, str] = {}
        self.body = b""
        self.timeout = 0.0

    def post(self, url: str, headers: dict[str, str], body: bytes, timeout: float) -> bytes:
        self.url = url
        self.headers = headers
        self.body = body
        self.timeout = timeout
        return self.response


def _fail_read_bytes(_self: Path) -> bytes:
    raise AssertionError("read_bytes must not run")


def test_openai_backend_sends_text_only_request_without_crop_root(tmp_path: Path) -> None:
    review_json = json.dumps({"verdict": "accept", "suggestions": []})
    sender = FakeJsonSender(
        json.dumps({"choices": [{"message": {"content": review_json}}]}).encode()
    )
    backend = OpenAICompatibleReviewBackend(
        base_url="http://localhost:8000/v1",
        model="qwen-vl",
        api_key="test-key",
        timeout=12,
        sender=sender,
    )
    span = OcrSpan(
        span_id="s1",
        text="공 고",
        confidence=0.4,
        start=3,
        end=6,
        page_index=0,
        bbox=(0.0, 0.0, 10.0, 10.0),
    )

    response = backend.complete(VlmReviewRequest("관보 공 고", (span,)))

    payload = json.loads(sender.body)
    content = payload["messages"][0]["content"]
    assert response == review_json
    assert payload["response_format"] == {"type": "json_object"}
    assert all(item["type"] == "text" for item in content)


def test_openai_backend_sends_span_context_and_crop(tmp_path: Path) -> None:
    crop = tmp_path / "crop.png"
    crop.write_bytes(b"fake-png")
    review_json = json.dumps({"verdict": "accept", "suggestions": []})
    sender = FakeJsonSender(
        json.dumps({"choices": [{"message": {"content": review_json}}]}).encode()
    )
    backend = OpenAICompatibleReviewBackend(
        base_url="http://localhost:8000/v1",
        model="qwen-vl",
        api_key="test-key",
        timeout=12,
        crop_root=tmp_path,
        sender=sender,
    )
    span = OcrSpan(
        span_id="s1",
        text="공 고",
        confidence=0.4,
        start=3,
        end=6,
        page_index=0,
        bbox=(0.0, 0.0, 10.0, 10.0),
        image_path=crop,
    )

    response = backend.complete(VlmReviewRequest("관보 공 고", (span,)))

    payload = json.loads(sender.body)
    content = payload["messages"][0]["content"]
    assert response == review_json
    assert sender.url == "http://localhost:8000/v1/chat/completions"
    assert sender.headers["Authorization"] == "Bearer test-key"
    assert payload["response_format"] == {"type": "json_object"}
    assert any(item["type"] == "image_url" for item in content)


def test_openai_backend_rejects_image_without_crop_root(tmp_path: Path) -> None:
    crop = tmp_path / "crop.png"
    crop.write_bytes(b"fake-png")
    sender = FakeJsonSender(json.dumps({"choices": [{"message": {"content": "{}"}}]}).encode())
    backend = OpenAICompatibleReviewBackend(
        base_url="http://localhost:8000/v1",
        model="qwen-vl",
        sender=sender,
    )
    span = OcrSpan(
        span_id="s1",
        text="공 고",
        confidence=0.4,
        start=3,
        end=6,
        image_path=crop,
    )

    with pytest.raises(UnsafePathError):
        backend.complete(VlmReviewRequest("관보 공 고", (span,)))


def test_openai_backend_rejects_image_path_outside_crop_root_before_read_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    crop_root = tmp_path / "crop"
    crop_root.mkdir()
    outside = tmp_path / "outside.png"
    outside.write_bytes(b"fake-png")
    sender = FakeJsonSender(json.dumps({"choices": [{"message": {"content": "{}"}}]}).encode())
    backend = OpenAICompatibleReviewBackend(
        base_url="http://localhost:8000/v1",
        model="qwen-vl",
        crop_root=crop_root,
        sender=sender,
    )
    span = OcrSpan(
        span_id="s1",
        text="공 고",
        confidence=0.4,
        start=3,
        end=6,
        page_index=0,
        bbox=(0.0, 0.0, 10.0, 10.0),
        image_path=outside,
    )

    monkeypatch.setattr(Path, "read_bytes", _fail_read_bytes)

    with pytest.raises(UnsafePathError):
        backend.complete(VlmReviewRequest("관보 공 고", (span,)))


def test_openai_backend_rejects_symlink_image_path_before_read_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    crop_root = tmp_path / "crop"
    crop_root.mkdir()
    outside = tmp_path / "outside.png"
    outside.write_bytes(b"fake-png")
    link = crop_root / "linked.png"
    link.symlink_to(outside)
    sender = FakeJsonSender(json.dumps({"choices": [{"message": {"content": "{}"}}]}).encode())
    backend = OpenAICompatibleReviewBackend(
        base_url="http://localhost:8000/v1",
        model="qwen-vl",
        crop_root=crop_root,
        sender=sender,
    )
    span = OcrSpan(
        span_id="s1",
        text="공 고",
        confidence=0.4,
        start=3,
        end=6,
        page_index=0,
        bbox=(0.0, 0.0, 10.0, 10.0),
        image_path=link,
    )

    monkeypatch.setattr(Path, "read_bytes", _fail_read_bytes)

    with pytest.raises(UnsafePathError):
        backend.complete(VlmReviewRequest("관보 공 고", (span,)))


def test_openai_backend_rejects_missing_message_content() -> None:
    sender = FakeJsonSender(json.dumps({"choices": []}).encode())
    backend = OpenAICompatibleReviewBackend(
        base_url="http://localhost:8000/v1",
        model="qwen-vl",
        sender=sender,
    )

    try:
        backend.complete(VlmReviewRequest("관보", ()))
    except ValueError as error:
        assert str(error) == "VLM response missing message content"
    else:
        raise AssertionError("missing content response must fail")


def test_openai_backend_rejects_plain_http_for_remote_host() -> None:
    with pytest.raises(VlmResponseError, match="HTTPS"):
        OpenAICompatibleReviewBackend(base_url="http://example.com/v1", model="qwen-vl")


def test_openai_backend_bounds_crop_read_before_allocation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    crop = tmp_path / "oversized.png"
    crop.write_bytes(b"x" * (10 * 1024 * 1024 + 1))
    monkeypatch.setattr(Path, "read_bytes", _fail_read_bytes)
    backend = OpenAICompatibleReviewBackend(
        base_url="http://localhost:8000/v1",
        model="qwen-vl",
        crop_root=tmp_path,
        sender=FakeJsonSender(b"{}"),
    )
    span = OcrSpan(span_id="s", text="a", confidence=0.2, start=0, end=1, image_path=crop)

    with pytest.raises(ImageResourceLimitError):
        backend.complete(VlmReviewRequest("a", (span,)))
