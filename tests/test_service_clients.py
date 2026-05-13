from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gwanbo_ocr.services import MarkItDownServiceClient, PaddleOcrServiceClient


def test_markitdown_service_client_posts_convert_path(monkeypatch: Any) -> None:
    captured: dict[str, Any] = {}

    def fake_post_json(url: str, payload: dict[str, Any], *, timeout: float) -> dict[str, Any]:
        captured.update({"url": url, "payload": payload, "timeout": timeout})
        return {"status": "ok", "markdown_content": "converted"}

    monkeypatch.setattr("gwanbo_ocr.services._post_json", fake_post_json)

    client = MarkItDownServiceClient("http://127.0.0.1:8081", timeout=12)
    result = client.convert_path(
        Path("/data/doc.pdf"),
        mode="ocr-llm",
        llm_model="Qwen/Qwen3.6-35B-A3B-FP8",
    )

    assert result["markdown_content"] == "converted"
    assert captured["url"] == "http://127.0.0.1:8081/convert/path"
    assert captured["payload"]["file_path"] == "/data/doc.pdf"
    assert captured["payload"]["mode"] == "ocr-llm"
    assert captured["payload"]["llm_model"] == "Qwen/Qwen3.6-35B-A3B-FP8"
    assert captured["timeout"] == 12


def test_paddle_service_client_returns_transcription_result(monkeypatch: Any) -> None:
    captured: dict[str, Any] = {}

    def fake_post_json(url: str, payload: dict[str, Any], *, timeout: float) -> dict[str, Any]:
        captured.update({"url": url, "payload": payload, "timeout": timeout})
        return {"status": "ok", "text": "관보 OCR", "blocks": [{"text": "관보 OCR"}]}

    monkeypatch.setattr("gwanbo_ocr.services._post_json", fake_post_json)

    client = PaddleOcrServiceClient("http://127.0.0.1:8082", timeout=30)
    result = client.transcribe_classic(Path("/images/page.png"), lang="korean", page_number=2)

    assert result.text == "관보 OCR"
    assert result.backend == "paddleocr_service"
    assert captured["url"] == "http://127.0.0.1:8082/ocr/classic"
    assert captured["payload"]["image_path"] == "/images/page.png"
    assert captured["payload"]["lang"] == "korean"
