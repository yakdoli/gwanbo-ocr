from __future__ import annotations

import sys
import urllib.error
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gwanbo_ocr.runners.paddle_service import (
    PaddleOcrServiceRunner,
    PaddleOcrVlServiceRunner,
    _image_path,
)
from gwanbo_ocr.services import (
    MarkItDownServiceClient,
    PaddleOcrServiceClient,
    ServiceClientError,
    _request_json,
    _result_from_service_payload,
)


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


# ---------------------------------------------------------------------------
# PaddleOcrServiceRunner / PaddleOcrVlServiceRunner
# ---------------------------------------------------------------------------


def test_paddle_ocr_service_runner_transcribes_classic(monkeypatch: Any) -> None:
    captured: dict[str, Any] = {}

    def fake_post_json(url: str, payload: dict[str, Any], *, timeout: float) -> dict[str, Any]:
        captured.update({"url": url, "payload": payload, "timeout": timeout})
        return {"status": "ok", "text": "classic OCR", "blocks": []}

    monkeypatch.setattr("gwanbo_ocr.services._post_json", fake_post_json)

    runner = PaddleOcrServiceRunner("http://127.0.0.1:8082", lang="korean", timeout=60)
    result = runner.transcribe(Path("/app/page.png"), page_number=3)

    assert result.text == "classic OCR"
    assert result.backend == "paddleocr_service"
    assert captured["url"] == "http://127.0.0.1:8082/ocr/classic"
    assert captured["payload"]["lang"] == "korean"
    assert captured["payload"]["page_number"] == 3
    assert captured["timeout"] == 60


def test_paddle_ocr_vl_service_runner_transcribes_vl(monkeypatch: Any) -> None:
    captured: dict[str, Any] = {}

    def fake_post_json(url: str, payload: dict[str, Any], *, timeout: float) -> dict[str, Any]:
        captured.update({"url": url, "payload": payload, "timeout": timeout})
        return {"status": "ok", "text": "VL OCR result", "markdown": "**VL**"}

    monkeypatch.setattr("gwanbo_ocr.services._post_json", fake_post_json)

    runner = PaddleOcrVlServiceRunner(
        "http://127.0.0.1:8085",
        pipeline_version="v2.0",
        vl_rec_backend="openai",
        vl_rec_server_url="http://openai:8000/v1",
        vl_rec_api_model_name="gpt-4o",
        vl_rec_api_key="sk-test",
        timeout=90,
    )
    result = runner.transcribe(Path("/images/test.png"))

    assert result.text == "VL OCR result"
    assert result.backend == "paddleocr_vl_service"
    assert captured["url"] == "http://127.0.0.1:8085/ocr/vl"
    assert captured["payload"]["pipeline_version"] == "v2.0"
    assert captured["payload"]["vl_rec_backend"] == "openai"
    assert captured["payload"]["vl_rec_server_url"] == "http://openai:8000/v1"
    assert captured["payload"]["vl_rec_api_model_name"] == "gpt-4o"
    assert captured["payload"]["vl_rec_api_key"] == "sk-test"
    assert captured["timeout"] == 90


def test_image_path_rejects_non_path_str() -> None:
    with pytest.raises(TypeError):
        _image_path(b"raw bytes")


# ---------------------------------------------------------------------------
# Service client error paths
# ---------------------------------------------------------------------------


def test_request_json_raises_on_http_error(monkeypatch: Any) -> None:
    import email.message

    hdrs = email.message.Message()
    exc = urllib.error.HTTPError(
        "http://127.0.0.1:8080/test", 500, "Internal Server Error", hdrs, None
    )

    monkeypatch.setattr(
        "gwanbo_ocr.services.urllib.request.urlopen",
        lambda *a, **kw: (_ for _ in ()).throw(exc),
    )

    req = urllib.request.Request("http://127.0.0.1:8080/test", method="GET")

    def fetch() -> None:
        _request_json(req, timeout=5)

    with pytest.raises(ServiceClientError, match="500"):
        fetch()


def test_request_json_raises_on_url_error(monkeypatch: Any) -> None:
    def fake_urlopen(request: Any, *, timeout: float) -> None:
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr("gwanbo_ocr.services.urllib.request.urlopen", fake_urlopen)

    req = urllib.request.Request("http://127.0.0.1:8080/test", method="GET")
    with pytest.raises(ServiceClientError, match="connection refused"):
        _request_json(req, timeout=5)


def test_request_json_raises_on_non_json_response(monkeypatch: Any) -> None:
    class FakeResponse:
        def read(self) -> bytes:
            return b"not json at all"

        def __enter__(self) -> FakeResponse:
            return self

        def __exit__(self, *args: Any) -> None:
            pass

    monkeypatch.setattr(
        "gwanbo_ocr.services.urllib.request.urlopen",
        lambda req, *, timeout: FakeResponse(),
    )

    req = urllib.request.Request("http://127.0.0.1:8080/test", method="GET")
    with pytest.raises(ServiceClientError, match="non-JSON"):
        _request_json(req, timeout=5)


def test_request_json_raises_on_non_object_response(monkeypatch: Any) -> None:
    class FakeResponse:
        def read(self) -> bytes:
            return b'["array", "not", "object"]'

        def __enter__(self) -> FakeResponse:
            return self

        def __exit__(self, *args: Any) -> None:
            pass

    monkeypatch.setattr(
        "gwanbo_ocr.services.urllib.request.urlopen",
        lambda req, *, timeout: FakeResponse(),
    )

    req = urllib.request.Request("http://127.0.0.1:8080/test", method="GET")
    with pytest.raises(ServiceClientError, match="non-object"):
        _request_json(req, timeout=5)


def test_result_from_service_payload_raises_on_error_status() -> None:
    with pytest.raises(ServiceClientError, match="OCR failed"):
        _result_from_service_payload(
            {"status": "error", "error": "OCR failed"},
            page_number=1,
            backend="test",
        )


def test_markitdown_service_client_health(monkeypatch: Any) -> None:
    def fake_get_json(url: str, *, timeout: float) -> dict[str, Any]:
        return {"status": "ok", "version": "1.0.0"}

    monkeypatch.setattr("gwanbo_ocr.services._get_json", fake_get_json)

    client = MarkItDownServiceClient("http://127.0.0.1:8081")
    result = client.health()

    assert result["status"] == "ok"
    assert result["version"] == "1.0.0"


def test_paddle_service_client_health(monkeypatch: Any) -> None:
    def fake_get_json(url: str, *, timeout: float) -> dict[str, Any]:
        return {"status": "ok"}

    monkeypatch.setattr("gwanbo_ocr.services._get_json", fake_get_json)

    client = PaddleOcrServiceClient("http://127.0.0.1:8082")
    result = client.health()

    assert result["status"] == "ok"


def test_paddle_service_client_transcribe_vl(monkeypatch: Any) -> None:
    captured: dict[str, Any] = {}

    def fake_post_json(url: str, payload: dict[str, Any], *, timeout: float) -> dict[str, Any]:
        captured.update({"url": url, "payload": payload})
        return {"status": "ok", "text": "vl text"}

    monkeypatch.setattr("gwanbo_ocr.services._post_json", fake_post_json)

    client = PaddleOcrServiceClient("http://127.0.0.1:8082")
    result = client.transcribe_vl(
        Path("/images/page.png"),
        vl_rec_backend="vllm",
        vl_rec_server_url="http://vllm:8000/v1",
        vl_rec_api_model_name="PaddleOCR-VL",
    )

    assert result.text == "vl text"
    assert result.backend == "paddleocr_vl_service"
    assert captured["url"] == "http://127.0.0.1:8082/ocr/vl"
    assert captured["payload"]["vl_rec_backend"] == "vllm"
