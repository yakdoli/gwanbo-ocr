from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_markitdown_service_convert_path_uses_shared_conversion(
    tmp_path: Path, monkeypatch: Any
) -> None:
    import scripts.markitdown_server as server

    monkeypatch.setenv("GWANBO_SERVICE_INPUT_ROOTS", str(tmp_path))
    monkeypatch.setenv("GWANBO_SERVICE_OUTPUT_ROOTS", str(tmp_path))
    output = tmp_path / "doc.md"
    output.write_text("converted markdown", encoding="utf-8")

    def fake_convert_document(**kwargs: Any) -> dict[str, Any]:
        assert kwargs["mode"] == "ocr-llm"
        assert kwargs["llm_model"] == "Qwen/Qwen3.6-35B-A3B-FP8"
        return {"output_path": str(output), "text_chars": 18}

    monkeypatch.setattr(server, "convert_document", fake_convert_document)
    client = TestClient(server.app)

    response = client.post(
        "/convert/path",
        json={
            "file_path": str(tmp_path / "doc.pdf"),
            "output_path": str(tmp_path / "out"),
            "mode": "ocr-llm",
            "llm_model": "Qwen/Qwen3.6-35B-A3B-FP8",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["markdown_content"] == "converted markdown"


def test_markitdown_service_rejects_paths_outside_allowed_roots(monkeypatch: Any) -> None:
    import scripts.markitdown_server as server

    monkeypatch.setenv("GWANBO_SERVICE_ALLOWED_ROOTS", "/workspace/runs")
    client = TestClient(server.app)

    response = client.post("/convert/path", json={"file_path": "/etc/passwd"})

    assert response.status_code == 400
    assert "not under allowed roots" in response.json()["detail"]


def test_markitdown_service_rejects_outputs_under_input_roots(monkeypatch: Any) -> None:
    import scripts.markitdown_server as server

    monkeypatch.delenv("GWANBO_SERVICE_ALLOWED_ROOTS", raising=False)
    monkeypatch.delenv("GWANBO_SERVICE_INPUT_ROOTS", raising=False)
    monkeypatch.delenv("GWANBO_SERVICE_OUTPUT_ROOTS", raising=False)
    client = TestClient(server.app)

    response = client.post(
        "/convert/path",
        json={
            "file_path": "/root/peti/artifacts/doc.pdf",
            "output_path": "/root/peti/artifacts/out.md",
        },
    )

    assert response.status_code == 400
    assert "not under allowed roots" in response.json()["detail"]


def test_markitdown_service_rejects_bad_batch_output_dir(monkeypatch: Any) -> None:
    import scripts.markitdown_server as server

    monkeypatch.setenv("GWANBO_SERVICE_ALLOWED_ROOTS", "/workspace/runs")
    client = TestClient(server.app)

    response = client.post(
        "/convert/batch",
        json={
            "file_paths": ["/workspace/runs/doc.pdf"],
            "output_dir": "/etc/out",
        },
    )

    assert response.status_code == 400
    assert "not under allowed roots" in response.json()["detail"]


def test_markitdown_service_legacy_roots_do_not_allow_outputs(monkeypatch: Any) -> None:
    import scripts.markitdown_server as server

    monkeypatch.setenv("GWANBO_SERVICE_ALLOWED_ROOTS", "/root/peti/artifacts")
    monkeypatch.delenv("GWANBO_SERVICE_OUTPUT_ROOTS", raising=False)
    client = TestClient(server.app)

    response = client.post(
        "/convert/path",
        json={
            "file_path": "/root/peti/artifacts/doc.pdf",
            "output_path": "/root/peti/artifacts/out.md",
        },
    )

    assert response.status_code == 400
    assert "not under allowed roots" in response.json()["detail"]


def test_markitdown_service_batch_suffixes_duplicate_stems(
    tmp_path: Path, monkeypatch: Any
) -> None:
    import scripts.markitdown_server as server

    monkeypatch.setenv("GWANBO_SERVICE_INPUT_ROOTS", str(tmp_path))
    monkeypatch.setenv("GWANBO_SERVICE_OUTPUT_ROOTS", str(tmp_path))
    inputs = [tmp_path / "a" / "doc.pdf", tmp_path / "b" / "doc.pdf"]
    for item in inputs:
        item.parent.mkdir()
        item.write_bytes(b"%PDF")

    output_paths: list[Path] = []

    def fake_convert_document(**kwargs: Any) -> dict[str, Any]:
        output = kwargs["output"]
        output_path = output if output.suffix == ".md" else output / "unexpected.md"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("converted", encoding="utf-8")
        output_paths.append(output_path)
        return {"output_path": str(output_path), "text_chars": 9}

    monkeypatch.setattr(server, "convert_document", fake_convert_document)
    client = TestClient(server.app)

    response = client.post(
        "/convert/batch",
        json={
            "file_paths": [str(item) for item in inputs],
            "output_dir": str(tmp_path / "out"),
        },
    )

    assert response.status_code == 200
    assert [path.name for path in output_paths] == ["doc.md", "doc_2.md"]


def test_paddle_service_classic_endpoint_uses_runner(tmp_path: Path, monkeypatch: Any) -> None:
    import scripts.paddleocr_server as server
    from gwanbo_ocr.runners.base import TranscriptionResult

    image = tmp_path / "page.png"
    image.write_bytes(b"png")
    monkeypatch.setenv("GWANBO_SERVICE_ALLOWED_ROOTS", str(tmp_path))

    class FakeRunner:
        def transcribe(self, image_path: Path, *, page_number: int | None = None) -> Any:
            assert image_path == image
            assert page_number == 1
            return TranscriptionResult.from_payload({"text": "관보 OCR"}, backend="fake")

    monkeypatch.setattr(server, "_classic", lambda _lang: FakeRunner())
    client = TestClient(server.app)

    response = client.post(
        "/ocr/classic",
        json={"image_path": str(image), "lang": "korean", "page_number": 1},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["text"] == "관보 OCR"


def test_paddle_service_rejects_paths_outside_allowed_roots(monkeypatch: Any) -> None:
    import scripts.paddleocr_server as server

    monkeypatch.setenv("GWANBO_SERVICE_ALLOWED_ROOTS", "/workspace/runs")
    client = TestClient(server.app)

    response = client.post("/ocr/classic", json={"image_path": "/etc/passwd"})

    assert response.status_code == 400
    assert "not under allowed roots" in response.json()["detail"]


def test_paddle_service_vl_runner_cache_is_keyed_by_request_options(monkeypatch: Any) -> None:
    import scripts.paddleocr_server as server

    created: list[dict[str, Any]] = []

    class FakeVlRunner:
        def __init__(self, **kwargs: Any) -> None:
            created.append(kwargs)

    monkeypatch.setattr(server, "PaddleOcrVlRunner", FakeVlRunner)
    monkeypatch.setattr(server, "_vl_runner", None)
    monkeypatch.setattr(server, "_vl_runner_key", None)

    first = server.VlOcrRequest(
        image_path="/tmp/page.png",
        vl_rec_server_url="http://one/v1",
        vl_rec_api_model_name="model-a",
    )
    second = server.VlOcrRequest(
        image_path="/tmp/page.png",
        vl_rec_server_url="http://two/v1",
        vl_rec_api_model_name="model-b",
    )

    assert server._vl(first) is server._vl(first)
    assert server._vl(second) is not server._vl(first)
    assert [item["vl_rec_server_url"] for item in created] == [
        "http://one/v1",
        "http://two/v1",
        "http://one/v1",
    ]
