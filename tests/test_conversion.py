from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gwanbo_ocr.conversion import ConversionError, convert_document, convert_manifest


class FakeResult:
    text_content = "# converted\n\n관보 본문"


class FakeMarkItDown:
    calls: list[dict[str, Any]] = []

    def __init__(self, **kwargs: Any) -> None:
        self.calls.append(kwargs)

    def convert(self, path: str) -> FakeResult:
        assert path
        return FakeResult()


class FakeOpenAI:
    calls: list[dict[str, Any]] = []

    def __init__(self, **kwargs: Any) -> None:
        self.calls.append(kwargs)


def test_convert_document_uses_ocr_llm_plugin_kwargs(tmp_path: Path, monkeypatch: Any) -> None:
    import gwanbo_ocr.conversion as conversion

    monkeypatch.setattr(conversion, "_MarkItDown", FakeMarkItDown)
    monkeypatch.setattr(conversion, "_OpenAI", FakeOpenAI)
    FakeMarkItDown.calls.clear()
    FakeOpenAI.calls.clear()

    source = tmp_path / "doc.pdf"
    source.write_bytes(b"%PDF")

    summary = convert_document(
        input_path=source,
        output=tmp_path / "markdown",
        mode="ocr-llm",
        llm_base_url="http://127.0.0.1:8000/v1",
        llm_model="Qwen/Qwen3.6-35B-A3B-FP8",
        llm_api_key="test-key",
        llm_prompt="Extract Korean text.",
    )

    output_path = Path(summary["output_path"])
    metadata_path = Path(summary["metadata_path"])
    assert output_path.read_text(encoding="utf-8").startswith("# converted")
    assert metadata_path.exists()
    assert FakeOpenAI.calls == [{"base_url": "http://127.0.0.1:8000/v1", "api_key": "test-key"}]
    assert len(FakeMarkItDown.calls) == 1
    assert FakeMarkItDown.calls[0]["enable_plugins"] is True
    assert FakeMarkItDown.calls[0]["llm_client"].__class__ is FakeOpenAI
    assert FakeMarkItDown.calls[0]["llm_model"] == "Qwen/Qwen3.6-35B-A3B-FP8"
    assert FakeMarkItDown.calls[0]["llm_prompt"] == "Extract Korean text."


def test_convert_manifest_writes_result_manifest(tmp_path: Path, monkeypatch: Any) -> None:
    import gwanbo_ocr.conversion as conversion

    monkeypatch.setattr(conversion, "_MarkItDown", FakeMarkItDown)
    monkeypatch.setattr(conversion, "_OpenAI", FakeOpenAI)

    source = tmp_path / "pdfs" / "doc.pdf"
    source.parent.mkdir()
    source.write_bytes(b"%PDF")
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps({"id": "item-1", "pdf_path": str(source)}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    summary = convert_manifest(
        manifest_path=manifest,
        output_dir=tmp_path / "converted",
        mode="plain",
    )

    rows = [
        json.loads(line)
        for line in Path(summary["manifest_path"]).read_text(encoding="utf-8").splitlines()
    ]
    assert summary["converted"] == 1
    assert rows[0]["status"] == "ok"
    assert rows[0]["id"] == "item-1"
    assert Path(rows[0]["markdown_path"]).exists()


def test_convert_manifest_rejects_invalid_mode_before_writing(tmp_path: Path) -> None:
    source = tmp_path / "doc.pdf"
    source.write_bytes(b"%PDF")
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps({"id": "item-1", "pdf_path": str(source)}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "converted"

    try:
        convert_manifest(
            manifest_path=manifest,
            output_dir=output_dir,
            mode="invalid",
        )
    except ConversionError as exc:
        assert "unsupported conversion mode" in str(exc)
    else:
        raise AssertionError("invalid mode should fail before row conversion")

    assert not (output_dir / "manifest.jsonl").exists()


def test_convert_manifest_suffixes_duplicate_output_keys(tmp_path: Path, monkeypatch: Any) -> None:
    import gwanbo_ocr.conversion as conversion

    monkeypatch.setattr(conversion, "_MarkItDown", FakeMarkItDown)
    monkeypatch.setattr(conversion, "_OpenAI", FakeOpenAI)

    source_a = tmp_path / "a.pdf"
    source_b = tmp_path / "b.pdf"
    source_a.write_bytes(b"%PDF")
    source_b.write_bytes(b"%PDF")
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        "\n".join(
            [
                json.dumps({"id": "same/key", "pdf_path": str(source_a)}, ensure_ascii=False),
                json.dumps({"id": "same key", "pdf_path": str(source_b)}, ensure_ascii=False),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    summary = convert_manifest(
        manifest_path=manifest,
        output_dir=tmp_path / "converted",
        key="id",
        workers=2,
    )

    rows = [
        json.loads(line)
        for line in Path(summary["manifest_path"]).read_text(encoding="utf-8").splitlines()
    ]
    markdown_paths = [row["markdown_path"] for row in rows]
    assert summary["converted"] == 2
    assert len(set(markdown_paths)) == 2
    assert markdown_paths[0].endswith("same_key.md")
    assert markdown_paths[1].endswith("same_key_2.md")
