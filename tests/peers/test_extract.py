from __future__ import annotations

import sys
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from gwanbo_ocr.peers import (
    extract_markitdown,
    extract_native_text,
    extract_pdfplumber,
    extract_vlm_ocr,
)

# Minimal valid PDF bytes with one text page.
TEXT_PDF = b"""%PDF-1.4
1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj
2 0 obj<</Type/Pages/Count 1/Kids[3 0 R]>>endobj
3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 300 144]/Contents 4 0 R/Resources<</Font<</F1 5 0 R>>>>>>endobj
4 0 obj<</Length 43>>stream
BT /F1 24 Tf 72 72 Td (Hello PDF Text) Tj ET
endstream endobj
5 0 obj<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>endobj
xref
0 6
0000000000 65535 f\x20
0000000010 00000 n\x20
0000000060 00000 n\x20
0000000117 00000 n\x20
0000000243 00000 n\x20
0000000346 00000 n\x20
trailer<</Size 6/Root 1 0 R>>
startxref
416
%%EOF
"""


def _write_pdf(path: Path, payload: bytes = TEXT_PDF) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


# ---------------------------------------------------------------------------
# Individual peer extractors
# ---------------------------------------------------------------------------


class TestExtractNativeText:
    def test_extracts_text_from_valid_pdf(self, tmp_path: Path) -> None:
        pdf = _write_pdf(tmp_path / "doc.pdf")
        result = extract_native_text(pdf, max_pages=1)

        assert result["status"] == "ok"
        assert result["method"] == "pypdf.extract_text"
        assert isinstance(result["text_chars"], int)
        assert isinstance(result["text_extractable"], bool)

    def test_returns_error_for_missing_file(self, tmp_path: Path) -> None:
        result = extract_native_text(tmp_path / "nonexistent.pdf", max_pages=1)
        assert result["status"] in {"error", "ok"}

    def test_sample_text_capped_at_sample_chars(self, tmp_path: Path) -> None:
        pdf = _write_pdf(tmp_path / "doc.pdf")
        result = extract_native_text(pdf, max_pages=1, sample_chars=5)
        assert len(result["sample_text"]) <= 5


class TestExtractPdfplumber:
    def test_extracts_text_when_pdfplumber_available(self, tmp_path: Path) -> None:
        try:
            import pdfplumber  # noqa: F401
        except ImportError:
            pytest.skip("pdfplumber not installed")

        pdf = _write_pdf(tmp_path / "doc.pdf")
        result = extract_pdfplumber(pdf, max_pages=1)

        assert result["method"] == "pdfplumber.extract_text"
        assert result["status"] in {"ok", "error"}
        assert isinstance(result["text_chars"], int)

    def test_skipped_when_pdfplumber_missing(self, tmp_path: Path) -> None:
        pdf = _write_pdf(tmp_path / "doc.pdf")
        with patch("gwanbo_ocr.peers.pdfplumber._pdfplumber", None):
            result = extract_pdfplumber(pdf)
        assert result["status"] == "skipped"
        assert "pdfplumber is not installed" in result["skip_reason"]


class TestExtractMarkitdown:
    def test_skipped_when_markitdown_missing(self, tmp_path: Path) -> None:
        pdf = _write_pdf(tmp_path / "doc.pdf")
        with patch("gwanbo_ocr.peers.markitdown._MarkItDown", None):
            result = extract_markitdown(pdf)
        assert result["status"] == "skipped"

    def test_calls_markitdown_when_available(self, tmp_path: Path) -> None:
        pdf = _write_pdf(tmp_path / "doc.pdf")

        class FakeResult:
            text_content = "관보 제목 내용"

        class FakeMarkItDown:
            def __init__(self, **_kwargs: Any) -> None:
                pass

            def convert(self, _path: str) -> FakeResult:
                return FakeResult()

        with patch("gwanbo_ocr.peers.markitdown._MarkItDown", FakeMarkItDown):
            result = extract_markitdown(pdf, sample_chars=100)

        assert result["status"] == "ok"
        assert result["text_chars"] > 0
        assert "관보" in result["sample_text"]

    def test_records_error_on_exception(self, tmp_path: Path) -> None:
        pdf = _write_pdf(tmp_path / "doc.pdf")

        class FailingMarkItDown:
            def __init__(self, **_kwargs: Any) -> None:
                pass

            def convert(self, _path: str) -> None:
                raise RuntimeError("conversion failed")

        with patch("gwanbo_ocr.peers.markitdown._MarkItDown", FailingMarkItDown):
            result = extract_markitdown(pdf)

        assert result["status"] == "error"
        assert "conversion failed" in result["error"]


class TestExtractVlmOcr:
    def _make_runner(self, text: str = "VLM transcribed text") -> Any:
        from unittest.mock import MagicMock

        from gwanbo_ocr.runners.base import TranscriptionResult

        runner = MagicMock()
        runner.model = "test-vlm"
        runner.transcribe.return_value = TranscriptionResult.from_payload(
            {"text": text}, backend="vlm"
        )
        return runner

    def test_renders_and_transcribes(self, tmp_path: Path) -> None:
        try:
            import fitz  # noqa: F401
        except ImportError:
            pytest.skip("PyMuPDF not installed")

        pdf = _write_pdf(tmp_path / "doc.pdf")
        runner = self._make_runner("관보 VLM 결과")
        result = extract_vlm_ocr(
            pdf,
            image_dir=tmp_path / "images",
            runner=runner,
            max_pages=1,
            dpi=72,
        )

        assert result["status"] == "ok"
        assert result["text_chars"] > 0
        assert "관보" in result["sample_text"]
        assert result["model"] == "test-vlm"

    def test_error_when_render_fails(self, tmp_path: Path) -> None:
        pdf = tmp_path / "missing.pdf"
        runner = self._make_runner()
        with patch(
            "gwanbo_ocr.peers.vlm._render_pages",
            return_value={"status": "error", "error": "render failed"},
        ):
            result = extract_vlm_ocr(pdf, image_dir=tmp_path / "img", runner=runner)
        assert result["status"] == "error"
