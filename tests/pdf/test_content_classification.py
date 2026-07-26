from __future__ import annotations

from io import BytesIO
from pathlib import Path

import pymupdf
import pytest
from PIL import Image

from gwanbo_ocr.pdf import classification as pdf_classification
from gwanbo_ocr.pdf.classification import classify_pdf_document
from gwanbo_ocr.pdf.content import ContentMode, inspect_pdf_content
from gwanbo_ocr.pdf.text import analyze_pdf_text


def _image_bytes() -> bytes:
    output = BytesIO()
    Image.new("RGB", (32, 32), "white").save(output, format="PNG")
    return output.getvalue()


def _save_pdf(path: Path, page_kinds: tuple[str, ...]) -> None:
    with pymupdf.open() as document:
        for page_kind in page_kinds:
            page = document.new_page(width=300, height=400)
            if page_kind in {"image", "ocr"}:
                page.insert_image(page.rect, stream=_image_bytes())
            if page_kind in {"text", "ocr"}:
                page.insert_text((40, 80), "Gwanbo native text")
        document.save(path)


def test_inspect_pdf_content_classifies_native_text(tmp_path: Path) -> None:
    pdf_path = tmp_path / "native.pdf"
    _save_pdf(pdf_path, ("text",))

    result = inspect_pdf_content(pdf_path)

    assert result.mode is ContentMode.NATIVE_TEXT
    assert result.pages[0].mode is ContentMode.NATIVE_TEXT


def test_inspect_pdf_content_classifies_image_only(tmp_path: Path) -> None:
    pdf_path = tmp_path / "scanned.pdf"
    _save_pdf(pdf_path, ("image",))

    result = inspect_pdf_content(pdf_path)

    assert result.mode is ContentMode.IMAGE_ONLY
    assert result.pages[0].image_coverage > 0.95


def test_inspect_pdf_content_classifies_image_with_text_layer(tmp_path: Path) -> None:
    pdf_path = tmp_path / "ocr-layer.pdf"
    _save_pdf(pdf_path, ("ocr",))

    result = inspect_pdf_content(pdf_path)

    assert result.mode is ContentMode.IMAGE_WITH_TEXT_LAYER
    assert result.pages[0].text_chars > 0


def test_inspect_pdf_content_classifies_mixed_document(tmp_path: Path) -> None:
    pdf_path = tmp_path / "mixed.pdf"
    _save_pdf(pdf_path, ("text", "image"))

    result = inspect_pdf_content(pdf_path)

    assert result.mode is ContentMode.MIXED
    assert [page.mode for page in result.pages] == [
        ContentMode.NATIVE_TEXT,
        ContentMode.IMAGE_ONLY,
    ]


def test_classify_pdf_document_exposes_content_mode_and_legacy_class(tmp_path: Path) -> None:
    pdf_path = tmp_path / "ocr-layer.pdf"
    _save_pdf(pdf_path, ("ocr",))

    result = classify_pdf_document(pdf_path)

    assert result["content_mode"] == ContentMode.IMAGE_WITH_TEXT_LAYER
    assert result["pdf_text_class"] == "text_extractable"
    assert result["needs_ocr"] is False


def test_classify_pdf_document_uses_supplied_text_metadata_when_analysis_disabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pdf_path = tmp_path / "metadata-only.pdf"
    _save_pdf(pdf_path, ("text",))

    def _unexpected_inspection(*args: object, **kwargs: object) -> None:
        raise AssertionError("inspect_pdf_content should not run when analyze_text=False")

    monkeypatch.setattr(pdf_classification, "inspect_pdf_content", _unexpected_inspection)

    result = classify_pdf_document(
        pdf_path,
        analyze_text=False,
        text_metadata={
            "status": "ok",
            "text_extractable": True,
            "total_chars": 77,
        },
    )

    assert result["document_class"] == "text_pdf"
    assert result["confidence"] == 0.95
    assert result["valid_pdf"] is True
    assert result["text_extractable"] is True
    assert result["needs_ocr"] is False
    assert result["total_chars"] == 77
    assert result["reasons"] == ["extractable_text_layer"]
    assert result["text"] == {
        "status": "ok",
        "text_extractable": True,
        "total_chars": 77,
    }
    assert "content_mode" not in result
    assert "content_pages" not in result
    assert "pdf_text_class" not in result


def test_classify_pdf_document_returns_error_for_malformed_content_inspection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pdf_path = tmp_path / "broken.pdf"
    _save_pdf(pdf_path, ("text",))

    def _broken_inspection(*args: object, **kwargs: object) -> None:
        raise RuntimeError("malformed pdf")

    monkeypatch.setattr(pdf_classification, "inspect_pdf_content", _broken_inspection)

    result = classify_pdf_document(
        pdf_path,
        text_metadata={
            "status": "ok",
            "text_extractable": False,
            "total_chars": 0,
        },
    )

    assert result["document_class"] == "error"
    assert result["confidence"] == 0.0
    assert result["valid_pdf"] is True
    assert result["text_extractable"] is False
    assert result["reasons"] == ["content_analysis_error"]
    assert result["text"]["status"] == "ok"


def test_text_and_content_use_same_representative_pages(tmp_path: Path) -> None:
    pdf_path = tmp_path / "late-text.pdf"
    _save_pdf(pdf_path, ("image", "image", "image", "image", "text"))

    content = inspect_pdf_content(pdf_path, max_pages=3)
    text = analyze_pdf_text(pdf_path, max_pages=3)

    assert content.analyzed_pages == (0, 2, 4)
    assert text["analyzed_pages"] == [0, 2, 4]
    assert text["text_extractable"] is True
