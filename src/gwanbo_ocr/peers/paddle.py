from __future__ import annotations

from pathlib import Path
from typing import Any

from ._helpers import (
    SAMPLE_CHARS_DEFAULT,
    _error,
    _normalize,
    _render_pages,
    _skipped,
    _with_timeout,
)


def extract_paddle_ocr(
    pdf_path: Path,
    *,
    image_dir: Path,
    max_pages: int | None = 1,
    sample_chars: int = SAMPLE_CHARS_DEFAULT,
    dpi: int = 200,
    lang: str = "korean",
    timeout_seconds: int = 60,
) -> dict[str, Any]:
    """Render pages with PyMuPDF and transcribe with PaddleOCR."""
    try:
        from gwanbo_ocr.runners.paddle import PaddleOcrRunner
    except ImportError:
        return _skipped("paddleocr is not installed", method="PaddleOCR")

    rendered = _render_pages(pdf_path, image_dir=image_dir, max_pages=max_pages, dpi=dpi)
    if rendered.get("status") != "ok":
        return _error(str(rendered.get("error", "render failed")), method="PaddleOCR")

    def _run() -> dict[str, Any]:
        runner = PaddleOcrRunner(lang=lang)
        parts: list[str] = []
        total_chars = 0
        page_results: list[dict[str, Any]] = []
        for page in rendered["pages"]:
            try:
                result = runner.transcribe(Path(page["path"]))
                text = _normalize(result.text)
                total_chars += len(text)
                if text and len(" ".join(parts)) < sample_chars:
                    parts.append(text)
                page_results.append(
                    {"page_index": page["page_index"], "status": "ok", "text_chars": len(text)}
                )
            except Exception as exc:  # noqa: BLE001
                page_results.append(
                    {"page_index": page["page_index"], "status": "error", "error": str(exc)}
                )
        return {
            "status": "ok",
            "method": "PaddleOCR",
            "text_extractable": total_chars > 0,
            "pages": rendered["page_count"],
            "scanned_pages": len(rendered["pages"]),
            "text_chars": total_chars,
            "sample_text": " ".join(parts)[:sample_chars],
            "dpi": dpi,
            "lang": lang,
            "page_results": page_results,
            "error": None,
        }

    return _with_timeout(_run, timeout_seconds, method="PaddleOCR")
