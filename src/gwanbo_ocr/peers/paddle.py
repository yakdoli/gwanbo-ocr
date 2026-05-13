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
    service_url: str | None = None,
    timeout_seconds: int = 60,
) -> dict[str, Any]:
    """Render pages with PyMuPDF and transcribe with PaddleOCR."""
    if service_url:
        from gwanbo_ocr.runners.paddle_service import PaddleOcrServiceRunner
    else:
        try:
            from gwanbo_ocr.runners.paddle import PaddleOcrRunner
        except ImportError:
            return _skipped("paddleocr is not installed", method="PaddleOCR")

    rendered = _render_pages(pdf_path, image_dir=image_dir, max_pages=max_pages, dpi=dpi)
    if rendered.get("status") != "ok":
        return _error(str(rendered.get("error", "render failed")), method="PaddleOCR")

    def _run() -> dict[str, Any]:
        runner = (
            PaddleOcrServiceRunner(service_url, lang=lang, timeout=timeout_seconds)
            if service_url
            else PaddleOcrRunner(lang=lang)
        )
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
        failed_pages = sum(1 for item in page_results if item.get("status") == "error")
        if page_results and failed_pages == len(page_results):
            return {
                "status": "error",
                "method": "PaddleOCR",
                "text_extractable": False,
                "pages": rendered["page_count"],
                "scanned_pages": len(rendered["pages"]),
                "text_chars": 0,
                "sample_text": "",
                "dpi": dpi,
                "lang": lang,
                "service_url": service_url,
                "page_results": page_results,
                "error": "all PaddleOCR pages failed",
            }
        return {
            "status": "partial" if failed_pages else "ok",
            "method": "PaddleOCR",
            "text_extractable": total_chars > 0,
            "pages": rendered["page_count"],
            "scanned_pages": len(rendered["pages"]),
            "text_chars": total_chars,
            "sample_text": " ".join(parts)[:sample_chars],
            "dpi": dpi,
            "lang": lang,
            "service_url": service_url,
            "page_results": page_results,
            "error": None,
        }

    return _with_timeout(_run, timeout_seconds, method="PaddleOCR")


def extract_paddle_ocr_vl(
    pdf_path: Path,
    *,
    image_dir: Path,
    max_pages: int | None = 1,
    sample_chars: int = SAMPLE_CHARS_DEFAULT,
    dpi: int = 200,
    pipeline_version: str = "v1.5",
    vl_rec_backend: str | None = None,
    vl_rec_server_url: str | None = None,
    vl_rec_api_model_name: str | None = None,
    vl_rec_api_key: str | None = None,
    service_url: str | None = None,
    timeout_seconds: int = 180,
) -> dict[str, Any]:
    """Render pages with PyMuPDF and parse them with PaddleOCR-VL."""
    if service_url:
        from gwanbo_ocr.runners.paddle_service import PaddleOcrVlServiceRunner
    else:
        try:
            from gwanbo_ocr.runners.paddle import PaddleOcrVlRunner
        except ImportError:
            return _skipped("paddleocr is not installed", method="PaddleOCR-VL")

    rendered = _render_pages(pdf_path, image_dir=image_dir, max_pages=max_pages, dpi=dpi)
    if rendered.get("status") != "ok":
        return _error(str(rendered.get("error", "render failed")), method="PaddleOCR-VL")

    def _run() -> dict[str, Any]:
        runner = (
            PaddleOcrVlServiceRunner(
                service_url,
                pipeline_version=pipeline_version,
                vl_rec_backend=vl_rec_backend,
                vl_rec_server_url=vl_rec_server_url,
                vl_rec_api_model_name=vl_rec_api_model_name,
                vl_rec_api_key=vl_rec_api_key,
                timeout=timeout_seconds,
            )
            if service_url
            else PaddleOcrVlRunner(
                pipeline_version=pipeline_version,
                vl_rec_backend=vl_rec_backend,
                vl_rec_server_url=vl_rec_server_url,
                vl_rec_api_model_name=vl_rec_api_model_name,
                vl_rec_api_key=vl_rec_api_key,
            )
        )
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
        failed_pages = sum(1 for item in page_results if item.get("status") == "error")
        if page_results and failed_pages == len(page_results):
            return {
                "status": "error",
                "method": "PaddleOCR-VL",
                "text_extractable": False,
                "pages": rendered["page_count"],
                "scanned_pages": len(rendered["pages"]),
                "text_chars": 0,
                "sample_text": "",
                "dpi": dpi,
                "pipeline_version": pipeline_version,
                "vl_rec_backend": vl_rec_backend,
                "vl_rec_server_url": vl_rec_server_url,
                "vl_rec_api_model_name": vl_rec_api_model_name,
                "service_url": service_url,
                "page_results": page_results,
                "error": "all PaddleOCR-VL pages failed",
            }
        return {
            "status": "partial" if failed_pages else "ok",
            "method": "PaddleOCR-VL",
            "text_extractable": total_chars > 0,
            "pages": rendered["page_count"],
            "scanned_pages": len(rendered["pages"]),
            "text_chars": total_chars,
            "sample_text": " ".join(parts)[:sample_chars],
            "dpi": dpi,
            "pipeline_version": pipeline_version,
            "vl_rec_backend": vl_rec_backend,
            "vl_rec_server_url": vl_rec_server_url,
            "vl_rec_api_model_name": vl_rec_api_model_name,
            "service_url": service_url,
            "page_results": page_results,
            "error": None,
        }

    return _with_timeout(_run, timeout_seconds, method="PaddleOCR-VL")
