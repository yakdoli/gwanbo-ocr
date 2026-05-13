from __future__ import annotations

from pathlib import Path
from typing import Any

from ._helpers import SAMPLE_CHARS_DEFAULT, _error, _normalize, _render_pages, _with_timeout


def extract_vlm_ocr(
    pdf_path: Path,
    *,
    image_dir: Path,
    runner: Any,
    max_pages: int | None = 1,
    sample_chars: int = SAMPLE_CHARS_DEFAULT,
    dpi: int = 200,
    timeout_seconds: int = 120,
) -> dict[str, Any]:
    """Render pages with PyMuPDF and transcribe with an OpenAI-compatible VLM."""
    rendered = _render_pages(pdf_path, image_dir=image_dir, max_pages=max_pages, dpi=dpi)
    if rendered.get("status") != "ok":
        return _error(str(rendered.get("error", "render failed")), method="VLM-OCR")

    method_label = f"VLM-OCR({getattr(runner, 'model', 'unknown')})"

    def _run() -> dict[str, Any]:
        parts: list[str] = []
        total_chars = 0
        page_results: list[dict[str, Any]] = []
        for page in rendered["pages"]:
            try:
                result = runner.transcribe(Path(page["path"]), page_number=page["page_index"] + 1)
                text = _normalize(result.text)
                total_chars += len(text)
                if text and len(" ".join(parts)) < sample_chars:
                    parts.append(text)
                page_results.append(
                    {
                        "page_index": page["page_index"],
                        "status": "ok",
                        "text_chars": len(text),
                        "latency_ms": result.data.get("latency_ms"),
                    }
                )
            except Exception as exc:  # noqa: BLE001
                page_results.append(
                    {"page_index": page["page_index"], "status": "error", "error": str(exc)}
                )
        return {
            "status": "ok",
            "method": method_label,
            "text_extractable": total_chars > 0,
            "pages": rendered["page_count"],
            "scanned_pages": len(rendered["pages"]),
            "text_chars": total_chars,
            "sample_text": " ".join(parts)[:sample_chars],
            "dpi": dpi,
            "model": getattr(runner, "model", None),
            "page_results": page_results,
            "error": None,
        }

    return _with_timeout(_run, timeout_seconds, method=method_label)
