from __future__ import annotations

from pathlib import Path
from typing import Any

from ._helpers import SAMPLE_CHARS_DEFAULT, _normalize, _skipped, _with_timeout

_pdfplumber: Any
try:
    import pdfplumber as _pdfplumber  # type: ignore[import-not-found,no-redef]
except ImportError:
    _pdfplumber = None


def extract_pdfplumber(
    pdf_path: Path,
    *,
    max_pages: int | None = 1,
    sample_chars: int = SAMPLE_CHARS_DEFAULT,
    timeout_seconds: int = 30,
) -> dict[str, Any]:
    """Extract text using pdfplumber.page.extract_text()."""
    plumber = _pdfplumber
    if plumber is None:
        return _skipped("pdfplumber is not installed", method="pdfplumber.extract_text")

    def _run() -> dict[str, Any]:
        with plumber.open(str(pdf_path)) as doc:
            page_count = len(doc.pages)
            pages_to_scan = page_count if max_pages is None else min(page_count, max_pages)
            parts: list[str] = []
            total_chars = 0
            page_errors: list[dict[str, Any]] = []
            for idx in range(pages_to_scan):
                try:
                    text = doc.pages[idx].extract_text() or ""
                    text = _normalize(text)
                    total_chars += len(text)
                    if text and len(" ".join(parts)) < sample_chars:
                        parts.append(text)
                except Exception as exc:  # noqa: BLE001
                    page_errors.append({"page_index": idx, "error": str(exc)})
            result: dict[str, Any] = {
                "status": "ok",
                "method": "pdfplumber.extract_text",
                "text_extractable": total_chars > 0,
                "pages": page_count,
                "scanned_pages": pages_to_scan,
                "text_chars": total_chars,
                "sample_text": " ".join(parts)[:sample_chars],
                "error": None,
            }
            if page_errors:
                result["page_errors"] = page_errors
                result["page_error_count"] = len(page_errors)
            return result

    return _with_timeout(_run, timeout_seconds, method="pdfplumber.extract_text")
