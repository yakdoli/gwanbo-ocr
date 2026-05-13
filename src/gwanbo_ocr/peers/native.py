from __future__ import annotations

from pathlib import Path
from typing import Any

from ._helpers import SAMPLE_CHARS_DEFAULT


def extract_native_text(
    pdf_path: Path,
    *,
    max_pages: int | None = 1,
    sample_chars: int = SAMPLE_CHARS_DEFAULT,
    timeout_seconds: int = 30,
) -> dict[str, Any]:
    """Extract text using pypdf / builtin parser."""
    from gwanbo_ocr.pdf.text import analyze_pdf_text

    metadata = analyze_pdf_text(
        pdf_path,
        include_sample=True,
        sample_chars=sample_chars,
        max_pages=max_pages,
        timeout_seconds=timeout_seconds,
    )
    sample = str(metadata.get("sample_text") or "")
    return {
        "status": str(metadata.get("status") or "unknown"),
        "method": "pypdf.extract_text",
        "text_extractable": bool(metadata.get("text_extractable")),
        "pages": metadata.get("pages"),
        "scanned_pages": metadata.get("scanned_pages"),
        "text_chars": int(metadata.get("total_chars") or len(sample)),
        "sample_text": sample[:sample_chars],
        "error": metadata.get("error"),
    }
