"""Native PDF text extraction metadata."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pymupdf

from .io import file_sha256, normalize_text
from .limits import (
    DEFAULT_MAX_FILE_BYTES,
    PdfResourceLimitError,
    PdfTimeoutError,
    ensure_pdf_size,
    pdf_deadline,
    sample_page_indexes,
)


def analyze_pdf_text(
    pdf_path: Path | str,
    *,
    include_sample: bool = False,
    sample_chars: int = 1000,
    include_sha256: bool = False,
    max_pages: int | None = None,
    timeout_seconds: float = 30,
    max_file_bytes: int = DEFAULT_MAX_FILE_BYTES,
) -> dict[str, Any]:
    """Extract native text and return stable routing metadata."""
    path = Path(pdf_path)
    result: dict[str, Any] = {
        "path": str(path),
        "filename": path.name,
        "status": "ok",
        "text_extractable": False,
        "text_pages": 0,
        "total_chars": 0,
        "extraction_method": "PyMuPDF.TextPage.extractText",
        "generated_at": datetime.now(UTC).isoformat(),
    }
    try:
        result["size_bytes"] = path.stat().st_size
        ensure_pdf_size(path, max_file_bytes)
        if include_sha256:
            result["sha256"] = file_sha256(path)
        with pdf_deadline(timeout_seconds), pymupdf.open(path) as document:
            result["pages"] = len(document)
            result["pdf_metadata"] = document.metadata if document.metadata is not None else {}
            page_indexes = sample_page_indexes(len(document), max_pages)
            samples: list[str] = []
            for page_index in page_indexes:
                text = normalize_text(document[page_index].get_textpage().extractText())
                if not text:
                    continue
                result["text_pages"] += 1
                result["total_chars"] += len(text)
                if include_sample:
                    samples.append(text)
            result["scanned_pages"] = len(page_indexes)
            result["analyzed_pages"] = list(page_indexes)
            result["text_extractable"] = result["text_pages"] > 0
            if result["text_pages"]:
                result["avg_chars_per_text_page"] = round(
                    result["total_chars"] / result["text_pages"], 2
                )
            if include_sample:
                result["sample_text"] = " ".join(samples)[:sample_chars]
    except (
        OSError,
        RuntimeError,
        ValueError,
        PdfResourceLimitError,
        PdfTimeoutError,
        pymupdf.FileDataError,
    ) as error:
        result.update({"status": "error", "error": str(error), "text_extractable": False})
    return result


def extract_pdf_text(
    pdf_path: Path | str,
    *,
    max_pages: int | None = None,
    timeout_seconds: int = 30,
) -> str:
    """Return normalized native PDF text."""
    metadata = analyze_pdf_text(
        pdf_path,
        include_sample=True,
        sample_chars=10_000_000,
        max_pages=max_pages,
        timeout_seconds=timeout_seconds,
    )
    return str(metadata.get("sample_text") or "")


def compact_text_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """Remove large sample text before storing metadata indexes."""
    return {key: value for key, value in metadata.items() if key != "sample_text"}
