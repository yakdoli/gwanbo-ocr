"""Page-level PDF content evidence for OCR routing."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import pymupdf

from .limits import (
    DEFAULT_MAX_FILE_BYTES,
    ensure_pdf_size,
    pdf_deadline,
    sample_page_indexes,
)
from .limits import (
    PdfResourceLimitError as PdfResourceLimitError,
)


class ContentMode(StrEnum):
    """Observable PDF page content, not unverifiable document provenance."""

    NATIVE_TEXT = "native_text"
    IMAGE_ONLY = "image_only"
    IMAGE_WITH_TEXT_LAYER = "image_with_text_layer"
    MIXED = "mixed"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class PageContentEvidence:
    """Signals collected from one PDF page."""

    page_index: int
    mode: ContentMode
    text_chars: int
    image_count: int
    image_coverage: float


@dataclass(frozen=True, slots=True)
class DocumentContentEvidence:
    """Aggregated page evidence for one PDF."""

    mode: ContentMode
    pages: tuple[PageContentEvidence, ...]
    total_pages: int
    analyzed_pages: tuple[int, ...]


def inspect_pdf_content(
    pdf_path: Path | str,
    *,
    max_pages: int | None = None,
    max_file_bytes: int = DEFAULT_MAX_FILE_BYTES,
    timeout_seconds: float = 30,
) -> DocumentContentEvidence:
    """Inspect text and raster coverage on representative PDF pages."""
    path = Path(pdf_path)
    ensure_pdf_size(path, max_file_bytes)
    with pdf_deadline(timeout_seconds), pymupdf.open(path) as document:
        indexes = sample_page_indexes(len(document), max_pages)
        pages = tuple(_inspect_page(document[index], index) for index in indexes)
        return DocumentContentEvidence(
            mode=_document_mode(pages),
            pages=pages,
            total_pages=len(document),
            analyzed_pages=indexes,
        )


def _inspect_page(page: pymupdf.Page, page_index: int) -> PageContentEvidence:
    text_chars = len("".join(page.get_textpage().extractText().split()))
    image_xrefs = {int(image[0]) for image in page.get_images(full=True)}
    page_area = page.rect.get_area()
    image_area = sum(
        rectangle.get_area() for xref in image_xrefs for rectangle in page.get_image_rects(xref)
    )
    image_coverage = min(1.0, image_area / page_area) if page_area else 0.0
    has_text = text_chars > 0
    has_page_image = bool(image_xrefs) and image_coverage >= 0.5

    if has_text and has_page_image:
        mode = ContentMode.IMAGE_WITH_TEXT_LAYER
    elif has_text:
        mode = ContentMode.NATIVE_TEXT
    elif image_xrefs:
        mode = ContentMode.IMAGE_ONLY
    else:
        mode = ContentMode.UNKNOWN

    return PageContentEvidence(
        page_index=page_index,
        mode=mode,
        text_chars=text_chars,
        image_count=len(image_xrefs),
        image_coverage=round(image_coverage, 4),
    )


def _document_mode(pages: tuple[PageContentEvidence, ...]) -> ContentMode:
    modes = {page.mode for page in pages}
    if not modes:
        return ContentMode.UNKNOWN
    if len(modes) == 1:
        return next(iter(modes))
    return ContentMode.MIXED
