"""Deterministic consistency gate for metadata extracted from scanned PDFs."""

from __future__ import annotations

import re
import unicodedata
from datetime import date, datetime
from enum import StrEnum
from typing import Final

from pydantic import BaseModel, ConfigDict

DATE_FORMATS: Final = ("%Y-%m-%d", "%Y%m%d", "%Y/%m/%d", "%Y.%m.%d")
SPACE_PATTERN: Final = re.compile(r"\s+")


class MetadataGateStatus(StrEnum):
    PASS = "pass"
    CORRECTED = "corrected"
    REVIEW = "review"
    REJECTED = "rejected"


class SourcePdfMetadata(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    sha256: str | None = None
    size_bytes: int | None = None


class SourcePdfTextMetadata(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    pages: int | None = None


class SourceDocumentMetadata(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    id: str | None = None
    toc_id: str | None = None
    stored_toc_seq: str | None = None
    content_id: str | None = None
    title: str | None = None
    keyword_field_subject: str | None = None
    stored_field_subject: str | None = None
    date: str | None = None
    keyword_field_regdate: str | None = None
    agency: str | None = None
    stored_organ_nm: str | None = None
    category: str | None = None
    stored_category_name: str | None = None
    pdf: SourcePdfMetadata | None = None
    pdf_text: SourcePdfTextMetadata | None = None
    url: str | None = None
    viewer_path: str | None = None


class ExtractedDocumentMetadata(BaseModel):
    model_config = ConfigDict(extra="allow", frozen=True)

    document_id: str | None = None
    row_document_id: str | None = None
    toc_id: str | None = None
    content_id: str | None = None
    title: str | None = None
    date: str | None = None
    date_compact: str | None = None
    year: int | None = None
    agency: str | None = None
    category: str | None = None
    page_count: int | None = None
    source_page_count: int | None = None
    source_pdf_sha256: str | None = None
    source_pdf_size_bytes: int | None = None
    source_url: str | None = None
    viewer_path: str | None = None
    ocr_engine: str | None = None
    correction_status: str | None = None


class MetadataChange(BaseModel):
    model_config = ConfigDict(frozen=True)

    field: str
    before: str | int | None
    after: str | int
    reason: str


class MetadataIssue(BaseModel):
    model_config = ConfigDict(frozen=True)

    code: str
    field: str
    observed: str | int | None = None
    expected: str | int | None = None


class MetadataGateResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    status: MetadataGateStatus
    can_publish: bool
    original_metadata: ExtractedDocumentMetadata
    corrected_metadata: ExtractedDocumentMetadata
    changes: tuple[MetadataChange, ...]
    issues: tuple[MetadataIssue, ...]


def correct_extracted_metadata(
    extracted: ExtractedDocumentMetadata,
    source: SourceDocumentMetadata,
) -> MetadataGateResult:
    """Correct deterministic fields and block ambiguous or mismatched records."""
    changes: list[MetadataChange] = []
    issues: list[MetadataIssue] = []
    updates: dict[str, str | int] = {}

    source_ids = tuple(
        value
        for value in (_clean(source.id), _clean(source.toc_id), _clean(source.stored_toc_seq))
        if value
    )
    canonical_id = source_ids[0] if source_ids else None
    if len(set(source_ids)) > 1:
        issues.append(MetadataIssue(code="source_identity_conflict", field="document_id"))
    elif canonical_id is None:
        issues.append(MetadataIssue(code="source_value_missing", field="document_id"))
    else:
        observed_ids = tuple(
            value
            for value in (
                _clean(extracted.document_id),
                _clean(extracted.row_document_id),
                _clean(extracted.toc_id),
            )
            if value
        )
        mismatched_id = next((value for value in observed_ids if value != canonical_id), None)
        if mismatched_id is not None:
            issues.append(
                MetadataIssue(
                    code="identity_mismatch",
                    field="document_id",
                    observed=mismatched_id,
                    expected=canonical_id,
                )
            )
    if canonical_id is not None:
        updates.update(document_id=canonical_id, toc_id=canonical_id)

    canonical_title = _first_text(
        source.title,
        source.keyword_field_subject,
        source.stored_field_subject,
    )
    if canonical_title:
        updates["title"] = canonical_title
    else:
        issues.append(MetadataIssue(code="source_value_missing", field="title"))

    canonical_date = _first_date(source.date, source.keyword_field_regdate)
    if canonical_date:
        updates.update(
            date=canonical_date,
            date_compact=canonical_date.replace("-", ""),
            year=date.fromisoformat(canonical_date).year,
        )
    else:
        issues.append(MetadataIssue(code="source_value_missing", field="date"))

    source_pdf = source.pdf or SourcePdfMetadata()
    source_pdf_text = source.pdf_text or SourcePdfTextMetadata()
    page_count = source_pdf_text.pages if (source_pdf_text.pages or 0) > 0 else None
    canonical_fields: dict[str, str | int | None] = {
        "content_id": _clean(source.content_id),
        "agency": _first_text(source.agency, source.stored_organ_nm),
        "category": _first_text(source.category, source.stored_category_name),
        "page_count": page_count,
        "source_page_count": page_count,
        "source_pdf_sha256": _clean(source_pdf.sha256),
        "source_pdf_size_bytes": source_pdf.size_bytes,
        "source_url": _clean(source.url),
        "viewer_path": _clean(source.viewer_path),
    }
    identity_values = (
        ("content_id", _clean(extracted.content_id), canonical_fields["content_id"]),
        (
            "source_pdf_sha256",
            _clean(extracted.source_pdf_sha256),
            canonical_fields["source_pdf_sha256"],
        ),
    )
    for field, observed, expected in identity_values:
        if observed is not None and expected is not None and observed != expected:
            issues.append(
                MetadataIssue(
                    code="identity_mismatch",
                    field=field,
                    observed=observed,
                    expected=expected,
                )
            )
    observed_size = extracted.source_pdf_size_bytes
    expected_size = canonical_fields["source_pdf_size_bytes"]
    if observed_size is not None and expected_size is not None and observed_size != expected_size:
        issues.append(
            MetadataIssue(
                code="value_mismatch",
                field="source_pdf_size_bytes",
                observed=observed_size,
                expected=expected_size,
            )
        )
    for field, value in canonical_fields.items():
        if value is not None:
            updates[field] = value

    for field, after in updates.items():
        before = getattr(extracted, field)
        if before == after:
            continue
        changes.append(
            MetadataChange(
                field=field,
                before=before,
                after=after,
                reason="authoritative_source",
            )
        )

    corrected = extracted.model_copy(update=updates)
    rejected = any(
        issue.code in {"identity_mismatch", "source_identity_conflict"} for issue in issues
    )
    if rejected:
        status = MetadataGateStatus.REJECTED
    elif issues:
        status = MetadataGateStatus.REVIEW
    elif changes:
        status = MetadataGateStatus.CORRECTED
    else:
        status = MetadataGateStatus.PASS
    corrected = corrected.model_copy(update={"correction_status": status.value})
    return MetadataGateResult(
        status=status,
        can_publish=status in {MetadataGateStatus.PASS, MetadataGateStatus.CORRECTED},
        original_metadata=extracted,
        corrected_metadata=corrected,
        changes=tuple(changes),
        issues=tuple(issues),
    )


def _clean(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = SPACE_PATTERN.sub(" ", unicodedata.normalize("NFC", value)).strip()
    return normalized or None


def _first_text(*values: str | None) -> str | None:
    return next((cleaned for value in values if (cleaned := _clean(value))), None)


def _first_date(*values: str | None) -> str | None:
    for value in values:
        cleaned = _clean(value)
        if cleaned is None:
            continue
        for date_format in DATE_FORMATS:
            try:
                return datetime.strptime(cleaned, date_format).date().isoformat()
            except ValueError:
                continue
    return None
