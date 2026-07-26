from __future__ import annotations

from gwanbo_ocr.pdf.metadata_correction import (
    ExtractedDocumentMetadata,
    MetadataGateStatus,
    SourceDocumentMetadata,
    correct_extracted_metadata,
)


def test_gate_passes_consistent_metadata() -> None:
    # Given
    extracted = ExtractedDocumentMetadata(
        document_id="doc-1",
        toc_id="doc-1",
        title="관보 공고",
        date="2024-01-02",
        date_compact="20240102",
        year=2024,
    )
    source = SourceDocumentMetadata(
        id="doc-1",
        toc_id="doc-1",
        title="관보 공고",
        date="2024-01-02",
    )

    # When
    result = correct_extracted_metadata(extracted, source)

    # Then
    assert result.status is MetadataGateStatus.PASS
    assert result.can_publish is True
    assert result.changes == ()
    assert result.issues == ()


def test_gate_corrects_metadata_from_authoritative_source() -> None:
    # Given
    extracted = ExtractedDocumentMetadata.model_validate(
        {
            "document_id": "doc-1",
            "title": "잘못 읽은 제목",
            "date": "2024/01/03",
            "page_count": 1,
            "ocr_engine": "qwen",
        }
    )
    source = SourceDocumentMetadata.model_validate(
        {
            "id": "doc-1",
            "toc_id": "doc-1",
            "title": "  관보   공고  ",
            "date": "2024-01-02",
            "pdf": {"sha256": "abc123", "size_bytes": 42},
            "pdf_text": {"pages": 2},
        }
    )

    # When
    result = correct_extracted_metadata(extracted, source)

    # Then
    assert result.status is MetadataGateStatus.CORRECTED
    assert result.can_publish is True
    assert result.corrected_metadata.title == "관보 공고"
    assert result.corrected_metadata.date == "2024-01-02"
    assert result.corrected_metadata.date_compact == "20240102"
    assert result.corrected_metadata.year == 2024
    assert result.corrected_metadata.page_count == 2
    assert result.corrected_metadata.source_page_count == 2
    assert result.corrected_metadata.source_pdf_sha256 == "abc123"
    assert result.corrected_metadata.source_pdf_size_bytes == 42
    assert result.corrected_metadata.ocr_engine == "qwen"
    assert {change.field for change in result.changes} >= {"title", "date", "page_count"}


def test_gate_requires_review_when_required_source_metadata_is_missing() -> None:
    # Given
    extracted = ExtractedDocumentMetadata(document_id="doc-1")
    source = SourceDocumentMetadata(id="doc-1")

    # When
    result = correct_extracted_metadata(extracted, source)

    # Then
    assert result.status is MetadataGateStatus.REVIEW
    assert result.can_publish is False
    assert {issue.field for issue in result.issues} == {"date", "title"}


def test_gate_rejects_mismatched_document_identity() -> None:
    # Given
    extracted = ExtractedDocumentMetadata(
        document_id="wrong-document",
        title="관보 공고",
        date="2024-01-02",
    )
    source = SourceDocumentMetadata(
        id="doc-1",
        title="관보 공고",
        date="2024-01-02",
    )

    # When
    result = correct_extracted_metadata(extracted, source)

    # Then
    assert result.status is MetadataGateStatus.REJECTED
    assert result.can_publish is False
    assert result.corrected_metadata.document_id == "doc-1"
    assert result.issues[0].field == "document_id"


def test_gate_rejects_source_identity_conflict_with_preferred_source_id() -> None:
    extracted = ExtractedDocumentMetadata(document_id="ocr-wrong", title="관보", date="2024-01-02")
    source = SourceDocumentMetadata(
        id="authoritative",
        toc_id="conflicting",
        title="관보",
        date="2024-01-02",
    )

    result = correct_extracted_metadata(extracted, source)

    assert (
        result.status,
        result.can_publish,
        result.corrected_metadata.document_id,
        result.issues[0].code,
    ) == (MetadataGateStatus.REJECTED, False, "authoritative", "source_identity_conflict")


def test_gate_rejects_mismatched_pdf_sha256() -> None:
    # Given
    extracted = ExtractedDocumentMetadata(
        document_id="doc-1",
        toc_id="doc-1",
        content_id="content-1",
        title="관보 공고",
        date="2024-01-02",
        source_pdf_sha256="wrong-hash",
    )
    source = SourceDocumentMetadata.model_validate(
        {
            "id": "doc-1",
            "toc_id": "doc-1",
            "content_id": "content-1",
            "title": "관보 공고",
            "date": "2024-01-02",
            "pdf": {"sha256": "expected-hash"},
        }
    )

    # When
    result = correct_extracted_metadata(extracted, source)

    # Then
    assert result.status is MetadataGateStatus.REJECTED
    assert result.can_publish is False
    assert {issue.field for issue in result.issues} == {"source_pdf_sha256"}


def test_gate_rejects_mismatched_content_and_pdf_identity() -> None:
    # Given
    extracted = ExtractedDocumentMetadata(
        document_id="doc-1",
        toc_id="doc-1",
        content_id="wrong-content",
        title="관보 공고",
        date="2024-01-02",
        source_pdf_sha256="wrong-hash",
    )
    source = SourceDocumentMetadata.model_validate(
        {
            "id": "doc-1",
            "toc_id": "doc-1",
            "content_id": "content-1",
            "title": "관보 공고",
            "date": "2024-01-02",
            "pdf": {"sha256": "expected-hash"},
        }
    )

    # When
    result = correct_extracted_metadata(extracted, source)

    # Then
    assert result.status is MetadataGateStatus.REJECTED
    assert result.can_publish is False
    assert {issue.field for issue in result.issues} == {"content_id", "source_pdf_sha256"}


def test_gate_reviews_mismatched_pdf_size() -> None:
    # Given
    extracted = ExtractedDocumentMetadata(
        document_id="doc-1",
        toc_id="doc-1",
        title="관보 공고",
        date="2024-01-02",
        source_pdf_size_bytes=41,
    )
    source = SourceDocumentMetadata.model_validate(
        {
            "id": "doc-1",
            "toc_id": "doc-1",
            "title": "관보 공고",
            "date": "2024-01-02",
            "pdf": {"size_bytes": 42},
        }
    )

    # When
    result = correct_extracted_metadata(extracted, source)

    # Then
    assert result.status is MetadataGateStatus.REVIEW
    assert result.can_publish is False
    assert result.issues[0].field == "source_pdf_size_bytes"
