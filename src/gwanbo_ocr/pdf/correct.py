"""JSONL adapter for auditable OCR correction."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, assert_never

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, ValidationError

from .correction import CorrectionResult, correct_ocr_text
from .io import iter_jsonl_records, write_json_lines_atomic
from .limits import MAX_OCR_TEXT_CHARS, MAX_VLM_SPANS
from .metadata_correction import (
    ExtractedDocumentMetadata,
    MetadataGateResult,
    MetadataGateStatus,
    SourceDocumentMetadata,
    correct_extracted_metadata,
)
from .vlm_correction import OcrSpan, ReviewBackend, ReviewedCorrection, review_ocr_text


class OcrInputRow(BaseModel):
    """Validated OCR row while preserving source metadata."""

    model_config = ConfigDict(extra="allow", frozen=True)

    document_id: str | None = None
    raw_text: str = Field(max_length=MAX_OCR_TEXT_CHARS)
    low_confidence_spans: tuple[OcrSpan, ...] = Field(default=(), max_length=MAX_VLM_SPANS)
    document_metadata: ExtractedDocumentMetadata | None = Field(
        default=None,
        validation_alias=AliasChoices("document_metadata", "metadata"),
    )
    source_metadata: SourceDocumentMetadata | None = None


class CorrectedOutputRow(BaseModel):
    model_config = ConfigDict(frozen=True)

    schema_version: Literal["ocr-correction/v1"] = "ocr-correction/v1"
    status: Literal["ok", "review", "rejected"] = "ok"
    document_id: str | None
    raw_text: str
    ocr_correction: CorrectionResult | ReviewedCorrection
    metadata_correction: MetadataGateResult | None = None


class CorrectionErrorRow(BaseModel):
    model_config = ConfigDict(frozen=True)

    schema_version: Literal["ocr-correction/v1"] = "ocr-correction/v1"
    status: Literal["error"] = "error"
    line_number: int
    error: str


@dataclass(frozen=True, slots=True)
class CorrectionBatchSummary:
    """Counts and output location for one correction batch."""

    processed: int
    errors: int
    output_path: str


def correct_manifest(
    input_path: Path | str,
    output_path: Path | str,
    review_backend: ReviewBackend | None = None,
) -> CorrectionBatchSummary:
    """Correct valid OCR rows and retain malformed rows as explicit errors."""
    processed = 0
    errors = 0

    def corrected_rows() -> Iterator[str]:
        nonlocal processed, errors
        for line_number, payload in iter_jsonl_records(input_path):
            try:
                source = OcrInputRow.model_validate(payload)
            except ValidationError as error:
                errors += 1
                yield CorrectionErrorRow(
                    line_number=line_number, error=str(error)
                ).model_dump_json()
                continue

            correction = (
                review_ocr_text(source.raw_text, source.low_confidence_spans, review_backend)
                if review_backend is not None and source.low_confidence_spans
                else correct_ocr_text(source.raw_text)
            )
            metadata_correction = None
            if source.document_metadata is not None or source.source_metadata is not None:
                document_metadata = source.document_metadata or ExtractedDocumentMetadata()
                if source.document_id is not None:
                    document_metadata = document_metadata.model_copy(
                        update={"row_document_id": source.document_id}
                    )
                metadata_correction = correct_extracted_metadata(
                    document_metadata,
                    source.source_metadata or SourceDocumentMetadata(),
                )
            output_status: Literal["ok", "review", "rejected"] = "ok"
            output_document_id = source.document_id
            if metadata_correction is not None:
                output_document_id = (
                    metadata_correction.corrected_metadata.document_id or source.document_id
                )
                match metadata_correction.status:
                    case MetadataGateStatus.PASS | MetadataGateStatus.CORRECTED:
                        pass
                    case MetadataGateStatus.REVIEW:
                        output_status = "review"
                    case MetadataGateStatus.REJECTED:
                        output_status = "rejected"
                    case unreachable:
                        assert_never(unreachable)
            if correction.requires_review and output_status == "ok":
                output_status = "review"
            if output_status == "ok":
                processed += 1
            else:
                errors += 1
            yield CorrectedOutputRow(
                status=output_status,
                document_id=output_document_id,
                raw_text=source.raw_text,
                ocr_correction=correction,
                metadata_correction=metadata_correction,
            ).model_dump_json()

    write_json_lines_atomic(output_path, corrected_rows())
    return CorrectionBatchSummary(
        processed=processed,
        errors=errors,
        output_path=str(output_path),
    )
