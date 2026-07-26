"""Review-gated semantic correction for low-confidence OCR spans."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum
from itertools import pairwise
from pathlib import Path
from typing import Protocol, assert_never

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from .correction import correct_ocr_text, extract_protected_values
from .limits import MAX_VLM_SUGGESTIONS


@dataclass(frozen=True, slots=True)
class InvalidOcrSpanError(ValueError):
    start: int
    end: int

    def __str__(self) -> str:
        return f"span end {self.end} must be greater than start {self.start}"


class OcrSpan(BaseModel):
    model_config = ConfigDict(frozen=True)

    span_id: str
    text: str
    confidence: float = Field(ge=0, le=1)
    start: int = Field(ge=0)
    end: int = Field(gt=0)
    page_index: int | None = Field(default=None, ge=0)
    bbox: tuple[float, float, float, float] | None = None
    image_path: Path | None = None

    @model_validator(mode="after")
    def validate_offsets(self) -> OcrSpan:
        if self.end <= self.start:
            raise InvalidOcrSpanError(start=self.start, end=self.end)
        return self


@dataclass(frozen=True, slots=True)
class VlmReviewRequest:
    text: str
    spans: tuple[OcrSpan, ...]


class VlmVerdict(StrEnum):
    ACCEPT = "accept"
    REVISE = "revise"
    REJECT = "reject"


class VlmSuggestion(BaseModel):
    model_config = ConfigDict(frozen=True)

    span_id: str
    original_text: str
    corrected_text: str
    confidence: float = Field(ge=0, le=1)
    reason: str = ""


class VlmReviewResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    verdict: VlmVerdict
    suggestions: tuple[VlmSuggestion, ...] = Field(default=(), max_length=MAX_VLM_SUGGESTIONS)


class ReviewBackend(Protocol):
    def complete(self, request: VlmReviewRequest) -> str: ...


@dataclass(frozen=True, slots=True)
class ReviewedCorrection:
    raw_text: str
    corrected_text: str
    applied_span_ids: tuple[str, ...]
    issues: tuple[str, ...]
    requires_review: bool


def review_ocr_text(
    raw_text: str,
    spans: tuple[OcrSpan, ...],
    backend: ReviewBackend,
) -> ReviewedCorrection:
    normalized = correct_ocr_text(raw_text)
    duplicate_span_ids = _duplicate_values(span.span_id for span in spans)
    if duplicate_span_ids:
        return ReviewedCorrection(
            raw_text,
            normalized.corrected_text,
            (),
            tuple(f"duplicate_span:{span_id}" for span_id in duplicate_span_ids),
            True,
        )
    invalid_bounds = tuple(
        span.span_id for span in spans if span.start >= len(raw_text) or span.end > len(raw_text)
    )
    if invalid_bounds:
        return ReviewedCorrection(
            raw_text,
            normalized.corrected_text,
            (),
            tuple(f"invalid_span_bounds:{span_id}" for span_id in invalid_bounds),
            True,
        )
    eligible = tuple(span for span in spans if span.confidence < 0.8)
    if not eligible:
        return ReviewedCorrection(raw_text, normalized.corrected_text, (), (), False)

    issues: list[str] = []
    reviewable: list[OcrSpan] = []
    for span in eligible:
        if (
            raw_text[span.start : span.end] != span.text
            or normalized.corrected_text[span.start : span.end] != span.text
        ):
            issues.append(f"span_mismatch:{span.span_id}")
        elif raw_text[: span.end] != normalized.corrected_text[: span.end]:
            issues.append(f"unstable_span_offsets:{span.span_id}")
        else:
            reviewable.append(span)
    if not reviewable:
        return ReviewedCorrection(raw_text, normalized.corrected_text, (), tuple(issues), True)

    ordered_spans = tuple(sorted(reviewable, key=lambda span: span.start))
    overlaps = tuple(
        f"overlapping_spans:{left.span_id}:{right.span_id}"
        for left, right in pairwise(ordered_spans)
        if right.start < left.end
    )
    if overlaps:
        return ReviewedCorrection(
            raw_text,
            normalized.corrected_text,
            (),
            (*issues, *overlaps),
            True,
        )

    try:
        response = VlmReviewResponse.model_validate_json(
            backend.complete(VlmReviewRequest(normalized.corrected_text, ordered_spans))
        )
    except (OSError, ValueError, ValidationError):
        return ReviewedCorrection(
            raw_text,
            normalized.corrected_text,
            (),
            (*issues, "invalid_vlm_response"),
            True,
        )

    match response.verdict:
        case VlmVerdict.ACCEPT:
            return ReviewedCorrection(
                raw_text, normalized.corrected_text, (), tuple(issues), bool(issues)
            )
        case VlmVerdict.REJECT:
            return ReviewedCorrection(
                raw_text,
                normalized.corrected_text,
                (),
                (*issues, "vlm_rejected"),
                True,
            )
        case VlmVerdict.REVISE:
            pass
        case unreachable:
            assert_never(unreachable)

    duplicate_suggestion_ids = _duplicate_values(
        suggestion.span_id for suggestion in response.suggestions
    )
    if duplicate_suggestion_ids:
        return ReviewedCorrection(
            raw_text,
            normalized.corrected_text,
            (),
            (
                *issues,
                *(f"duplicate_suggestion:{span_id}" for span_id in duplicate_suggestion_ids),
            ),
            True,
        )

    by_id = {span.span_id: span for span in ordered_spans}
    accepted: list[tuple[OcrSpan, VlmSuggestion]] = []
    for suggestion in response.suggestions:
        span = by_id.get(suggestion.span_id)
        if span is None or suggestion.original_text != span.text or suggestion.confidence < 0.9:
            issues.append(f"untrusted_suggestion:{suggestion.span_id}")
            continue
        if extract_protected_values(span.text) != extract_protected_values(
            suggestion.corrected_text
        ):
            issues.append(f"protected_value_drift:{span.span_id}")
            continue
        accepted.append((span, suggestion))

    corrected = normalized.corrected_text
    for span, suggestion in sorted(accepted, key=lambda item: item[0].start, reverse=True):
        corrected = corrected[: span.start] + suggestion.corrected_text + corrected[span.end :]
    applied = tuple(span.span_id for span, _suggestion in accepted)
    unresolved = {span.span_id for span in ordered_spans} - set(applied)
    return ReviewedCorrection(
        raw_text=raw_text,
        corrected_text=corrected,
        applied_span_ids=applied,
        issues=tuple(issues),
        requires_review=bool(issues or unresolved),
    )


def _duplicate_values(values: Iterable[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    duplicates: list[str] = []
    for value in values:
        if value in seen:
            duplicates.append(value)
        else:
            seen.add(value)
    return tuple(duplicates)
