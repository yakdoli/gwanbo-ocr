"""Compatibility decision for native-text recovery and OCR routing."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class PdfTextClass(StrEnum):
    ERROR = "error"
    DIGITAL_TEXT_RECOVERED = "digital_text_recovered"
    TEXT_EXTRACTABLE = "text_extractable"
    DIGITAL_TEXT_UNRECOVERED = "digital_text_unrecovered"
    IMAGE_OR_SCANNED = "image_or_scanned"
    UNKNOWN_UNEXTRACTABLE = "unknown_unextractable"


@dataclass(frozen=True, slots=True)
class PdfTextSignals:
    status_error: bool = False
    recovered_text: bool = False
    text_extractable: bool = False
    has_digital_evidence: bool = False
    has_images: bool = False


@dataclass(frozen=True, slots=True)
class PdfTextDecision:
    pdf_text_class: PdfTextClass
    needs_ocr: bool


def classify_pdf_text_metadata(signals: PdfTextSignals) -> PdfTextDecision:
    if signals.status_error:
        return PdfTextDecision(PdfTextClass.ERROR, True)
    if signals.recovered_text:
        return PdfTextDecision(PdfTextClass.DIGITAL_TEXT_RECOVERED, False)
    if signals.text_extractable:
        return PdfTextDecision(PdfTextClass.TEXT_EXTRACTABLE, False)
    if signals.has_digital_evidence:
        return PdfTextDecision(PdfTextClass.DIGITAL_TEXT_UNRECOVERED, True)
    if signals.has_images:
        return PdfTextDecision(PdfTextClass.IMAGE_OR_SCANNED, True)
    return PdfTextDecision(PdfTextClass.UNKNOWN_UNEXTRACTABLE, True)
