"""Auditable, deterministic normalization for extracted OCR text."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from enum import StrEnum
from typing import Final

PROTECTED_VALUE_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"\d{1,3}(?:,\d{3})+|\d{4}-\d{2}-\d{2}|\d+"
)
ZERO_WIDTH_PATTERN: Final[re.Pattern[str]] = re.compile(r"[\u200b\u200c\u200d\ufeff]")
HORIZONTAL_SPACE_PATTERN: Final[re.Pattern[str]] = re.compile(r"[^\S\n]+")


class CorrectionKind(StrEnum):
    """Safe normalization stage applied to OCR text."""

    UNICODE_NFC = "unicode_nfc"
    LINE_ENDINGS = "line_endings"
    ZERO_WIDTH = "zero_width"
    HORIZONTAL_SPACE = "horizontal_space"


@dataclass(frozen=True, slots=True)
class CorrectionChange:
    """One deterministic transformation with its before/after values."""

    kind: CorrectionKind
    before: str
    after: str


@dataclass(frozen=True, slots=True)
class CorrectionResult:
    """Raw OCR, corrected OCR, and a complete transformation audit."""

    raw_text: str
    corrected_text: str
    changes: tuple[CorrectionChange, ...]
    protected_values: tuple[str, ...]
    requires_review: bool


def correct_ocr_text(raw_text: str) -> CorrectionResult:
    """Apply only lossless OCR normalizations and retain the original text."""
    corrected = raw_text
    changes: list[CorrectionChange] = []

    normalized = unicodedata.normalize("NFC", corrected)
    if normalized != corrected:
        changes.append(CorrectionChange(CorrectionKind.UNICODE_NFC, corrected, normalized))
        corrected = normalized

    normalized = corrected.replace("\r\n", "\n").replace("\r", "\n")
    if normalized != corrected:
        changes.append(CorrectionChange(CorrectionKind.LINE_ENDINGS, corrected, normalized))
        corrected = normalized

    normalized = ZERO_WIDTH_PATTERN.sub("", corrected)
    if normalized != corrected:
        changes.append(CorrectionChange(CorrectionKind.ZERO_WIDTH, corrected, normalized))
        corrected = normalized

    normalized = HORIZONTAL_SPACE_PATTERN.sub(" ", corrected)
    if normalized != corrected:
        changes.append(CorrectionChange(CorrectionKind.HORIZONTAL_SPACE, corrected, normalized))
        corrected = normalized

    protected_values = extract_protected_values(raw_text)
    corrected_values = extract_protected_values(corrected)
    return CorrectionResult(
        raw_text=raw_text,
        corrected_text=corrected,
        changes=tuple(changes),
        protected_values=protected_values,
        requires_review=protected_values != corrected_values,
    )


def extract_protected_values(text: str) -> tuple[str, ...]:
    return tuple(PROTECTED_VALUE_PATTERN.findall(text))
