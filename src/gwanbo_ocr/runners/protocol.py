"""Compatibility exports for the shared runner protocol."""

from gwanbo_ocr.runners.base import (
    ImageInput,
    OCRRunner,
    OcrRunner,
    Runner,
    TranscriptionResult,
)

__all__ = ["ImageInput", "Runner", "OcrRunner", "OCRRunner", "TranscriptionResult"]
