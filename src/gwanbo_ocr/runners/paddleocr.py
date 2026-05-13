"""Compatibility export for PaddleOCR runner adapters."""

from gwanbo_ocr.runners.paddle import (
    PaddleOCRRunner,
    PaddleOcrRunner,
    PaddleOCRVLRunner,
    PaddleOcrVLRunner,
    PaddleRunner,
)

__all__ = [
    "PaddleOcrRunner",
    "PaddleOcrVLRunner",
    "PaddleOCRRunner",
    "PaddleOCRVLRunner",
    "PaddleRunner",
]
