"""Compatibility export for PaddleOCR runner adapters."""

from gwanbo_ocr.runners.paddle import PaddleOCRRunner, PaddleOcrRunner, PaddleRunner

__all__ = ["PaddleOcrRunner", "PaddleOCRRunner", "PaddleRunner"]
