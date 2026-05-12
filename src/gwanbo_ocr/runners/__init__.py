"""OCR runner implementations."""

from gwanbo_ocr.runners.base import (
    OCRRunner,
    OcrRunner,
    Runner,
    TranscriptionResult,
    parse_json_response,
)
from gwanbo_ocr.runners.paddle import PaddleOCRRunner, PaddleOcrRunner, PaddleRunner
from gwanbo_ocr.runners.vllm import VLLMChatRunner, VllmChatRunner

__all__ = [
    "Runner",
    "OcrRunner",
    "OCRRunner",
    "TranscriptionResult",
    "parse_json_response",
    "VllmChatRunner",
    "VLLMChatRunner",
    "PaddleOcrRunner",
    "PaddleOCRRunner",
    "PaddleRunner",
]
