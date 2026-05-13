from __future__ import annotations

from pathlib import Path
from typing import Any

from ._helpers import SAMPLE_CHARS_DEFAULT, _normalize, _skipped, _with_timeout

_MarkItDown: Any
try:
    from markitdown import MarkItDown as _MarkItDown  # type: ignore[import-not-found]
except ImportError:
    _MarkItDown = None

_OpenAI: Any
try:
    from openai import OpenAI as _OpenAI  # type: ignore[import-not-found]
except ImportError:
    _OpenAI = None


def extract_markitdown(
    pdf_path: Path,
    *,
    sample_chars: int = SAMPLE_CHARS_DEFAULT,
    timeout_seconds: int = 30,
) -> dict[str, Any]:
    """Extract text using MarkItDown.convert()."""
    MarkItDown = _MarkItDown
    if MarkItDown is None:
        return _skipped("markitdown is not installed", method="MarkItDown.convert")

    def _run() -> dict[str, Any]:
        converter = MarkItDown(enable_plugins=False)
        converted = converter.convert(str(pdf_path))
        text = _normalize(str(getattr(converted, "text_content", "") or ""))
        return {
            "status": "ok",
            "method": "MarkItDown.convert",
            "text_extractable": bool(text),
            "text_chars": len(text),
            "sample_text": text[:sample_chars],
            "error": None,
        }

    return _with_timeout(_run, timeout_seconds, method="MarkItDown.convert")


def extract_markitdown_ocr_llm(
    pdf_path: Path,
    *,
    sample_chars: int = SAMPLE_CHARS_DEFAULT,
    timeout_seconds: int = 120,
    service_url: str | None = None,
    llm_base_url: str | None = None,
    llm_model: str | None = None,
    llm_api_key: str = "dummy",
    llm_prompt: str | None = None,
) -> dict[str, Any]:
    """Extract text using MarkItDown's OCR plugin with an OpenAI-compatible VLM."""
    if service_url:
        return _extract_markitdown_ocr_llm_service(
            pdf_path,
            sample_chars=sample_chars,
            timeout_seconds=timeout_seconds,
            service_url=service_url,
            llm_base_url=llm_base_url,
            llm_model=llm_model,
            llm_api_key=llm_api_key,
            llm_prompt=llm_prompt,
        )

    MarkItDown = _MarkItDown
    OpenAI = _OpenAI
    if MarkItDown is None:
        return _skipped("markitdown is not installed", method="MarkItDown.ocr_llm")
    if OpenAI is None:
        return _skipped("openai is not installed", method="MarkItDown.ocr_llm")
    if not llm_model:
        return _skipped("llm_model is required for OCR+LLM mode", method="MarkItDown.ocr_llm")

    def _run() -> dict[str, Any]:
        client_kwargs = {"api_key": llm_api_key}
        if llm_base_url:
            client_kwargs["base_url"] = llm_base_url
        kwargs: dict[str, Any] = {
            "enable_plugins": True,
            "llm_client": OpenAI(**client_kwargs),
            "llm_model": llm_model,
        }
        if llm_prompt:
            kwargs["llm_prompt"] = llm_prompt
        converter = MarkItDown(**kwargs)
        converted = converter.convert(str(pdf_path))
        text = _normalize(str(getattr(converted, "text_content", "") or ""))
        return {
            "status": "ok",
            "method": "MarkItDown.ocr_llm",
            "text_extractable": bool(text),
            "text_chars": len(text),
            "sample_text": text[:sample_chars],
            "llm_base_url": llm_base_url,
            "llm_model": llm_model,
            "service_url": None,
            "error": None,
        }

    return _with_timeout(_run, timeout_seconds, method="MarkItDown.ocr_llm")


def _extract_markitdown_ocr_llm_service(
    pdf_path: Path,
    *,
    sample_chars: int,
    timeout_seconds: int,
    service_url: str,
    llm_base_url: str | None,
    llm_model: str | None,
    llm_api_key: str,
    llm_prompt: str | None,
) -> dict[str, Any]:
    def _run() -> dict[str, Any]:
        from gwanbo_ocr.services import MarkItDownServiceClient

        client = MarkItDownServiceClient(service_url, timeout=timeout_seconds)
        converted = client.convert_path(
            pdf_path,
            mode="ocr-llm",
            llm_base_url=llm_base_url,
            llm_model=llm_model,
            llm_api_key=llm_api_key,
            llm_prompt=llm_prompt,
        )
        text = _normalize(_text_from_service_payload(converted))
        return {
            "status": "ok",
            "method": "MarkItDown.ocr_llm",
            "text_extractable": bool(text),
            "text_chars": len(text),
            "sample_text": text[:sample_chars],
            "llm_base_url": llm_base_url,
            "llm_model": llm_model,
            "service_url": service_url,
            "error": None,
        }

    return _with_timeout(_run, timeout_seconds, method="MarkItDown.ocr_llm")


def _text_from_service_payload(payload: dict[str, Any]) -> str:
    for key in ("markdown_content", "text_content", "markdown", "text"):
        value = payload.get(key)
        if isinstance(value, str):
            return value
    return ""
