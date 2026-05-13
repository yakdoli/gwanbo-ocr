"""Small HTTP clients for containerized OCR/conversion services."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from gwanbo_ocr.runners.base import TranscriptionResult


class ServiceClientError(RuntimeError):
    """Raised when a configured OCR service returns an unusable response."""


class MarkItDownServiceClient:
    """Client for the MarkItDown OCR+LLM FastAPI service."""

    def __init__(self, base_url: str, *, timeout: float = 120) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def health(self) -> dict[str, Any]:
        return _get_json(_join_url(self.base_url, "/health"), timeout=self.timeout)

    def convert_path(
        self,
        file_path: str | Path,
        *,
        mode: str = "plain",
        output_path: str | Path | None = None,
        llm_model: str | None = None,
        llm_base_url: str | None = None,
        llm_api_key: str | None = None,
        llm_prompt: str | None = None,
    ) -> dict[str, Any]:
        payload = _drop_none(
            {
                "file_path": str(file_path),
                "output_path": str(output_path) if output_path is not None else None,
                "mode": mode,
                "llm_model": llm_model,
                "llm_base_url": llm_base_url,
                "llm_api_key": llm_api_key,
                "llm_prompt": llm_prompt,
            }
        )
        return _post_json(_join_url(self.base_url, "/convert/path"), payload, timeout=self.timeout)


class PaddleOcrServiceClient:
    """Client for the containerized PaddleOCR API service."""

    def __init__(self, base_url: str, *, timeout: float = 120) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def health(self) -> dict[str, Any]:
        return _get_json(_join_url(self.base_url, "/health"), timeout=self.timeout)

    def transcribe_classic(
        self,
        image_path: str | Path,
        *,
        lang: str = "korean",
        page_number: int | None = None,
    ) -> TranscriptionResult:
        payload = _post_json(
            _join_url(self.base_url, "/ocr/classic"),
            _drop_none(
                {
                    "image_path": str(image_path),
                    "lang": lang,
                    "page_number": page_number,
                }
            ),
            timeout=self.timeout,
        )
        return _result_from_service_payload(
            payload,
            page_number=page_number,
            backend="paddleocr_service",
        )

    def transcribe_vl(
        self,
        image_path: str | Path,
        *,
        page_number: int | None = None,
        pipeline_version: str = "v1.5",
        vl_rec_backend: str | None = None,
        vl_rec_server_url: str | None = None,
        vl_rec_api_model_name: str | None = None,
        vl_rec_api_key: str | None = None,
    ) -> TranscriptionResult:
        payload = _post_json(
            _join_url(self.base_url, "/ocr/vl"),
            _drop_none(
                {
                    "image_path": str(image_path),
                    "page_number": page_number,
                    "pipeline_version": pipeline_version,
                    "vl_rec_backend": vl_rec_backend,
                    "vl_rec_server_url": vl_rec_server_url,
                    "vl_rec_api_model_name": vl_rec_api_model_name,
                    "vl_rec_api_key": vl_rec_api_key,
                }
            ),
            timeout=self.timeout,
        )
        return _result_from_service_payload(
            payload,
            page_number=page_number,
            backend="paddleocr_vl_service",
        )


def _result_from_service_payload(
    payload: dict[str, Any],
    *,
    page_number: int | None,
    backend: str,
) -> TranscriptionResult:
    status = str(payload.get("status") or "ok")
    if status not in {"ok", "success"}:
        raise ServiceClientError(str(payload.get("error") or payload.get("detail") or status))
    normalized = dict(payload)
    normalized.setdefault("text", payload.get("markdown") or payload.get("markdown_content") or "")
    return TranscriptionResult.from_payload(
        normalized,
        raw_text=str(normalized.get("text") or ""),
        page_number=page_number,
        raw_response=payload,
        backend=backend,
    )


def _join_url(base_url: str, path: str) -> str:
    return f"{base_url.rstrip('/')}/{path.lstrip('/')}"


def _drop_none(values: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in values.items() if value is not None}


def _get_json(url: str, *, timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(url, method="GET")
    return _request_json(request, timeout=timeout)


def _post_json(url: str, payload: dict[str, Any], *, timeout: float) -> dict[str, Any]:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    return _request_json(request, timeout=timeout)


def _request_json(request: urllib.request.Request, *, timeout: float) -> dict[str, Any]:
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = response.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise ServiceClientError(f"{exc.code} {exc.reason}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise ServiceClientError(str(exc.reason)) from exc

    try:
        payload = json.loads(data.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise ServiceClientError("service returned non-JSON response") from exc
    if not isinstance(payload, dict):
        raise ServiceClientError("service returned non-object JSON response")
    return payload
