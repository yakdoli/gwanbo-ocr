"""Runner adapters for the containerized PaddleOCR HTTP service."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from gwanbo_ocr.runners.base import ImageInput, TranscriptionResult
from gwanbo_ocr.services import PaddleOcrServiceClient


class PaddleOcrServiceRunner:
    """Classic PaddleOCR runner backed by the `/ocr/classic` service endpoint."""

    def __init__(self, base_url: str, *, lang: str = "korean", timeout: float = 120) -> None:
        self.client = PaddleOcrServiceClient(base_url, timeout=timeout)
        self.lang = lang

    def transcribe(
        self,
        image: ImageInput,
        *,
        page_number: int | None = None,
        **_kwargs: Any,
    ) -> TranscriptionResult:
        return self.client.transcribe_classic(
            _image_path(image),
            lang=self.lang,
            page_number=page_number,
        )


class PaddleOcrVlServiceRunner:
    """PaddleOCR-VL runner backed by the `/ocr/vl` service endpoint."""

    def __init__(
        self,
        base_url: str,
        *,
        pipeline_version: str = "v1.5",
        vl_rec_backend: str | None = None,
        vl_rec_server_url: str | None = None,
        vl_rec_api_model_name: str | None = None,
        vl_rec_api_key: str | None = None,
        timeout: float = 180,
    ) -> None:
        self.client = PaddleOcrServiceClient(base_url, timeout=timeout)
        self.pipeline_version = pipeline_version
        self.vl_rec_backend = vl_rec_backend
        self.vl_rec_server_url = vl_rec_server_url
        self.vl_rec_api_model_name = vl_rec_api_model_name
        self.vl_rec_api_key = vl_rec_api_key

    def transcribe(
        self,
        image: ImageInput,
        *,
        page_number: int | None = None,
        **_kwargs: Any,
    ) -> TranscriptionResult:
        return self.client.transcribe_vl(
            _image_path(image),
            page_number=page_number,
            pipeline_version=self.pipeline_version,
            vl_rec_backend=self.vl_rec_backend,
            vl_rec_server_url=self.vl_rec_server_url,
            vl_rec_api_model_name=self.vl_rec_api_model_name,
            vl_rec_api_key=self.vl_rec_api_key,
        )


def _image_path(image: ImageInput) -> Path:
    if isinstance(image, Path):
        return image
    if isinstance(image, str):
        return Path(image)
    raise TypeError("PaddleOCR service runners require image paths")
