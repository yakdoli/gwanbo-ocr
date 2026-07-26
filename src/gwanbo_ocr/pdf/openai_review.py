"""OpenAI-compatible transport for gated OCR span review."""

from __future__ import annotations

import base64
import json
import mimetypes
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final, Literal, Protocol
from urllib.error import HTTPError
from urllib.parse import urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener

from pydantic import BaseModel, ConfigDict

from .io import UnsafePathError
from .limits import MAX_VLM_SPANS
from .vlm_correction import VlmReviewRequest

MAX_VLM_RESPONSE_BYTES: Final = 2 * 1024 * 1024
MAX_CROP_BYTES: Final = 10 * 1024 * 1024
MAX_VLM_TEXT_BYTES: Final = 2 * 1024 * 1024
MAX_VLM_CROPS: Final = 16
MAX_VLM_REQUEST_BYTES: Final = 16 * 1024 * 1024


class VlmResponseError(ValueError):
    __slots__ = ("message",)

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message

    message: str

    def __str__(self) -> str:
        return self.message


class ImageResourceLimitError(ValueError):
    __slots__ = ("path",)

    def __init__(self, path: str) -> None:
        super().__init__(path)
        self.path = path

    path: str

    def __str__(self) -> str:
        return f"VLM crop exceeds 10 MiB: {self.path}"


class JsonSender(Protocol):
    def post(self, url: str, headers: dict[str, str], body: bytes, timeout: float) -> bytes: ...


class _RejectRedirects(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl) -> None:
        del newurl
        raise HTTPError(req.full_url, code, msg, headers, fp)


class UrlLibJsonSender:
    def post(self, url: str, headers: dict[str, str], body: bytes, timeout: float) -> bytes:
        request = Request(url, data=body, headers=headers, method="POST")
        with build_opener(_RejectRedirects()).open(request, timeout=timeout) as response:
            payload = response.read(MAX_VLM_RESPONSE_BYTES + 1)
        if len(payload) > MAX_VLM_RESPONSE_BYTES:
            raise VlmResponseError("VLM response exceeds 2 MiB")
        return payload


class _TextPart(BaseModel):
    model_config = ConfigDict(frozen=True)

    type: Literal["text"] = "text"
    text: str


class _ImageUrl(BaseModel):
    model_config = ConfigDict(frozen=True)

    url: str


class _ImagePart(BaseModel):
    model_config = ConfigDict(frozen=True)

    type: Literal["image_url"] = "image_url"
    image_url: _ImageUrl


class _UserMessage(BaseModel):
    model_config = ConfigDict(frozen=True)

    role: Literal["user"] = "user"
    content: tuple[_TextPart | _ImagePart, ...]


class _ResponseFormat(BaseModel):
    model_config = ConfigDict(frozen=True)

    type: Literal["json_object"] = "json_object"


class _CompletionRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    model: str
    messages: tuple[_UserMessage, ...]
    temperature: float = 0.0
    response_format: _ResponseFormat = _ResponseFormat()


class _ResponseMessage(BaseModel):
    content: str


class _ResponseChoice(BaseModel):
    message: _ResponseMessage


class _CompletionResponse(BaseModel):
    choices: tuple[_ResponseChoice, ...]


@dataclass(frozen=True, slots=True)
class OpenAICompatibleReviewBackend:
    base_url: str
    model: str
    api_key: str = field(default="EMPTY", repr=False)
    timeout: float = 30
    crop_root: Path | None = None
    sender: JsonSender = field(default_factory=UrlLibJsonSender)

    def __post_init__(self) -> None:
        parsed = urlparse(self.base_url)
        if parsed.hostname is None:
            raise VlmResponseError("VLM base URL must include a host")
        if parsed.scheme == "https":
            return
        if parsed.scheme == "http" and parsed.hostname in {"localhost", "127.0.0.1", "::1"}:
            return
        raise VlmResponseError("VLM base URL must use HTTPS unless host is localhost")

    def complete(self, request: VlmReviewRequest) -> str:
        if len(request.text.encode()) > MAX_VLM_TEXT_BYTES:
            raise VlmResponseError("VLM text exceeds 2 MiB")
        if len(request.spans) > MAX_VLM_SPANS:
            raise VlmResponseError("VLM span count exceeds 1000")
        image_paths = tuple(
            dict.fromkeys(span.image_path for span in request.spans if span.image_path is not None)
        )
        if len(image_paths) > MAX_VLM_CROPS:
            raise VlmResponseError("VLM crop count exceeds 16")
        prompt = json.dumps(
            {
                "task": "Review only the supplied OCR spans. Return verdict and span suggestions.",
                "text": request.text,
                "spans": [
                    span.model_dump(mode="json", exclude={"image_path"}) for span in request.spans
                ],
                "response_schema": {
                    "verdict": "accept|revise|reject",
                    "suggestions": [
                        {
                            "span_id": "string",
                            "original_text": "string",
                            "corrected_text": "string",
                            "confidence": 0.0,
                            "reason": "string",
                        }
                    ],
                },
            },
            ensure_ascii=False,
        )
        parts: list[_TextPart | _ImagePart] = [_TextPart(text=prompt)]
        request_bytes = len(prompt.encode())
        for image_path in image_paths:
            image_url = _image_data_url(image_path, self.crop_root)
            request_bytes += len(image_url.encode())
            if request_bytes > MAX_VLM_REQUEST_BYTES:
                raise VlmResponseError("VLM request exceeds 16 MiB")
            parts.append(_ImagePart(image_url=_ImageUrl(url=image_url)))
        payload = (
            _CompletionRequest(
                model=self.model,
                messages=(_UserMessage(content=tuple(parts)),),
            )
            .model_dump_json(exclude_none=True)
            .encode()
        )
        if len(payload) > MAX_VLM_REQUEST_BYTES:
            raise VlmResponseError("VLM request exceeds 16 MiB")
        response = self.sender.post(
            f"{self.base_url.rstrip('/')}/chat/completions",
            {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            payload,
            self.timeout,
        )
        parsed = _CompletionResponse.model_validate_json(response)
        if not parsed.choices:
            raise VlmResponseError("VLM response missing message content")
        return parsed.choices[0].message.content


def _image_data_url(path: Path, crop_root: Path | None) -> str:
    trusted_path = _trusted_crop_path(path, crop_root)
    with trusted_path.open("rb") as handle:
        data = handle.read(MAX_CROP_BYTES + 1)
    if len(data) > MAX_CROP_BYTES:
        raise ImageResourceLimitError(str(trusted_path))
    mime_type = mimetypes.guess_type(trusted_path.name)[0] or "application/octet-stream"
    return f"data:{mime_type};base64,{base64.b64encode(data).decode()}"


def _trusted_crop_path(path: Path, crop_root: Path | None) -> Path:
    if crop_root is None:
        raise UnsafePathError(str(path), "trusted crop root is required")
    if path.is_symlink():
        raise UnsafePathError(str(path), "symlinks are not allowed")
    root = Path(crop_root).resolve(strict=False)
    resolved = path.resolve(strict=False)
    if not resolved.is_relative_to(root):
        raise UnsafePathError(str(path), f"outside trusted root {root}")
    return resolved
