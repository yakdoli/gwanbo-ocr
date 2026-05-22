"""Shared runner protocol and result helpers."""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

ImageInput = str | Path | bytes | bytearray | memoryview | Any


@dataclass(frozen=True)
class TranscriptionResult(Mapping[str, Any]):
    """Normalized OCR result returned by runner implementations."""

    data: Mapping[str, Any] = field(default_factory=dict)
    text: str = ""
    raw_text: str = ""
    page_number: int | None = None
    raw_response: Any = None
    backend: str | None = None

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, Any],
        *,
        raw_text: str = "",
        page_number: int | None = None,
        raw_response: Any = None,
        backend: str | None = None,
    ) -> TranscriptionResult:
        text = _payload_text(payload)
        return cls(
            data=dict(payload),
            text=text,
            raw_text=raw_text,
            page_number=page_number,
            raw_response=raw_response,
            backend=backend,
        )

    def to_dict(self) -> dict[str, Any]:
        result = dict(self.data)
        result.setdefault("text", self.text)
        if self.page_number is not None:
            result.setdefault("page_number", self.page_number)
        if self.backend:
            result.setdefault("backend", self.backend)
        if self.raw_text:
            result.setdefault("raw_text", self.raw_text)
        return result

    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.to_dict())

    def __len__(self) -> int:
        return len(self.to_dict())


@runtime_checkable
class Runner(Protocol):
    """Protocol implemented by OCR backends."""

    def transcribe(
        self,
        image: ImageInput,
        *,
        page_number: int | None = None,
        **kwargs: Any,
    ) -> TranscriptionResult:
        """Transcribe one rendered page image."""


OcrRunner = Runner
OCRRunner = Runner


def parse_json_response(text: str, *, strict: bool = True) -> dict[str, Any]:
    """Parse a JSON-only model response.

    ``strict=False`` returns ``{"text": text}`` when JSON extraction fails.
    This is useful while developing against models that sometimes leak prose.
    """

    candidate = _strip_reasoning_blocks(_strip_code_fence(text.strip()))
    decoder = json.JSONDecoder()

    try:
        parsed = json.loads(candidate)
        return _normalize_json_payload(parsed)
    except json.JSONDecodeError:
        pass

    for start, char in enumerate(candidate):
        if char not in "[{":
            continue
        try:
            parsed, _ = decoder.raw_decode(candidate[start:])
            return _normalize_json_payload(parsed)
        except json.JSONDecodeError:
            continue

    if strict:
        raise ValueError("Model response did not contain valid JSON")
    return {"text": candidate, "blocks": [{"text": candidate}], "warnings": ["non_json_response"]}


def _strip_code_fence(text: str) -> str:
    if not text.startswith("```"):
        return text
    lines = text.splitlines()
    if len(lines) >= 2 and lines[-1].strip() == "```":
        return "\n".join(lines[1:-1]).strip()
    return text


def _strip_reasoning_blocks(text: str) -> str:
    candidate = text.lstrip()
    while candidate.startswith("<think>"):
        end = candidate.find("</think>")
        if end < 0:
            return text
        candidate = candidate[end + len("</think>") :].lstrip()
    return candidate


def _normalize_json_payload(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, list):
        return {"items": value, "text": _text_from_list(value)}
    return {"value": value, "text": str(value)}


def _payload_text(payload: Mapping[str, Any]) -> str:
    for key in ("text", "transcription", "markdown", "content"):
        value = payload.get(key)
        if isinstance(value, str):
            return value
    blocks = payload.get("blocks")
    if isinstance(blocks, list):
        texts = [
            block.get("text", "")
            for block in blocks
            if isinstance(block, Mapping) and isinstance(block.get("text"), str)
        ]
        return "\n".join(text for text in texts if text)
    return ""


def _text_from_list(values: list[Any]) -> str:
    texts: list[str] = []
    for value in values:
        if isinstance(value, str):
            texts.append(value)
        elif isinstance(value, Mapping) and isinstance(value.get("text"), str):
            texts.append(value["text"])
    return "\n".join(texts)
