"""Prompts and message builders for page transcription."""

from __future__ import annotations

import base64
import io
import json
import mimetypes
from collections.abc import Mapping
from pathlib import Path
from typing import Any

TRANSCRIPTION_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["text", "blocks"],
    "properties": {
        "text": {
            "type": "string",
            "description": "Full page transcription in natural reading order.",
        },
        "blocks": {
            "type": "array",
            "description": "Logical text blocks in reading order.",
            "items": {
                "type": "object",
                "required": ["text"],
                "properties": {
                    "text": {"type": "string"},
                    "bbox": {
                        "type": ["array", "null"],
                        "items": {"type": "number"},
                        "minItems": 4,
                        "maxItems": 4,
                        "description": "Optional [x0, y0, x1, y1] image bbox.",
                    },
                    "confidence": {"type": ["number", "null"]},
                },
            },
        },
        "tables": {
            "type": "array",
            "description": "Tables preserved as markdown where present.",
            "items": {
                "type": "object",
                "required": ["markdown"],
                "properties": {
                    "markdown": {"type": "string"},
                    "bbox": {
                        "type": ["array", "null"],
                        "items": {"type": "number"},
                        "minItems": 4,
                        "maxItems": 4,
                    },
                },
            },
        },
        "warnings": {
            "type": "array",
            "description": "Uncertainties such as illegible text or rotated areas.",
            "items": {"type": "string"},
        },
    },
}


JSON_ONLY_TRANSCRIPTION_PROMPT = (
    "Return only valid JSON. Do not include markdown fences, prose, comments, "
    "or field descriptions."
)


TRANSCRIPTION_SYSTEM_PROMPT = "\n".join(
    [
        "You are an OCR transcription engine for scanned document pages.",
        "Transcribe visible text faithfully in natural reading order.",
        "Preserve line breaks when they carry meaning.",
        "Represent tables as markdown in the tables array and include their text",
        "in the full text field.",
        "Do not guess illegible content; use warnings for uncertainty.",
        JSON_ONLY_TRANSCRIPTION_PROMPT,
    ]
)


TRANSCRIPTION_USER_PROMPT = "\n".join(
    [
        "Transcribe the attached page image.",
        "Return exactly one JSON object in this shape:",
        '{"text":"...","blocks":[{"text":"...","bbox":null,'
        '"confidence":null}],"tables":[],"warnings":[]}',
        "Do not return a JSON schema. Do not repeat these instructions.",
    ]
)


def schema_to_prompt(schema: Mapping[str, Any] | None = None) -> str:
    """Serialize the transcription schema for inclusion in a prompt."""

    return json.dumps(schema or TRANSCRIPTION_JSON_SCHEMA, ensure_ascii=False, indent=2)


def build_transcription_prompt(
    *,
    page_number: int | None = None,
    language_hint: str | None = None,
    schema: Mapping[str, Any] | None = None,
) -> str:
    """Build the JSON-only user prompt for one page image."""

    prompt = TRANSCRIPTION_USER_PROMPT
    if schema is not None and schema is not TRANSCRIPTION_JSON_SCHEMA:
        prompt = "\n".join(
            [
                prompt,
                "Additional field contract for reference only; do not output it:",
                schema_to_prompt(schema),
            ]
        )
    context: list[str] = []
    if page_number is not None:
        context.append(f"Page number: {page_number}.")
    if language_hint:
        context.append(f"Language hint: {language_hint}.")
    if context:
        prompt = "\n".join([*context, prompt])
    return prompt


def image_to_data_url(image: Any, *, mime_type: str | None = None) -> str:
    """Convert common image inputs to a data URL for chat-completion APIs.

    ``image`` may be an HTTP(S) URL, an existing data URL, bytes, a filesystem
    path, or a Pillow image object.
    """

    if isinstance(image, str):
        if image.startswith(("data:", "http://", "https://")):
            return image
        return _path_to_data_url(Path(image), mime_type=mime_type)

    if isinstance(image, Path):
        return _path_to_data_url(image, mime_type=mime_type)

    if isinstance(image, (bytes, bytearray, memoryview)):
        data = bytes(image)
        return _bytes_to_data_url(data, mime_type=mime_type or "image/png")

    if hasattr(image, "save"):
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return _bytes_to_data_url(buffer.getvalue(), mime_type=mime_type or "image/png")

    raise TypeError(
        "image must be a URL, data URL, bytes, pathlib.Path, path string, or Pillow image"
    )


def build_transcription_messages(
    image: Any | None = None,
    *,
    page_number: int | None = None,
    language_hint: str | None = None,
    schema: Mapping[str, Any] | None = None,
    system_prompt: str = TRANSCRIPTION_SYSTEM_PROMPT,
    user_prompt: str | None = None,
    image_mime_type: str | None = None,
) -> list[dict[str, Any]]:
    """Build OpenAI/vLLM chat messages for OCR transcription."""

    prompt = user_prompt or build_transcription_prompt(
        page_number=page_number,
        language_hint=language_hint,
        schema=schema,
    )

    if image is None:
        user_content: str | list[dict[str, Any]] = prompt
    else:
        user_content = [
            {"type": "text", "text": prompt},
            {
                "type": "image_url",
                "image_url": {
                    "url": image_to_data_url(image, mime_type=image_mime_type),
                },
            },
        ]

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]


def _path_to_data_url(path: Path, *, mime_type: str | None) -> str:
    expanded = path.expanduser()
    guessed_type = mime_type or mimetypes.guess_type(expanded.name)[0] or "image/png"
    return _bytes_to_data_url(expanded.read_bytes(), mime_type=guessed_type)


def _bytes_to_data_url(data: bytes, *, mime_type: str) -> str:
    encoded = base64.b64encode(data).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


# Friendly aliases for likely call sites.
TRANSCRIBE_JSON_PROMPT = JSON_ONLY_TRANSCRIPTION_PROMPT
build_prompt = build_transcription_prompt
build_messages = build_transcription_messages
