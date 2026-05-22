"""OpenAI-compatible vLLM chat runner."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any
from urllib.parse import urljoin

from gwanbo_ocr.prompts import (
    TRANSCRIPTION_JSON_SCHEMA,
    TRANSCRIPTION_SYSTEM_PROMPT,
    build_transcription_messages,
)
from gwanbo_ocr.runners.base import (
    ImageInput,
    TranscriptionResult,
    parse_json_response,
)


class VllmChatRunner:
    """Transcribe page images through a vLLM OpenAI-compatible chat server."""

    def __init__(
        self,
        *,
        model: str,
        base_url: str = "http://localhost:8000/v1",
        api_key: str = "EMPTY",
        client: Any | None = None,
        http_client: Any | None = None,
        timeout: float | None = 120.0,
        temperature: float = 0.0,
        top_p: float | None = None,
        max_tokens: int | None = 4096,
        response_format: Mapping[str, Any] | None = {"type": "json_object"},
        strict_json: bool = True,
        use_httpx: bool = False,
        extra_body: Mapping[str, Any] | None = None,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.response_format = None if response_format is None else dict(response_format)
        self.strict_json = strict_json
        self.extra_body = dict(extra_body or {})
        self._client = client
        self._http_client = http_client
        self._use_httpx = use_httpx

    @property
    def client(self) -> Any:
        if self._client is None:
            self._client = self._make_client()
        return self._client

    def transcribe(
        self,
        image: ImageInput,
        *,
        page_number: int | None = None,
        language_hint: str | None = None,
        schema: Mapping[str, Any] | None = TRANSCRIPTION_JSON_SCHEMA,
        user_prompt: str | None = None,
        system_prompt: str | None = TRANSCRIPTION_SYSTEM_PROMPT,
        image_mime_type: str | None = None,
        **request_overrides: Any,
    ) -> TranscriptionResult:
        messages = build_transcription_messages(
            image,
            page_number=page_number,
            language_hint=language_hint,
            schema=schema,
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            image_mime_type=image_mime_type,
        )
        payload = self._build_payload(messages, request_overrides)
        raw_response = self._create_completion(payload)
        raw_text = _extract_chat_content(raw_response)
        parsed = parse_json_response(raw_text, strict=self.strict_json)
        _validate_transcription_payload(parsed)
        return TranscriptionResult.from_payload(
            parsed,
            raw_text=raw_text,
            page_number=page_number,
            raw_response=raw_response,
            backend="vllm-chat",
        )

    def run(self, image: ImageInput, **kwargs: Any) -> TranscriptionResult:
        """Alias for ``transcribe`` used by simple pipeline code."""

        return self.transcribe(image, **kwargs)

    def transcribe_json(self, image: ImageInput, **kwargs: Any) -> dict[str, Any]:
        """Transcribe and return only the normalized JSON payload."""

        return dict(self.transcribe(image, **kwargs).data)

    def _make_client(self) -> Any:
        if self._use_httpx:
            import httpx

            return httpx.Client(base_url=self.base_url, timeout=self.timeout)

        try:
            from openai import OpenAI  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise RuntimeError(
                "The openai package is required for VllmChatRunner unless "
                "client= or use_httpx=True is provided."
            ) from exc

        kwargs: dict[str, Any] = {
            "base_url": self.base_url,
            "api_key": self.api_key,
        }
        if self.timeout is not None:
            kwargs["timeout"] = self.timeout
        if self._http_client is not None:
            kwargs["http_client"] = self._http_client
        return OpenAI(**kwargs)

    def _build_payload(
        self,
        messages: list[dict[str, Any]],
        overrides: Mapping[str, Any],
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
        }
        if self.top_p is not None:
            payload["top_p"] = self.top_p
        if self.max_tokens is not None:
            payload["max_tokens"] = self.max_tokens
        if self.response_format is not None:
            payload["response_format"] = dict(self.response_format)
        payload.update(overrides)
        return payload

    def _create_completion(self, payload: Mapping[str, Any]) -> Any:
        client = self.client
        if _is_openai_chat_client(client):
            create_kwargs = dict(payload)
            if self.extra_body:
                create_kwargs["extra_body"] = dict(self.extra_body)
            return client.chat.completions.create(**create_kwargs)

        if hasattr(client, "post"):
            http_payload = dict(payload)
            http_payload.update(self.extra_body)
            response = client.post(
                _chat_completions_url(self.base_url),
                json=http_payload,
            )
            if hasattr(response, "raise_for_status"):
                response.raise_for_status()
            if hasattr(response, "json"):
                return response.json()
            return response

        raise TypeError("client must expose chat.completions.create(...) or post(...).")


def _is_openai_chat_client(client: Any) -> bool:
    chat = getattr(client, "chat", None)
    completions = getattr(chat, "completions", None)
    return hasattr(completions, "create")


def _chat_completions_url(base_url: str) -> str:
    """Return the chat completions endpoint for a /v1-prefixed base URL."""
    return urljoin(base_url.rstrip("/"), "chat/completions")


def _extract_chat_content(response: Any) -> str:
    if isinstance(response, Mapping):
        choice = response["choices"][0]
        message = choice.get("message", choice)
        content = message.get("content", "")
        text = _content_to_text(content)
        if text:
            return text
        reasoning = message.get("reasoning_content", "")
        if reasoning:
            return _content_to_text(reasoning)

    choices = getattr(response, "choices", None)
    if choices:
        choice = choices[0]
        message = getattr(choice, "message", choice)
        content = getattr(message, "content", "")
        text = _content_to_text(content)
        if text:
            return text
        reasoning = getattr(message, "reasoning_content", "")
        if reasoning:
            return _content_to_text(reasoning)

    content = getattr(response, "content", None)
    if content is not None:
        return _content_to_text(content)

    raise ValueError("Could not extract chat completion content from response")


def _validate_transcription_payload(payload: Mapping[str, Any]) -> None:
    if _looks_like_json_schema(payload):
        raise ValueError("Model returned the output schema instead of OCR content")
    text = payload.get("text")
    if not isinstance(text, str):
        raise ValueError("Model response is missing a string 'text' field")
    blocks = payload.get("blocks")
    if blocks is not None and not isinstance(blocks, list):
        raise ValueError("Model response 'blocks' field must be an array")
    tables = payload.get("tables")
    if tables is not None and not isinstance(tables, list):
        raise ValueError("Model response 'tables' field must be an array")


def _looks_like_json_schema(payload: Mapping[str, Any]) -> bool:
    return (
        payload.get("type") == "object"
        and isinstance(payload.get("properties"), Mapping)
        and not isinstance(payload.get("text"), str)
    )


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, Mapping):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(parts)
    if content is None:
        return ""
    return str(content)


VLLMChatRunner = VllmChatRunner
