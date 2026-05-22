from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from gwanbo_ocr.prompts import build_transcription_prompt
from gwanbo_ocr.runners.vllm import VllmChatRunner


class _Response:
    def __init__(self, content: str) -> None:
        self._content = content

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return {"choices": [{"message": {"content": self._content}}]}


class _PostClient:
    def __init__(self, content: str) -> None:
        self.content = content

    def post(self, _url: str, *, json: dict[str, Any]) -> _Response:
        return _Response(self.content)


def test_transcription_prompt_discourages_schema_echo() -> None:
    prompt = build_transcription_prompt(page_number=1, language_hint="ko,en")

    assert "Do not return a JSON schema" in prompt
    assert '"text":"..."' in prompt
    assert '"properties"' not in prompt


def test_vllm_runner_rejects_schema_echo() -> None:
    runner = VllmChatRunner(
        model="model",
        client=_PostClient('{"type":"object","properties":{"text":{"type":"string"}}}'),
        strict_json=False,
    )

    with pytest.raises(ValueError, match="schema"):
        runner.transcribe(b"fake-png-bytes")


class _FakeOpenAICompletions:
    def __init__(self, content: str) -> None:
        self.content = content

    def create(self, **kwargs: Any) -> Any:
        return type(
            "FakeCompletion",
            (),
            {
                "choices": [
                    type(
                        "FakeChoice",
                        (),
                        {"message": type("FakeMsg", (), {"content": self.content})()},
                    )()
                ]
            },
        )()


class _FakeOpenAIClient:
    def __init__(self, content: str) -> None:
        self.chat = type(
            "FakeChat",
            (),
            {"completions": _FakeOpenAICompletions(content)},
        )()


def test_transcribe_with_openai_client_returns_result() -> None:
    runner = VllmChatRunner(
        model="test-model",
        client=_FakeOpenAIClient('{"text":"Hello OCR","blocks":[{"text":"Hello OCR"}]}'),
        temperature=0.1,
    )

    result = runner.transcribe(b"fake-png-bytes", page_number=1)

    assert result.text == "Hello OCR"
    assert result.backend == "vllm-chat"
    assert result.page_number == 1
    assert result.data["text"] == "Hello OCR"


def test_transcribe_with_httpx_client_returns_result() -> None:
    runner = VllmChatRunner(
        model="test-model",
        client=_PostClient('{"text":"httpx OCR","blocks":[{"text":"httpx OCR"}]}'),
        use_httpx=True,
    )

    result = runner.transcribe(b"fake-png-bytes")

    assert result.text == "httpx OCR"
    assert result.backend == "vllm-chat"


def test_build_payload_includes_top_p_and_response_format() -> None:
    runner = VllmChatRunner(
        model="test",
        client=_FakeOpenAIClient('{"text":"ok"}'),
        top_p=0.9,
        response_format={"type": "json_schema"},
        extra_body={"guided_json": True},
    )

    result = runner.transcribe(b"fake")

    assert result.text == "ok"


def test_response_format_can_be_disabled() -> None:
    captured: dict[str, Any] = {}

    class _CaptureCompletions:
        def create(self, **kwargs: Any) -> Any:
            captured.update(kwargs)
            return type(
                "FakeCompletion",
                (),
                {
                    "choices": [
                        type(
                            "FakeChoice",
                            (),
                            {"message": type("FakeMsg", (), {"content": "plain OCR"})()},
                        )()
                    ]
                },
            )()

    client = type(
        "FakeClient",
        (),
        {"chat": type("FakeChat", (), {"completions": _CaptureCompletions()})()},
    )()
    runner = VllmChatRunner(
        model="test",
        client=client,
        response_format=None,
        strict_json=False,
    )

    result = runner.transcribe(b"fake", schema=None, system_prompt=None)

    assert result.text == "plain OCR"
    assert "response_format" not in captured
    assert captured["messages"][0]["role"] == "user"


def test_transcribe_handles_invalid_json_with_strict_false() -> None:
    runner = VllmChatRunner(
        model="model",
        client=_FakeOpenAIClient("not valid json at all"),
    )

    with pytest.raises(ValueError):
        runner.transcribe(b"fake")

    runner_strict_false = VllmChatRunner(
        model="model",
        client=_FakeOpenAIClient("raw prose from model"),
        strict_json=False,
    )
    result = runner_strict_false.transcribe(b"fake")
    assert result.text == "raw prose from model"


def test_transcribe_raises_when_missing_text_field() -> None:
    runner = VllmChatRunner(
        model="model",
        client=_FakeOpenAIClient('{"not_text": 42}'),
    )

    with pytest.raises(ValueError, match="text"):
        runner.transcribe(b"fake")


def test_transcribe_rejects_non_array_blocks() -> None:
    runner = VllmChatRunner(
        model="model",
        client=_FakeOpenAIClient('{"text":"ok","blocks":"not-array"}'),
    )

    with pytest.raises(ValueError, match="blocks"):
        runner.transcribe(b"fake")


def test_transcribe_handles_list_content_from_model() -> None:
    class _ListContentClient:
        def __init__(self) -> None:
            class _FakeCompletions:
                def create(self, **kw: Any) -> Any:
                    class _FakeChoice:
                        message = type(
                            "FakeMsg",
                            (),
                            {"content": [{"text": "Hello "}, {"text": "World"}]},
                        )()

                    return type("FakeCompletion", (), {"choices": [_FakeChoice()]})()

            self.chat = type("FakeChat", (), {"completions": _FakeCompletions()})()

    runner = VllmChatRunner(model="model", client=_ListContentClient(), strict_json=False)

    result = runner.transcribe(b"fake")
    assert "Hello" in result.raw_text
    assert "World" in result.raw_text
