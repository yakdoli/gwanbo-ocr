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
