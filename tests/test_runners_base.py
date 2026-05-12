from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gwanbo_ocr.runners.base import TranscriptionResult, parse_json_response


class TestTranscriptionResult:
    def test_from_payload_extracts_text_field(self) -> None:
        result = TranscriptionResult.from_payload({"text": "hello", "blocks": []})
        assert result.text == "hello"

    def test_from_payload_extracts_transcription_field(self) -> None:
        result = TranscriptionResult.from_payload({"transcription": "world"})
        assert result.text == "world"

    def test_from_payload_extracts_text_from_blocks(self) -> None:
        result = TranscriptionResult.from_payload(
            {"blocks": [{"text": "line1"}, {"text": "line2"}]}
        )
        assert result.text == "line1\nline2"

    def test_from_payload_empty_payload_gives_empty_text(self) -> None:
        result = TranscriptionResult.from_payload({})
        assert result.text == ""

    def test_from_payload_preserves_page_number_and_backend(self) -> None:
        result = TranscriptionResult.from_payload({"text": "ok"}, page_number=3, backend="vllm")
        assert result.page_number == 3
        assert result.backend == "vllm"

    def test_to_dict_includes_text_and_data(self) -> None:
        result = TranscriptionResult.from_payload(
            {"text": "abc", "tables": []}, page_number=1, backend="paddle"
        )
        d = result.to_dict()
        assert d["text"] == "abc"
        assert d["page_number"] == 1
        assert d["backend"] == "paddle"

    def test_mapping_protocol_getitem(self) -> None:
        result = TranscriptionResult.from_payload({"text": "hi"}, backend="test")
        assert result["text"] == "hi"
        assert result["backend"] == "test"

    def test_mapping_protocol_iter_and_len(self) -> None:
        result = TranscriptionResult.from_payload({"text": "x"})
        keys = list(result)
        assert "text" in keys
        assert len(result) == len(keys)

    def test_mapping_protocol_missing_key_raises(self) -> None:
        result = TranscriptionResult.from_payload({"text": "x"})
        with pytest.raises(KeyError):
            _ = result["no_such_key"]


class TestParseJsonResponse:
    def test_parses_plain_json(self) -> None:
        result = parse_json_response('{"text": "hello", "blocks": []}')
        assert result["text"] == "hello"

    def test_strips_code_fences(self) -> None:
        fenced = '```json\n{"text": "fenced"}\n```'
        result = parse_json_response(fenced)
        assert result["text"] == "fenced"

    def test_extracts_embedded_json_with_leading_prose(self) -> None:
        response = 'Sure, here is the result: {"text": "embedded"}'
        result = parse_json_response(response)
        assert result["text"] == "embedded"

    def test_raises_on_non_json_in_strict_mode(self) -> None:
        with pytest.raises(ValueError, match="JSON"):
            parse_json_response("this is not json at all", strict=True)

    def test_non_strict_returns_text_fallback(self) -> None:
        result = parse_json_response("just plain text", strict=False)
        assert result["text"] == "just plain text"
        assert "non_json_response" in result.get("warnings", [])

    def test_normalizes_list_response(self) -> None:
        result = parse_json_response('["a", "b"]')
        assert "items" in result
        assert result["text"] == "a\nb"

    def test_normalizes_scalar_response(self) -> None:
        result = parse_json_response("42")
        assert result["value"] == 42
        assert result["text"] == "42"
