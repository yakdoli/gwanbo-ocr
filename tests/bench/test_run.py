from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any
from urllib import error as url_error

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from gwanbo_ocr.bench import resolve_runner_model, run_benchmark
from gwanbo_ocr.bench.run import _merge_gold_references, _run_benchmark_task, resolve_runner_config


def test_resolve_runner_model_uses_model_config_alias(tmp_path: Path) -> None:
    config = tmp_path / "models.yaml"
    config.write_text(
        """
vision_language_models:
  qwen36_baseline:
    model: Qwen/Qwen3.6-35B-A3B-FP8
""",
        encoding="utf-8",
    )

    assert resolve_runner_model("qwen36_baseline", config_path=config) == "Qwen/Qwen3.6-35B-A3B-FP8"
    assert resolve_runner_model("direct-model", config_path=config) == "direct-model"


def test_run_benchmark_preflight_validates_served_model(tmp_path: Path, monkeypatch: Any) -> None:
    image = tmp_path / "page.png"
    image.write_bytes(b"png")
    suite = tmp_path / "suite.jsonl"
    suite.write_text(
        json.dumps({"sample_id": "s1", "image_path": str(image), "page_number": 1}) + "\n",
        encoding="utf-8",
    )
    calls: list[tuple[str, float]] = []

    class FakeResponse:
        status = 200

        def __enter__(self) -> FakeResponse:
            return self

        def __exit__(self, *_args: Any) -> None:
            return None

        def read(self) -> bytes:
            return json.dumps({"data": [{"id": "direct-model"}]}).encode("utf-8")

    def fake_urlopen(req: Any, timeout: float) -> FakeResponse:
        calls.append((req.full_url, timeout))
        return FakeResponse()

    class FakeResult:
        text = "ok"
        data = {"text": "ok"}

        def to_dict(self) -> dict[str, str]:
            return {"text": self.text}

    class FakeRunner:
        def __init__(self, **_kwargs: Any) -> None:
            pass

        def transcribe(self, *_args: Any, **_kwargs: Any) -> FakeResult:
            return FakeResult()

    import gwanbo_ocr.runners.preflight as preflight_mod
    import gwanbo_ocr.runners.vllm_chat as vllm_chat

    monkeypatch.setattr(preflight_mod.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(vllm_chat, "VllmChatRunner", FakeRunner)

    summary = run_benchmark(
        suite=str(suite),
        runner_name="direct-model",
        run_dir=tmp_path / "run",
        concurrency=1,
        preflight_vllm=True,
        preflight_timeout_s=0.25,
    )

    assert calls == [("http://127.0.0.1:8000/v1/models", 0.25)]
    assert summary["preflight"]["status"] == "ok"
    assert summary["preflight"]["model_id"] == "direct-model"
    assert summary["preflight"]["served_models"] == ["direct-model"]


def test_run_benchmark_preflight_rejects_unserved_model(tmp_path: Path, monkeypatch: Any) -> None:
    image = tmp_path / "page.png"
    image.write_bytes(b"png")
    suite = tmp_path / "suite.jsonl"
    suite.write_text(
        json.dumps({"sample_id": "s1", "image_path": str(image), "page_number": 1}) + "\n",
        encoding="utf-8",
    )

    class FakeResponse:
        status = 200

        def __enter__(self) -> FakeResponse:
            return self

        def __exit__(self, *_args: Any) -> None:
            return None

        def read(self) -> bytes:
            return json.dumps({"data": [{"id": "other-model"}]}).encode("utf-8")

    import gwanbo_ocr.runners.preflight as preflight_mod

    monkeypatch.setattr(preflight_mod.request, "urlopen", lambda *_args, **_kwargs: FakeResponse())

    with pytest.raises(RuntimeError, match="not served"):
        run_benchmark(
            suite=str(suite),
            runner_name="direct-model",
            run_dir=tmp_path / "run",
            concurrency=1,
            preflight_vllm=True,
        )


def test_run_benchmark_preflight_rejects_http_errors(tmp_path: Path, monkeypatch: Any) -> None:
    image = tmp_path / "page.png"
    image.write_bytes(b"png")
    suite = tmp_path / "suite.jsonl"
    suite.write_text(
        json.dumps({"sample_id": "s1", "image_path": str(image), "page_number": 1}) + "\n",
        encoding="utf-8",
    )

    def fake_urlopen(req: Any, timeout: float) -> Any:
        raise url_error.HTTPError(req.full_url, 404, "Not Found", None, None)  # type: ignore[arg-type]

    import gwanbo_ocr.runners.preflight as preflight_mod

    monkeypatch.setattr(preflight_mod.request, "urlopen", fake_urlopen)

    with pytest.raises(RuntimeError, match="404"):
        run_benchmark(
            suite=str(suite),
            runner_name="direct-model",
            run_dir=tmp_path / "run",
            concurrency=1,
            preflight_vllm=True,
        )


def test_run_benchmark_honors_concurrency_option(tmp_path: Path, monkeypatch: Any) -> None:
    image = tmp_path / "page.png"
    image.write_bytes(b"png")
    suite = tmp_path / "suite.jsonl"
    suite.write_text(
        "\n".join(
            json.dumps({"sample_id": f"s{i}", "image_path": str(image), "page_number": 1})
            for i in range(3)
        )
        + "\n",
        encoding="utf-8",
    )

    class FakeResult:
        text = "ok"
        data = {"text": "ok"}

        def to_dict(self) -> dict[str, str]:
            return {"text": self.text}

    class FakeRunner:
        def __init__(self, **_kwargs: Any) -> None:
            pass

        def transcribe(self, *_args: Any, **_kwargs: Any) -> FakeResult:
            return FakeResult()

    import gwanbo_ocr.runners.vllm_chat as vllm_chat

    monkeypatch.setattr(vllm_chat, "VllmChatRunner", FakeRunner)

    summary = run_benchmark(
        suite=str(suite),
        runner_name="direct-model",
        run_dir=tmp_path / "run",
        concurrency=2,
    )

    records = (tmp_path / "run" / "results.jsonl").read_text(encoding="utf-8").splitlines()
    assert summary["concurrency"] == 2
    assert summary["tasks"] == 3
    assert len(records) == 3


def test_run_benchmark_preserves_strategy_metadata(tmp_path: Path, monkeypatch: Any) -> None:
    image = tmp_path / "page.png"
    image.write_bytes(b"png")
    suite = tmp_path / "suite.jsonl"
    suite.write_text(
        json.dumps(
            {
                "sample_id": "s1",
                "image_path": str(image),
                "page_number": 1,
                "strategy": "ocr_vlm_structured",
                "cluster_id": "cluster-1",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    class FakeResult:
        text = "ok"
        data = {"text": "ok"}

        def to_dict(self) -> dict[str, str]:
            return {"text": self.text}

    class FakeRunner:
        def __init__(self, **_kwargs: Any) -> None:
            pass

        def transcribe(self, *_args: Any, **_kwargs: Any) -> FakeResult:
            return FakeResult()

    import gwanbo_ocr.runners.vllm_chat as vllm_chat

    monkeypatch.setattr(vllm_chat, "VllmChatRunner", FakeRunner)
    run_benchmark(
        suite=str(suite),
        runner_name="direct-model",
        run_dir=tmp_path / "run",
        concurrency=1,
    )
    record = json.loads(
        (tmp_path / "run" / "results.jsonl").read_text(encoding="utf-8").splitlines()[0]
    )
    assert record["strategy"] == "ocr_vlm_structured"
    assert record["cluster_id"] == "cluster-1"


def test_run_benchmark_skips_native_strategy_when_enforced(
    tmp_path: Path, monkeypatch: Any
) -> None:
    image = tmp_path / "page.png"
    image.write_bytes(b"png")
    suite = tmp_path / "suite.jsonl"
    suite.write_text(
        json.dumps(
            {
                "sample_id": "s1",
                "image_path": str(image),
                "page_number": 1,
                "strategy": "native_text_body",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    class ShouldNotRunVllm:
        def __init__(self, **_kwargs: Any) -> None:
            pass

        def transcribe(self, *_args: Any, **_kwargs: Any) -> Any:
            raise AssertionError("vllm should not run for native strategy")

    import gwanbo_ocr.runners.vllm_chat as vllm_chat

    monkeypatch.setattr(vllm_chat, "VllmChatRunner", ShouldNotRunVllm)
    summary = run_benchmark(
        suite=str(suite),
        runner_name="direct-model",
        run_dir=tmp_path / "run",
        concurrency=1,
        enforce_strategy_routing=True,
    )
    record = json.loads(
        (tmp_path / "run" / "results.jsonl").read_text(encoding="utf-8").splitlines()[0]
    )
    assert summary["tasks"] == 1
    assert record["status"] == "skipped"
    assert record["route"] == "native_strategy_skip"
    assert summary["by_route"]["native_strategy_skip"] == 1


def test_run_benchmark_preflight_skips_native_only_suite(tmp_path: Path, monkeypatch: Any) -> None:
    image = tmp_path / "page.png"
    image.write_bytes(b"png")
    suite = tmp_path / "suite.jsonl"
    suite.write_text(
        json.dumps(
            {
                "sample_id": "s1",
                "image_path": str(image),
                "page_number": 1,
                "strategy": "native_pdfplumber_table",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    import gwanbo_ocr.runners.preflight as preflight_mod

    monkeypatch.setattr(
        preflight_mod.request,
        "urlopen",
        lambda *_args, **_kwargs: pytest.fail("preflight should not call VLM for native routes"),
    )

    summary = run_benchmark(
        suite=str(suite),
        runner_name="direct-model",
        run_dir=tmp_path / "run",
        concurrency=1,
        enforce_strategy_routing=True,
        preflight_vllm=True,
    )

    assert summary["preflight"]["status"] == "skipped_no_vllm_route"
    assert summary["by_route"]["native_strategy_skip"] == 1


def test_run_benchmark_falls_back_to_vllm_when_paddle_fails(
    tmp_path: Path, monkeypatch: Any
) -> None:
    image = tmp_path / "page.png"
    image.write_bytes(b"png")
    suite = tmp_path / "suite.jsonl"
    suite.write_text(
        json.dumps(
            {
                "sample_id": "s1",
                "image_path": str(image),
                "page_number": 1,
                "strategy": "ocr_paddle_simple",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    class FailingPaddle:
        def __init__(self, **_kwargs: Any) -> None:
            pass

        def transcribe(self, *_args: Any, **_kwargs: Any) -> Any:
            raise RuntimeError("paddle unavailable")

    class FakeResult:
        text = "ok"
        data = {"text": "ok"}

        def to_dict(self) -> dict[str, str]:
            return {"text": self.text}

    class FakeVllm:
        def __init__(self, **_kwargs: Any) -> None:
            pass

        def transcribe(self, *_args: Any, **_kwargs: Any) -> FakeResult:
            return FakeResult()

    import gwanbo_ocr.runners.paddle_service as paddle_service
    import gwanbo_ocr.runners.vllm_chat as vllm_chat

    monkeypatch.setattr(paddle_service, "PaddleOcrServiceRunner", FailingPaddle)
    monkeypatch.setattr(vllm_chat, "VllmChatRunner", FakeVllm)

    run_benchmark(
        suite=str(suite),
        runner_name="direct-model",
        run_dir=tmp_path / "run",
        concurrency=1,
        enforce_strategy_routing=True,
    )
    record = json.loads(
        (tmp_path / "run" / "results.jsonl").read_text(encoding="utf-8").splitlines()[0]
    )
    assert record["status"] == "ok"
    assert record["route"] == "paddle_to_vllm_fallback"


def test_run_benchmark_uses_paddle_service_for_paddle_strategy(
    tmp_path: Path, monkeypatch: Any
) -> None:
    image = tmp_path / "page.png"
    image.write_bytes(b"png")
    suite = tmp_path / "suite.jsonl"
    suite.write_text(
        json.dumps(
            {
                "sample_id": "s1",
                "image_path": str(image),
                "page_number": 1,
                "strategy": "ocr_paddle_simple",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    captured: dict[str, Any] = {}

    class FakeResult:
        text = "paddle ok"
        data = {"text": "paddle ok"}

        def to_dict(self) -> dict[str, str]:
            return {"text": self.text}

    class FakeServicePaddle:
        def __init__(self, base_url: str, **kwargs: Any) -> None:
            captured["base_url"] = base_url
            captured["kwargs"] = kwargs

        def transcribe(self, image_path: Path, *, page_number: int | None = None) -> FakeResult:
            captured["image_path"] = image_path
            captured["page_number"] = page_number
            return FakeResult()

    class ShouldNotUseLocalPaddle:
        def __init__(self, **_kwargs: Any) -> None:
            raise AssertionError("host PaddleOCR runner should not be constructed")

    import gwanbo_ocr.runners.paddle as paddle
    import gwanbo_ocr.runners.paddle_service as paddle_service

    monkeypatch.setenv("GWANBO_PADDLEOCR_SERVICE_URL", "http://paddle-service:8080")
    monkeypatch.setattr(paddle_service, "PaddleOcrServiceRunner", FakeServicePaddle)
    monkeypatch.setattr(paddle, "PaddleOcrRunner", ShouldNotUseLocalPaddle)

    summary = run_benchmark(
        suite=str(suite),
        runner_name="direct-model",
        run_dir=tmp_path / "run",
        concurrency=1,
        enforce_strategy_routing=True,
    )

    record = json.loads(
        (tmp_path / "run" / "results.jsonl").read_text(encoding="utf-8").splitlines()[0]
    )
    assert summary["paddle_service_url"] == "http://paddle-service:8080"
    assert captured["base_url"] == "http://paddle-service:8080"
    assert captured["image_path"] == image
    assert captured["page_number"] == 1
    assert record["status"] == "ok"
    assert record["route"] == "paddle_primary"


def test_reference_text_propagates_to_record(tmp_path: Path, monkeypatch: Any) -> None:
    image = tmp_path / "page.png"
    image.write_bytes(b"png")

    class FakeResult:
        text = "ocr output"
        raw_response = {
            "id": "chatcmpl-test",
            "model": "ONTHEIT/BizOnAI-OCR",
            "object": "chat.completion",
            "choices": [{"finish_reason": "length", "stop_reason": None}],
            "usage": {
                "prompt_tokens": 11,
                "completion_tokens": 22,
                "total_tokens": 33,
            },
        }

        def to_dict(self) -> dict[str, str]:
            return {"text": self.text}

    class FakeRunner:
        def __init__(self, **_kwargs: Any) -> None:
            pass

        def transcribe(self, *_args: Any, **_kwargs: Any) -> FakeResult:
            return FakeResult()

    import gwanbo_ocr.runners.vllm_chat as vllm_chat

    monkeypatch.setattr(vllm_chat, "VllmChatRunner", FakeRunner)

    task = {
        "sample_id": "s1",
        "image_path": str(image),
        "page_number": 1,
        "reference_text": "expected text",
    }
    record = _run_benchmark_task(
        task,
        runner_name="direct-model",
        model_id="direct-model",
        base_url="http://127.0.0.1:8000/v1",
        api_key="dummy",
        enforce_strategy_routing=False,
        paddle_service_url=None,
    )
    assert record["reference_text"] == "expected text"


def test_gold_text_propagates_as_reference_text(tmp_path: Path, monkeypatch: Any) -> None:
    image = tmp_path / "page.png"
    image.write_bytes(b"png")

    class FakeResult:
        text = "ocr output"
        raw_response = {
            "id": "chatcmpl-test",
            "model": "ONTHEIT/BizOnAI-OCR",
            "object": "chat.completion",
            "choices": [{"finish_reason": "length", "stop_reason": None}],
            "usage": {
                "prompt_tokens": 11,
                "completion_tokens": 22,
                "total_tokens": 33,
            },
        }

        def to_dict(self) -> dict[str, str]:
            return {"text": self.text}

    class FakeRunner:
        def __init__(self, **_kwargs: Any) -> None:
            pass

        def transcribe(self, *_args: Any, **_kwargs: Any) -> FakeResult:
            return FakeResult()

    import gwanbo_ocr.runners.vllm_chat as vllm_chat

    monkeypatch.setattr(vllm_chat, "VllmChatRunner", FakeRunner)

    task = {
        "sample_id": "s1",
        "image_path": str(image),
        "page_number": 1,
        "gold_text": "gold expected",
    }
    record = _run_benchmark_task(
        task,
        runner_name="direct-model",
        model_id="direct-model",
        base_url="http://127.0.0.1:8000/v1",
        api_key="dummy",
        enforce_strategy_routing=False,
        paddle_service_url=None,
    )
    assert record["reference_text"] == "gold expected"


def test_vlm_max_tokens_is_passed_to_runner(tmp_path: Path, monkeypatch: Any) -> None:
    image = tmp_path / "page.png"
    image.write_bytes(b"png")
    captured: dict[str, Any] = {}

    class FakeResult:
        text = "ocr output"
        raw_response = {
            "id": "chatcmpl-test",
            "model": "ONTHEIT/BizOnAI-OCR",
            "object": "chat.completion",
            "choices": [{"finish_reason": "length", "stop_reason": None}],
            "usage": {
                "prompt_tokens": 11,
                "completion_tokens": 22,
                "total_tokens": 33,
            },
        }

        def to_dict(self) -> dict[str, str]:
            return {"text": self.text}

    class FakeRunner:
        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)

        def transcribe(self, *_args: Any, **_kwargs: Any) -> FakeResult:
            return FakeResult()

    import gwanbo_ocr.runners.vllm_chat as vllm_chat

    monkeypatch.setattr(vllm_chat, "VllmChatRunner", FakeRunner)

    record = _run_benchmark_task(
        {"sample_id": "s1", "image_path": str(image), "page_number": 1},
        runner_name="bizonai_ocr",
        model_id="ONTHEIT/BizOnAI-OCR",
        base_url="http://127.0.0.1:8000/v1",
        api_key="dummy",
        enforce_strategy_routing=False,
        paddle_service_url=None,
        runner_config={
            "timeout_seconds": 180,
            "max_retries": 2,
            "max_tokens": 1024,
            "temperature": 0.2,
            "top_p": 0.9,
        },
    )

    assert record["status"] == "ok"
    assert captured["max_tokens"] == 1024
    assert captured["temperature"] == 0.2
    assert captured["top_p"] == 0.9
    assert record["finish_reason"] == "length"
    assert record["prompt_tokens"] == 11
    assert record["completion_tokens"] == 22
    assert record["total_tokens"] == 33
    assert record["response_metadata"]["id"] == "chatcmpl-test"


def test_paddle_preprocess_context_is_added_to_vlm_prompt(tmp_path: Path, monkeypatch: Any) -> None:
    image = tmp_path / "page.png"
    image.write_bytes(b"png")
    captured: dict[str, Any] = {}

    class FakePaddleResult:
        text = "Paddle draft text"
        backend = "paddleocr_service"

        def to_dict(self) -> dict[str, str]:
            return {"text": self.text, "backend": self.backend}

    class FakePaddle:
        def __init__(self, base_url: str, **_kwargs: Any) -> None:
            captured["paddle_base_url"] = base_url

        def transcribe(
            self, image_path: Path, *, page_number: int | None = None
        ) -> FakePaddleResult:
            captured["paddle_image_path"] = image_path
            captured["paddle_page_number"] = page_number
            return FakePaddleResult()

    class FakeVlmResult:
        text = "vlm output"

        def to_dict(self) -> dict[str, str]:
            return {"text": self.text}

    class FakeVlm:
        def __init__(self, **_kwargs: Any) -> None:
            pass

        def transcribe(self, *_args: Any, **kwargs: Any) -> FakeVlmResult:
            captured["user_prompt"] = kwargs["user_prompt"]
            return FakeVlmResult()

    import gwanbo_ocr.runners.paddle_service as paddle_service
    import gwanbo_ocr.runners.vllm_chat as vllm_chat

    monkeypatch.setattr(paddle_service, "PaddleOcrServiceRunner", FakePaddle)
    monkeypatch.setattr(vllm_chat, "VllmChatRunner", FakeVlm)

    record = _run_benchmark_task(
        {"sample_id": "s1", "image_path": str(image), "page_number": 2},
        runner_name="direct-model",
        model_id="direct-model",
        base_url="http://127.0.0.1:8000/v1",
        api_key="dummy",
        enforce_strategy_routing=False,
        paddle_service_url="http://paddle-service:8080",
        paddle_preprocess=True,
        paddle_preprocess_max_chars=4000,
    )

    assert record["status"] == "ok"
    assert record["paddle_preprocess"]["status"] == "ok"
    assert captured["paddle_base_url"] == "http://paddle-service:8080"
    assert "PaddleOCR preliminary transcription follows" in captured["user_prompt"]
    assert "Paddle draft text" in captured["user_prompt"]


def test_paddle_preprocess_fails_open_to_vlm(tmp_path: Path, monkeypatch: Any) -> None:
    image = tmp_path / "page.png"
    image.write_bytes(b"png")
    captured: dict[str, Any] = {}

    class FailingPaddle:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            pass

        def transcribe(self, *_args: Any, **_kwargs: Any) -> Any:
            raise RuntimeError("paddle down")

    class FakeVlmResult:
        text = "vlm output"

        def to_dict(self) -> dict[str, str]:
            return {"text": self.text}

    class FakeVlm:
        def __init__(self, **_kwargs: Any) -> None:
            pass

        def transcribe(self, *_args: Any, **kwargs: Any) -> FakeVlmResult:
            captured["user_prompt"] = kwargs.get("user_prompt")
            return FakeVlmResult()

    import gwanbo_ocr.runners.paddle_service as paddle_service
    import gwanbo_ocr.runners.vllm_chat as vllm_chat

    monkeypatch.setattr(paddle_service, "PaddleOcrServiceRunner", FailingPaddle)
    monkeypatch.setattr(vllm_chat, "VllmChatRunner", FakeVlm)

    record = _run_benchmark_task(
        {"sample_id": "s1", "image_path": str(image), "page_number": 1},
        runner_name="direct-model",
        model_id="direct-model",
        base_url="http://127.0.0.1:8000/v1",
        api_key="dummy",
        enforce_strategy_routing=False,
        paddle_service_url="http://paddle-service:8080",
        paddle_preprocess=True,
    )

    assert record["status"] == "ok"
    assert record["paddle_preprocess"]["status"] == "error"
    assert "paddle down" in record["paddle_preprocess"]["error"]
    assert captured["user_prompt"] is None


def test_merge_gold_references_fills_missing(tmp_path: Path) -> None:
    gold = tmp_path / "gold.jsonl"
    gold.write_text(
        json.dumps({"sample_id": "s1", "reference_text": "ref one"})
        + "\n"
        + json.dumps({"sample_id": "s2", "gold_text": "ref two"})
        + "\n"
        + json.dumps({"sample_id": "s4", "reference_text": "gold overwrite attempt"})
        + "\n",
        encoding="utf-8",
    )
    tasks: list[dict[str, Any]] = [
        {"sample_id": "s1", "image_path": "a.png"},
        {"sample_id": "s2", "image_path": "b.png"},
        {"sample_id": "s3", "image_path": "c.png"},
        {"sample_id": "s4", "image_path": "d.png", "reference_text": "already set"},
    ]
    result = _merge_gold_references(tasks, gold)
    assert result[0]["reference_text"] == "ref one"
    assert result[1]["reference_text"] == "ref two"
    assert result[2].get("reference_text") is None
    assert result[3]["reference_text"] == "already set"  # pre-set value wins over gold


def test_resolve_runner_config_returns_baseline_values(tmp_path: Path) -> None:
    config = tmp_path / "models.yaml"
    config.write_text(
        """
vision_language_models:
  qwen36_baseline:
    model: Qwen/Qwen3.6-35B-A3B-FP8
    timeout_seconds: 120
    max_retries: 2
    max_tokens: 1024
    temperature: 0.2
    top_p: 0.9
""",
        encoding="utf-8",
    )
    result = resolve_runner_config("qwen36_baseline", config_path=config)
    assert result["timeout_seconds"] == 120
    assert result["max_retries"] == 2
    assert result["max_tokens"] == 1024
    assert result["temperature"] == 0.2
    assert result["top_p"] == 0.9


def test_resolve_runner_config_returns_defaults_for_unknown(tmp_path: Path) -> None:
    config = tmp_path / "models.yaml"
    config.write_text(
        """
vision_language_models:
  qwen36_baseline:
    model: Qwen/Qwen3.6-35B-A3B-FP8
    timeout_seconds: 120
    max_retries: 2
""",
        encoding="utf-8",
    )
    result = resolve_runner_config("nonexistent_runner", config_path=config)
    assert result["timeout_seconds"] == 120
    assert result["max_retries"] == 2
    assert result["max_tokens"] == 4096


def test_resolve_runner_config_returns_defaults_when_no_file(tmp_path: Path) -> None:
    result = resolve_runner_config("any_runner", config_path=tmp_path / "missing.yaml")
    assert result["timeout_seconds"] == 120
    assert result["max_retries"] == 2
    assert result["max_tokens"] == 4096
