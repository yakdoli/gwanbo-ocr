from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any
from urllib import error as url_error

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from gwanbo_ocr.bench import (
    RunRecord,
    format_throughput_report,
    resolve_runner_model,
    run_benchmark,
    score_benchmark,
    summarize_throughput,
)


def test_summarize_throughput_uses_wall_elapsed_time() -> None:
    summary = summarize_throughput(
        [
            RunRecord(
                item_id="a",
                status="ok",
                started_at="2026-05-12T00:00:00Z",
                ended_at="2026-05-12T00:00:10Z",
                pages=10,
                bytes_processed=1_000_000,
                engine="ocr-a",
            ),
            RunRecord(
                item_id="b",
                status="failed",
                started_at="2026-05-12T00:00:00Z",
                ended_at="2026-05-12T00:00:05Z",
                pages=20,
                bytes_processed=2_000_000,
                engine="ocr-a",
            ),
        ]
    )

    assert summary["documents"] == 2
    assert summary["succeeded"] == 1
    assert summary["failed"] == 1
    assert summary["pages"] == 30
    assert summary["elapsed_s"] == 10
    assert summary["pages_per_s"] == 3.0
    assert summary["by_engine"] == {"ocr-a": 2}


def test_format_throughput_report_is_markdown() -> None:
    report = format_throughput_report(
        {
            "documents": 1,
            "succeeded": 1,
            "failed": 0,
            "pages": 2,
            "elapsed_s": 1.0,
            "documents_per_s": 1.0,
            "pages_per_s": 2.0,
            "mb_per_s": 0.5,
            "latency_s": {"p50": 1.0, "p95": 1.0},
        }
    )

    assert report.startswith("# OCR Throughput")
    assert "- pages_per_s: 2.000" in report


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

    import gwanbo_ocr.runners.paddle as paddle
    import gwanbo_ocr.runners.vllm_chat as vllm_chat

    monkeypatch.setattr(paddle, "PaddleOcrRunner", FailingPaddle)
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


def test_score_benchmark_includes_route_summary(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True)
    (run_dir / "results.jsonl").write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "item_id": "a",
                        "status": "ok",
                        "route": "vlm_primary",
                        "strategy": "ocr_vlm_structured",
                        "started_at": "2026-05-12T00:00:00Z",
                        "ended_at": "2026-05-12T00:00:01Z",
                    }
                ),
                json.dumps(
                    {
                        "item_id": "b",
                        "status": "skipped",
                        "route": "native_strategy_skip",
                        "strategy": "native_text_body",
                        "started_at": "2026-05-12T00:00:00Z",
                        "ended_at": "2026-05-12T00:00:00Z",
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    summary = score_benchmark(run_dir=run_dir, output_dir=tmp_path / "report")
    assert summary["by_route"]["vlm_primary"] == 1
    assert summary["by_route"]["native_strategy_skip"] == 1
    assert summary["by_strategy"]["ocr_vlm_structured"] == 1
