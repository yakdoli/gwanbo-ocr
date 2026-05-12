from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from gwanbo_ocr.bench import (
    RunRecord,
    format_throughput_report,
    resolve_runner_model,
    run_benchmark,
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
