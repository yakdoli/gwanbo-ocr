"""Smoke tests for the gwanbo-ocr CLI using Typer's test runner."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from typer.testing import CliRunner

from gwanbo_ocr.cli import app

runner = CliRunner()


def test_help_exits_cleanly() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "manifest" in result.output
    assert "pdf" in result.output
    assert "bench" in result.output
    assert "convert" in result.output
    assert "strategy" in result.output


def test_manifest_build_help() -> None:
    result = runner.invoke(app, ["manifest", "build", "--help"])
    assert result.exit_code == 0
    assert "--peti-root" in result.output
    assert "--output" in result.output


def test_pdf_classify_help() -> None:
    result = runner.invoke(app, ["pdf", "classify", "--help"])
    assert result.exit_code == 0
    assert "--input" in result.output
    assert "--output" in result.output


def test_pdf_layout_help() -> None:
    result = runner.invoke(app, ["pdf", "layout", "--help"])
    assert result.exit_code == 0
    assert "--classification" in result.output


def test_convert_file_forwards_mode_and_llm_options(tmp_path: Path, monkeypatch: Any) -> None:
    captured: dict[str, Any] = {}

    def fake_convert_document(**kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return {"status": "ok", "output_path": str(tmp_path / "out" / "doc.md")}

    import gwanbo_ocr.conversion as conversion

    monkeypatch.setattr(conversion, "convert_document", fake_convert_document)
    input_path = tmp_path / "doc.pdf"
    input_path.write_bytes(b"%PDF")

    result = runner.invoke(
        app,
        [
            "convert",
            "file",
            str(input_path),
            "--output",
            str(tmp_path / "out"),
            "--mode",
            "ocr-llm",
            "--llm-base-url",
            "http://127.0.0.1:8000/v1",
            "--llm-model",
            "Qwen/Qwen3.6-35B-A3B-FP8",
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["input_path"] == input_path
    assert captured["output"] == tmp_path / "out"
    assert captured["mode"] == "ocr-llm"
    assert captured["llm_base_url"] == "http://127.0.0.1:8000/v1"
    assert captured["llm_model"] == "Qwen/Qwen3.6-35B-A3B-FP8"


def test_pdf_render_help() -> None:
    result = runner.invoke(app, ["pdf", "render", "--help"])
    assert result.exit_code == 0
    assert "--dpi" in result.output


def test_pdf_validate_help() -> None:
    result = runner.invoke(app, ["pdf", "validate", "--help"])
    assert result.exit_code == 0


def test_pdf_profile_help() -> None:
    result = runner.invoke(app, ["pdf", "profile", "--help"])
    assert result.exit_code == 0
    assert "--sample-per-bucket" in result.output


def test_bench_run_help() -> None:
    result = runner.invoke(app, ["bench", "run", "--help"])
    assert result.exit_code == 0
    assert "--runner" in result.output
    assert "--base-url" in result.output
    assert "--preflight-vllm" in result.output


def test_bench_score_help() -> None:
    result = runner.invoke(app, ["bench", "score", "--help"])
    assert result.exit_code == 0
    assert "--run" in result.output
    assert "--output" in result.output


def test_bench_run_forwards_preflight_options(tmp_path: Path, monkeypatch: Any) -> None:
    captured: dict[str, Any] = {}

    def fake_run_benchmark(**kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return {"status": "ok"}

    import gwanbo_ocr.bench as bench

    monkeypatch.setattr(bench, "run_benchmark", fake_run_benchmark)

    result = runner.invoke(
        app,
        [
            "bench",
            "run",
            "--suite",
            "smoke",
            "--run-dir",
            str(tmp_path / "run"),
            "--preflight-timeout-s",
            "0.25",
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["preflight_vllm"] is True
    assert captured["preflight_timeout_s"] == 0.25


def test_strategy_cluster_help() -> None:
    result = runner.invoke(app, ["strategy", "cluster", "--help"])
    assert result.exit_code == 0
    assert "--profiles" in result.output


def test_strategy_evaluate_help() -> None:
    result = runner.invoke(app, ["strategy", "evaluate", "--help"])
    assert result.exit_code == 0
    assert "--clusters" in result.output


def test_strategy_pipeline_help() -> None:
    result = runner.invoke(app, ["strategy", "pipeline", "--help"])
    assert result.exit_code == 0
    assert "--manifest" in result.output
    assert "--output" in result.output
    assert "--preflight-vllm" in result.output
    assert "--paddle-vl" in result.output


def test_strategy_pipeline_forwards_paddle_vl_options(tmp_path: Path, monkeypatch: Any) -> None:
    captured: dict[str, Any] = {}

    def fake_run_pipeline(**kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return {"strategy_eval": {}, "bench_run": {}}

    import gwanbo_ocr.strategy as strategy

    monkeypatch.setattr(strategy, "run_pipeline", fake_run_pipeline)

    result = runner.invoke(
        app,
        [
            "strategy",
            "pipeline",
            "--manifest",
            str(tmp_path / "manifest.jsonl"),
            "--output",
            str(tmp_path / "out"),
            "--paddle-vl",
            "--paddle-vl-server-url",
            "http://127.0.0.1:8000/v1",
            "--paddle-vl-model",
            "PaddlePaddle/PaddleOCR-VL-1.5",
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["run_paddle_vl"] is True
    assert captured["paddle_vl_backend"] is None
    assert captured["paddle_vl_server_url"] == "http://127.0.0.1:8000/v1"
    assert captured["paddle_vl_model"] == "PaddlePaddle/PaddleOCR-VL-1.5"


def test_strategy_pipeline_forwards_service_peer_options(tmp_path: Path, monkeypatch: Any) -> None:
    captured: dict[str, Any] = {}

    def fake_run_pipeline(**kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return {"strategy_eval": {}, "bench_run": {}}

    import gwanbo_ocr.strategy as strategy

    monkeypatch.setattr(strategy, "run_pipeline", fake_run_pipeline)

    result = runner.invoke(
        app,
        [
            "strategy",
            "pipeline",
            "--manifest",
            str(tmp_path / "manifest.jsonl"),
            "--output",
            str(tmp_path / "out"),
            "--markitdown-ocr-llm",
            "--markitdown-service-url",
            "http://127.0.0.1:8081",
            "--markitdown-llm-base-url",
            "http://127.0.0.1:8000/v1",
            "--markitdown-llm-model",
            "Qwen/Qwen3.6-35B-A3B-FP8",
            "--paddle-service-url",
            "http://127.0.0.1:8082",
            "--paddle-vl-service-url",
            "http://127.0.0.1:8082",
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["run_markitdown_ocr_llm"] is True
    assert captured["markitdown_service_url"] == "http://127.0.0.1:8081"
    assert captured["markitdown_llm_base_url"] == "http://127.0.0.1:8000/v1"
    assert captured["markitdown_llm_model"] == "Qwen/Qwen3.6-35B-A3B-FP8"
    assert captured["paddle_service_url"] == "http://127.0.0.1:8082"
    assert captured["paddle_vl_service_url"] == "http://127.0.0.1:8082"


def test_manifest_build_with_stub_peti_root(tmp_path: Path) -> None:
    """manifest build with empty metadata files produces an empty manifest."""
    peti = tmp_path / "peti"
    for theme in ("searchThema", "pety"):
        meta_dir = peti / "artifacts" / theme / "metadata"
        meta_dir.mkdir(parents=True)
        (meta_dir / "metadata.json").write_text("{}", encoding="utf-8")
    output = tmp_path / "manifest.jsonl"

    result = runner.invoke(
        app,
        [
            "manifest",
            "build",
            "--peti-root",
            str(peti),
            "--output",
            str(output),
        ],
    )
    assert result.exit_code == 0, result.output
    assert output.exists()
    rows = output.read_text(encoding="utf-8").strip()
    assert rows == "" or all(json.loads(line) for line in rows.splitlines())


def test_manifest_build_refuses_output_inside_peti(tmp_path: Path) -> None:
    """manifest build rejects output paths that are under the peti root."""
    peti = tmp_path / "peti"
    peti.mkdir()
    bad_output = peti / "manifest.jsonl"

    result = runner.invoke(
        app,
        [
            "manifest",
            "build",
            "--peti-root",
            str(peti),
            "--output",
            str(bad_output),
        ],
    )
    assert result.exit_code != 0


def test_manifest_build_with_real_peti_produces_jsonl(tmp_path: Path) -> None:
    """Smoke-test against actual /root/peti with a small limit."""
    peti = Path("/root/peti")
    if not peti.exists():
        import pytest

        pytest.skip("/root/peti not present in this environment")

    output = tmp_path / "manifest.jsonl"
    result = runner.invoke(
        app,
        [
            "manifest",
            "build",
            "--peti-root",
            str(peti),
            "--output",
            str(output),
            "--limit",
            "5",
        ],
    )
    assert result.exit_code == 0, result.output
    assert output.exists()
    rows = [
        json.loads(line) for line in output.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    assert len(rows) <= 5
    for row in rows:
        assert "id" in row
        assert "pdf_path" in row
        assert "theme" in row
