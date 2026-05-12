"""Smoke tests for the gwanbo-ocr CLI using Typer's test runner."""

from __future__ import annotations

import json
import sys
from pathlib import Path

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


def test_bench_score_help() -> None:
    result = runner.invoke(app, ["bench", "score", "--help"])
    assert result.exit_code == 0
    assert "--run" in result.output
    assert "--output" in result.output


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
