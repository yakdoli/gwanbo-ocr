from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from gwanbo_ocr.bench import score_benchmark


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


def test_score_benchmark_computes_metrics_when_reference_text_present(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True)
    (run_dir / "results.jsonl").write_text(
        json.dumps(
            {
                "item_id": "a",
                "status": "ok",
                "route": "vlm_primary",
                "strategy": "ocr_vlm_structured",
                "started_at": "2026-05-12T00:00:00Z",
                "ended_at": "2026-05-12T00:00:01Z",
                "text": "hello world",
                "reference_text": "hello world",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    score_benchmark(run_dir=run_dir, output_dir=tmp_path / "report")
    scores_lines = (tmp_path / "report" / "scores.jsonl").read_text(encoding="utf-8").splitlines()
    scored = json.loads(scores_lines[0])
    assert isinstance(scored["metrics"], dict)
    assert scored["metrics"] != {}
    assert scored["metrics"]["cer"] == 0.0
    assert scored["metrics"]["wer"] == 0.0
