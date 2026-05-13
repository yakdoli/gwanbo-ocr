from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .report import (
    _count_by,
    format_throughput_report,
    load_records_jsonl,
    summarize_throughput,
    write_records_jsonl,
)


def score_benchmark(*, run_dir: str | Path, output_dir: str | Path) -> dict[str, Any]:
    """Score benchmark outputs and write aggregate JSON/Markdown reports."""
    from gwanbo_ocr.metrics import evaluate_document

    run = Path(run_dir)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    result_path = run / "results.jsonl"
    records = load_records_jsonl(result_path) if result_path.exists() else []
    scored: list[dict[str, Any]] = []
    for record in records:
        reference = record.get("reference_text") or record.get("gold_text") or ""
        hypothesis = record.get("text") or ""
        metrics = evaluate_document(reference, hypothesis) if reference else {}
        scored.append({**record, "metrics": metrics})
    scores_path = write_records_jsonl(scored, output / "scores.jsonl")
    throughput = summarize_throughput(records)
    report = format_throughput_report(throughput, title="gwanbo-ocr Benchmark")
    (output / "summary.md").write_text(report, encoding="utf-8")
    summary = {
        "status": "ok",
        "run_dir": str(run),
        "output_dir": str(output),
        "records": len(records),
        "scores": str(scores_path),
        "throughput": throughput,
        "by_strategy": _count_by(records, "strategy"),
        "by_route": _count_by(records, "route"),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    return summary
