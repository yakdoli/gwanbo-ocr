from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from gwanbo_ocr.bench import RunRecord, format_throughput_report, summarize_throughput


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
