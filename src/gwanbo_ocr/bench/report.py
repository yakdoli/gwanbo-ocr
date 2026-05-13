from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SUCCESS_STATUSES = {"ok", "success", "completed"}


def summarize_throughput(records: Iterable[Any]) -> dict[str, Any]:
    rows = [_as_mapping(record) for record in records]
    durations: list[float] = []
    starts: list[datetime] = []
    ends: list[datetime] = []
    for row in rows:
        duration = _duration(row)
        if duration is not None:
            durations.append(duration)
        start = _parse_time(row.get("started_at"))
        if start is not None:
            starts.append(start)
        end = _parse_time(row.get("ended_at"))
        if end is not None:
            ends.append(end)

    elapsed_s = 0.0
    if starts and ends:
        elapsed_s = max((max(ends) - min(starts)).total_seconds(), 0.0)
    elif durations:
        elapsed_s = sum(durations)

    total = len(rows)
    succeeded = sum(1 for row in rows if _status(row) in SUCCESS_STATUSES)
    failed = total - succeeded
    pages = sum(int(row.get("pages") or row.get("page_count") or 0) for row in rows)
    bytes_processed = sum(
        int(row.get("bytes_processed") or row.get("size_bytes") or 0) for row in rows
    )
    worker_time_s = sum(durations)

    return {
        "documents": total,
        "succeeded": succeeded,
        "failed": failed,
        "pages": pages,
        "bytes_processed": bytes_processed,
        "elapsed_s": elapsed_s,
        "worker_time_s": worker_time_s,
        "documents_per_s": _rate(total, elapsed_s),
        "pages_per_s": _rate(pages, elapsed_s),
        "mb_per_s": _rate(bytes_processed / 1_000_000, elapsed_s),
        "worker_pages_per_s": _rate(pages, worker_time_s),
        "latency_s": {
            "min": min(durations) if durations else 0.0,
            "p50": percentile(durations, 50),
            "p95": percentile(durations, 95),
            "max": max(durations) if durations else 0.0,
        },
        "by_status": _count_by(rows, "status"),
        "by_engine": _count_by(rows, "engine"),
    }


def format_throughput_report(summary: Mapping[str, Any], *, title: str = "OCR Throughput") -> str:
    latency_payload = summary.get("latency_s")
    latency: Mapping[str, Any] = latency_payload if isinstance(latency_payload, Mapping) else {}
    lines = [
        f"# {title}",
        "",
        f"- documents: {summary.get('documents', 0)}",
        f"- succeeded: {summary.get('succeeded', 0)}",
        f"- failed: {summary.get('failed', 0)}",
        f"- pages: {summary.get('pages', 0)}",
        f"- elapsed_s: {_fmt(summary.get('elapsed_s', 0.0))}",
        f"- documents_per_s: {_fmt(summary.get('documents_per_s', 0.0))}",
        f"- pages_per_s: {_fmt(summary.get('pages_per_s', 0.0))}",
        f"- mb_per_s: {_fmt(summary.get('mb_per_s', 0.0))}",
        f"- latency_p50_s: {_fmt(latency.get('p50', 0.0))}",
        f"- latency_p95_s: {_fmt(latency.get('p95', 0.0))}",
    ]
    return "\n".join(lines) + "\n"


def load_records_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_records_jsonl(
    records: Iterable[Any], path: str | Path
) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(_as_mapping(record), sort_keys=True))
            handle.write("\n")
    return output


def percentile(values: Iterable[float], percent: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return 0.0
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * (percent / 100)
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    weight = rank - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _as_mapping(record: Any) -> dict[str, Any]:
    if hasattr(record, "to_dict"):
        mapped: dict[str, Any] = record.to_dict()
        return mapped
    return dict(record)


def _duration(row: Mapping[str, Any]) -> float | None:
    value = row.get("duration_s")
    if value is not None:
        return float(value)
    start = _parse_time(row.get("started_at"))
    end = _parse_time(row.get("ended_at"))
    if start is None or end is None:
        return None
    return max((end - start).total_seconds(), 0.0)


def _parse_time(value: Any) -> datetime | None:
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value
    text = str(value)
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed


def _status(row: Mapping[str, Any]) -> str:
    return str(row.get("status") or "").casefold()


def _rate(numerator: float, seconds: float) -> float:
    return numerator / seconds if seconds > 0 else 0.0


def _count_by(rows: Iterable[Mapping[str, Any]], field: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        value = str(row.get(field) or "")
        if not value:
            continue
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def _fmt(value: Any) -> str:
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return "0.000"
