from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SUCCESS_STATUSES = {"ok", "success", "completed"}


@dataclass(frozen=True)
class RunRecord:
    item_id: str
    status: str
    started_at: str | None = None
    ended_at: str | None = None
    duration_s: float | None = None
    pages: int | None = None
    bytes_processed: int | None = None
    engine: str = ""
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id,
            "status": self.status,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "duration_s": self.duration_s,
            "pages": self.pages,
            "bytes_processed": self.bytes_processed,
            "engine": self.engine,
            "error": self.error,
        }


def summarize_throughput(records: Iterable[RunRecord | Mapping[str, Any]]) -> dict[str, Any]:
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


def write_records_jsonl(records: Iterable[RunRecord | Mapping[str, Any]], path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(_as_mapping(record), sort_keys=True))
            handle.write("\n")
    return output


def run_benchmark(
    *,
    suite: str,
    runner_name: str,
    run_dir: str | Path,
    base_url: str = "http://127.0.0.1:8000/v1",
    api_key: str = "dummy",
    concurrency: int = 4,
    limit: int | None = None,
    enforce_strategy_routing: bool = True,
) -> dict[str, Any]:
    """Run a lightweight OCR/VLM benchmark over rendered image tasks.

    The function is intentionally conservative: it consumes an existing sample
    JSON/JSONL with `image_path` fields. Suite names such as `smoke` are
    recorded but not auto-expanded until a sample builder has produced files.
    """
    output = Path(run_dir)
    output.mkdir(parents=True, exist_ok=True)
    tasks = _load_suite_tasks(suite)
    if limit is not None:
        tasks = tasks[:limit]

    model_id = resolve_runner_model(runner_name)
    worker_count = max(1, concurrency)
    if worker_count == 1:
        records = [
            _run_benchmark_task(
                task,
                runner_name=runner_name,
                model_id=model_id,
                base_url=base_url,
                api_key=api_key,
                enforce_strategy_routing=enforce_strategy_routing,
            )
            for task in tasks
        ]
    else:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            records = list(
                executor.map(
                    lambda task: _run_benchmark_task(
                        task,
                        runner_name=runner_name,
                        model_id=model_id,
                        base_url=base_url,
                        api_key=api_key,
                        enforce_strategy_routing=enforce_strategy_routing,
                    ),
                    tasks,
                )
            )

    results_path = write_records_jsonl(records, output / "results.jsonl")
    summary = {
        "status": "ok",
        "suite": suite,
        "runner": runner_name,
        "model_id": model_id,
        "base_url": base_url,
        "concurrency": worker_count,
        "enforce_strategy_routing": enforce_strategy_routing,
        "tasks": len(tasks),
        "results": str(results_path),
        "throughput": summarize_throughput(records),
        "by_strategy": _count_by(records, "strategy"),
        "by_route": _count_by(records, "route"),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    return summary


def _run_benchmark_task(
    task: Mapping[str, Any],
    *,
    runner_name: str,
    model_id: str,
    base_url: str,
    api_key: str,
    enforce_strategy_routing: bool,
) -> dict[str, Any]:
    image_path = task.get("image_path")
    strategy = str(task.get("strategy") or "")
    route = ""
    started_at = now_iso()
    record: dict[str, Any] = {
        "item_id": str(task.get("sample_id") or task.get("id") or task.get("pdf_key") or ""),
        "runner": runner_name,
        "model_id": model_id,
        "engine": "vllm-chat",
        "image_path": image_path,
        "strategy": strategy,
        "cluster_id": task.get("cluster_id"),
        "strategy_confidence": task.get("strategy_confidence"),
        "started_at": started_at,
        "pages": 1,
        "bytes_processed": _file_size(image_path),
    }
    try:
        if not image_path:
            raise ValueError("task is missing image_path")
        if enforce_strategy_routing and strategy in {
            "native_text_body",
            "native_pdfplumber_table",
        }:
            route = "native_strategy_skip"
            record.update(
                {
                    "status": "skipped",
                    "ended_at": now_iso(),
                    "route": route,
                    "skip_reason": "native_text_layer_strategy",
                }
            )
            record["duration_s"] = _duration(record)
            return record

        page_number = int(task.get("page_number") or 1)
        if enforce_strategy_routing and strategy == "ocr_paddle_simple":
            try:
                route = "paddle_primary"
                result = _transcribe_with_paddle(image_path, page_number=page_number)
            except Exception:  # noqa: BLE001
                route = "paddle_to_vllm_fallback"
                result = _transcribe_with_vllm(
                    image_path,
                    page_number=page_number,
                    model_id=model_id,
                    base_url=base_url,
                    api_key=api_key,
                )
        else:
            route = "vlm_primary"
            result = _transcribe_with_vllm(
                image_path,
                page_number=page_number,
                model_id=model_id,
                base_url=base_url,
                api_key=api_key,
            )

        if enforce_strategy_routing and strategy == "peer_review_escalation":
            route = "vlm_escalation"

        record.update(
            {
                "status": "ok",
                "ended_at": now_iso(),
                "text": result.text,
                "result": result.to_dict(),
                "route": route,
            }
        )
    except Exception as exc:  # noqa: BLE001
        record.update({"status": "error", "ended_at": now_iso(), "error": str(exc)})
        if route:
            record["route"] = route
    record["duration_s"] = _duration(record)
    return record


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


def resolve_runner_model(
    runner_name: str,
    *,
    config_path: str | Path = "configs/models.yaml",
) -> str:
    """Resolve a benchmark runner alias to the actual vLLM/OpenAI model id."""

    config = Path(config_path)
    if not config.exists():
        return runner_name
    try:
        import yaml

        data = yaml.safe_load(config.read_text(encoding="utf-8")) or {}
    except Exception:  # noqa: BLE001
        return runner_name
    models = data.get("vision_language_models")
    if not isinstance(models, Mapping):
        return runner_name
    entry = models.get(runner_name)
    if not isinstance(entry, Mapping):
        return runner_name
    model = entry.get("model_id") or entry.get("model")
    return str(model) if model else runner_name


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


def now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _load_suite_tasks(suite: str) -> list[dict[str, Any]]:
    path = Path(suite)
    if not path.exists():
        return []
    if path.suffix == ".jsonl":
        return load_records_jsonl(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, Mapping):
        samples = payload.get("samples")
        if isinstance(samples, list):
            return [dict(item) for item in samples if isinstance(item, Mapping)]
    if isinstance(payload, list):
        return [dict(item) for item in payload if isinstance(item, Mapping)]
    return []


def _file_size(path_text: Any) -> int:
    if not path_text:
        return 0
    try:
        return Path(str(path_text)).stat().st_size
    except OSError:
        return 0


def _transcribe_with_vllm(
    image_path: Any,
    *,
    page_number: int,
    model_id: str,
    base_url: str,
    api_key: str,
) -> Any:
    from gwanbo_ocr.runners.vllm_chat import VllmChatRunner

    runner = VllmChatRunner(
        model=model_id,
        base_url=base_url,
        api_key=api_key,
        strict_json=False,
    )
    return runner.transcribe(
        image_path,
        page_number=page_number,
        language_hint="ko,en",
    )


def _transcribe_with_paddle(image_path: Any, *, page_number: int) -> Any:
    from gwanbo_ocr.runners.paddle import PaddleOcrRunner

    runner = PaddleOcrRunner(lang="korean")
    return runner.transcribe(image_path, page_number=page_number)


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


def _as_mapping(record: RunRecord | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(record, RunRecord):
        return record.to_dict()
    return dict(record)
