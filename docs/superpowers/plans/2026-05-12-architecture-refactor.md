# Architecture Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

> **Status (2026-05-13):** Historical record. The main architecture refactor described here has already been implemented in recent commits; unchecked boxes below do not represent current task state.

**Goal:** Split `bench.py` → `bench/` package, `peer_review.py` → `peers/` package, extract `runners/preflight.py`, and thin `cli.strategy_pipeline` via `strategy.run_pipeline()`.

**Architecture:** Each unit owns one responsibility and communicates through clear import boundaries. No backward-compat shims — all callers updated directly. Tests restructured to mirror new module layout.

**Tech Stack:** Python 3.12, Typer, pytest, ruff, mypy

---

## File Map

### New files (create)
- `src/gwanbo_ocr/runners/preflight.py` — preflight endpoint validation
- `src/gwanbo_ocr/bench/__init__.py` — package public API
- `src/gwanbo_ocr/bench/run.py` — RunRecord, run_benchmark, _run_benchmark_task
- `src/gwanbo_ocr/bench/score.py` — score_benchmark
- `src/gwanbo_ocr/bench/report.py` — summarize_throughput, format_throughput_report, I/O helpers
- `src/gwanbo_ocr/peers/__init__.py` — orchestration public API
- `src/gwanbo_ocr/peers/_helpers.py` — shared utilities for all peer method files
- `src/gwanbo_ocr/peers/native.py` — extract_native_text
- `src/gwanbo_ocr/peers/pdfplumber.py` — extract_pdfplumber
- `src/gwanbo_ocr/peers/markitdown.py` — extract_markitdown
- `src/gwanbo_ocr/peers/paddle.py` — extract_paddle_ocr
- `src/gwanbo_ocr/peers/vlm.py` — extract_vlm_ocr
- `tests/peers/__init__.py`
- `tests/peers/test_extract.py`
- `tests/peers/test_review.py`
- `tests/peers/test_orchestration.py`
- `tests/bench/test_report.py`
- `tests/bench/test_run.py`
- `tests/bench/test_score.py`

### Modified files
- `src/gwanbo_ocr/bench.py` → deleted after Task 2
- `src/gwanbo_ocr/peer_review.py` → deleted after Task 4
- `src/gwanbo_ocr/strategy.py` — add `run_pipeline()`; update peer_review → peers import in Task 4
- `src/gwanbo_ocr/cli.py` — thin `strategy_pipeline`; update peer_review → peers import
- `tests/bench/test_bench.py` — monkeypatch target update in Task 1; deleted after Task 5
- `tests/test_peer_review.py` — import update in Task 4; deleted after Task 5

---

## Task 1: Extract runners/preflight.py

**Files:**
- Create: `src/gwanbo_ocr/runners/preflight.py`
- Modify: `src/gwanbo_ocr/bench.py`
- Modify: `tests/bench/test_bench.py`

- [ ] **Step 1: Create runners/preflight.py**

```python
# src/gwanbo_ocr/runners/preflight.py
from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any
from urllib import error, request


def preflight_openai_endpoint(
    *,
    base_url: str,
    api_key: str,
    model_id: str,
    timeout_s: float,
) -> dict[str, Any]:
    """Check endpoint reachability and confirm model_id is served."""
    models_url = _models_endpoint(base_url)
    headers = {"Accept": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = request.Request(models_url, headers=headers)
    try:
        with request.urlopen(req, timeout=timeout_s) as response:
            status_value = getattr(response, "status", None)
            if status_value is None:
                status_value = response.getcode()
            status = int(status_value)
            payload = _read_json_response(response, models_url)
            served_models = _served_model_ids(payload)
            if model_id not in served_models:
                available = ", ".join(served_models) if served_models else "<none>"
                raise RuntimeError(
                    f"vLLM preflight failed at {models_url}: model {model_id!r} "
                    f"is not served (available: {available})"
                )
            return {
                "status": "ok",
                "url": models_url,
                "http_status": status,
                "model_id": model_id,
                "served_models": served_models,
            }
    except error.HTTPError as exc:
        status = int(exc.code)
        raise RuntimeError(
            f"vLLM preflight failed ({status}) at {models_url}: {exc.reason}"
        ) from exc
    except RuntimeError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"vLLM preflight failed at {models_url}: {exc}") from exc


def _models_endpoint(base_url: str) -> str:
    normalized = base_url.rstrip("/")
    if normalized.endswith("/v1"):
        return f"{normalized}/models"
    return f"{normalized}/v1/models"


def _read_json_response(response: Any, url: str) -> Mapping[str, Any]:
    try:
        body = response.read()
        text = body.decode("utf-8") if isinstance(body, bytes) else str(body)
        payload = json.loads(text)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"vLLM preflight failed at {url}: invalid JSON response") from exc
    if not isinstance(payload, Mapping):
        raise RuntimeError(f"vLLM preflight failed at {url}: expected JSON object response")
    return payload


def _served_model_ids(payload: Mapping[str, Any]) -> list[str]:
    data = payload.get("data")
    if not isinstance(data, list):
        return []
    model_ids = {
        str(item.get("id"))
        for item in data
        if isinstance(item, Mapping) and item.get("id") not in {None, ""}
    }
    return sorted(model_ids)
```

- [ ] **Step 2: Update bench.py — remove preflight functions, add import**

In `src/gwanbo_ocr/bench.py`:

Remove the top-level import:
```python
from urllib import error, request
```

Remove the four functions: `_preflight_openai_endpoint`, `_models_endpoint`, `_read_json_response`, `_served_model_ids` (lines 437–509).

Replace the call site in `run_benchmark` (line 171):
```python
# before
        preflight = _preflight_openai_endpoint(
            base_url=base_url,
            api_key=api_key,
            model_id=model_id,
            timeout_s=preflight_timeout_s,
        )
```
```python
# after
        from gwanbo_ocr.runners.preflight import preflight_openai_endpoint
        preflight = preflight_openai_endpoint(
            base_url=base_url,
            api_key=api_key,
            model_id=model_id,
            timeout_s=preflight_timeout_s,
        )
```

- [ ] **Step 3: Update monkeypatch targets in test_bench.py**

There are three tests that patch `bench.request.urlopen`. Each needs updating:

Find every occurrence of:
```python
    import gwanbo_ocr.bench as bench
    ...
    monkeypatch.setattr(bench.request, "urlopen", fake_urlopen)
```
And:
```python
    import gwanbo_ocr.bench as bench
    monkeypatch.setattr(bench.request, "urlopen", lambda *_args, **_kwargs: FakeResponse())
```

Replace with:
```python
    import gwanbo_ocr.runners.preflight as preflight_mod
    ...
    monkeypatch.setattr(preflight_mod.request, "urlopen", fake_urlopen)
```
and:
```python
    import gwanbo_ocr.runners.preflight as preflight_mod
    monkeypatch.setattr(preflight_mod.request, "urlopen", lambda *_args, **_kwargs: FakeResponse())
```

Affected tests: `test_run_benchmark_preflight_validates_served_model` (line 130–133),
`test_run_benchmark_preflight_rejects_unserved_model` (line 172–174),
`test_run_benchmark_preflight_rejects_http_errors` (line 186+ — check for `bench.request`).

- [ ] **Step 4: Run tests**

```bash
cd /root/gwanbo-ocr && source .venv/bin/activate && pytest tests/bench/test_bench.py -v
```
Expected: all 12 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/gwanbo_ocr/runners/preflight.py src/gwanbo_ocr/bench.py tests/bench/test_bench.py
git commit -m "refactor: extract runners/preflight.py from bench.py"
```

---

## Task 2: Create bench/ Package

**Files:**
- Create: `src/gwanbo_ocr/bench/report.py`
- Create: `src/gwanbo_ocr/bench/score.py`
- Create: `src/gwanbo_ocr/bench/run.py`
- Create: `src/gwanbo_ocr/bench/__init__.py`
- Delete: `src/gwanbo_ocr/bench.py`

- [ ] **Step 1: Create bench/report.py**

```python
# src/gwanbo_ocr/bench/report.py
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
        return record.to_dict()
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
```

Note: `summarize_throughput` has a local import of `RunRecord` to resolve the forward reference in the type hint. Alternatively, simplify the signature to `Iterable[Any]` — choose whichever avoids the circular import cleanly. The simplest approach: remove `RunRecord` from the `Iterable` type hint in report.py since it's called with dicts in practice:

```python
def summarize_throughput(records: Iterable[Any]) -> dict[str, Any]:
    rows = [_as_mapping(record) for record in records]
    ...
```

- [ ] **Step 2: Create bench/score.py**

```python
# src/gwanbo_ocr/bench/score.py
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
```

- [ ] **Step 3: Create bench/run.py**

```python
# src/gwanbo_ocr/bench/run.py
from __future__ import annotations

import json
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .report import _count_by, _duration, summarize_throughput, write_records_jsonl


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
    preflight_vllm: bool = False,
    preflight_timeout_s: float = 5.0,
) -> dict[str, Any]:
    from gwanbo_ocr.runners.preflight import preflight_openai_endpoint

    output = Path(run_dir)
    output.mkdir(parents=True, exist_ok=True)
    tasks = _load_suite_tasks(suite)
    if limit is not None:
        tasks = tasks[:limit]

    model_id = resolve_runner_model(runner_name)
    worker_count = max(1, concurrency)

    preflight: dict[str, Any]
    if not preflight_vllm:
        preflight = {"status": "disabled"}
    elif not tasks:
        preflight = {"status": "skipped_no_tasks"}
    elif not _tasks_require_vllm(tasks, enforce_strategy_routing=enforce_strategy_routing):
        preflight = {"status": "skipped_no_vllm_route"}
    else:
        preflight = preflight_openai_endpoint(
            base_url=base_url,
            api_key=api_key,
            model_id=model_id,
            timeout_s=preflight_timeout_s,
        )

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
        "preflight": preflight,
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


def now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _load_suite_tasks(suite: str) -> list[dict[str, Any]]:
    path = Path(suite)
    if not path.exists():
        return []
    if path.suffix == ".jsonl":
        from .report import load_records_jsonl
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


def _tasks_require_vllm(
    tasks: Any,
    *,
    enforce_strategy_routing: bool,
) -> bool:
    if not enforce_strategy_routing:
        return any(True for _ in tasks)
    for task in tasks:
        strategy = str(task.get("strategy") or "")
        if strategy in {"native_text_body", "native_pdfplumber_table"}:
            continue
        return True
    return False


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
```

- [ ] **Step 4: Create bench/__init__.py**

```python
# src/gwanbo_ocr/bench/__init__.py
from .report import (
    SUCCESS_STATUSES,
    format_throughput_report,
    load_records_jsonl,
    percentile,
    summarize_throughput,
    write_records_jsonl,
)
from .run import RunRecord, now_iso, resolve_runner_model, run_benchmark
from .score import score_benchmark

__all__ = [
    "RunRecord",
    "SUCCESS_STATUSES",
    "format_throughput_report",
    "load_records_jsonl",
    "now_iso",
    "percentile",
    "resolve_runner_model",
    "run_benchmark",
    "score_benchmark",
    "summarize_throughput",
    "write_records_jsonl",
]
```

- [ ] **Step 5: Run tests to verify package works before deleting bench.py**

```bash
pytest tests/bench/ tests/test_cli.py -v
```
Expected: all tests pass (cli.py imports from `gwanbo_ocr.bench` still resolve via `__init__.py`).

- [ ] **Step 6: Delete old bench.py**

```bash
rm src/gwanbo_ocr/bench.py
```

- [ ] **Step 7: Run full test suite**

```bash
pytest -v
```
Expected: all tests pass.

- [ ] **Step 8: Commit**

```bash
git add src/gwanbo_ocr/bench/ && git rm src/gwanbo_ocr/bench.py
git commit -m "refactor: convert bench.py to bench/ package (run/score/report)"
```

---

## Task 3: Add strategy.run_pipeline() and Thin CLI

**Files:**
- Modify: `src/gwanbo_ocr/strategy.py`
- Modify: `src/gwanbo_ocr/cli.py`

- [ ] **Step 1: Add run_pipeline() to strategy.py**

Append to the end of `src/gwanbo_ocr/strategy.py` (after the last helper function):

```python
def run_pipeline(
    manifest: Path,
    output: Path,
    *,
    runner: str = "qwen36_baseline",
    base_url: str = "http://127.0.0.1:8000/v1",
    api_key: str = "dummy",
    sample_per_bucket: int = 20,
    profile_max_pages: int = 3,
    render_max_pages: int = 1,
    cluster_sample_keys: int = 20,
    workers: int = 1,
    render_workers: int = 4,
    concurrency: int = 4,
    enforce_strategy_routing: bool = True,
    preflight_vllm: bool = True,
    preflight_timeout_s: float = 5.0,
    run_peer: bool = True,
    run_paddle: bool = False,
    limit: int | None = None,
) -> dict[str, Any]:
    """Run the end-to-end strategy pipeline: profile → cluster → render → bench → evaluate."""
    from gwanbo_ocr.bench import run_benchmark, score_benchmark
    from gwanbo_ocr.pdf.io import write_json_atomic
    from gwanbo_ocr.pdf.profile import profile_manifest
    from gwanbo_ocr.peer_review import aggregate_peer_scores, run_peer_review_manifest
    from gwanbo_ocr.render import render_manifest

    output.mkdir(parents=True, exist_ok=True)
    profiles_dir = output / "profiles"
    clusters_dir = output / "clusters"
    images_dir = output / "images"
    suite_path = output / "bench" / "strategy_suite.jsonl"
    bench_run_dir = output / "bench" / runner
    bench_report_dir = output / "reports" / runner
    peer_dir = output / "peer_review"
    peer_report_dir = output / "reports" / "peer"
    strategy_eval_dir = output / "strategy_eval"

    profile_summary = profile_manifest(
        manifest_path=manifest,
        output_dir=profiles_dir,
        max_pages=None if profile_max_pages == 0 else profile_max_pages,
        workers=workers,
        sample_per_bucket=None if sample_per_bucket == 0 else sample_per_bucket,
        limit=limit,
    )
    cluster_summary = cluster_profiles(
        profiles_path=profiles_dir / "manifest.jsonl",
        output_dir=clusters_dir,
        sample_keys=cluster_sample_keys,
    )
    render_summary = render_manifest(
        manifest_path=manifest,
        output_dir=images_dir,
        max_pages=None if render_max_pages == 0 else render_max_pages,
        workers=render_workers,
        limit=limit,
    )
    suite_summary = build_strategy_benchmark_suite(
        render_manifest_path=images_dir / "manifest.jsonl",
        clusters_path=clusters_dir / "cluster_manifest.jsonl",
        output_path=suite_path,
    )
    bench_run_summary = run_benchmark(
        suite=str(suite_path),
        runner_name=runner,
        run_dir=bench_run_dir,
        base_url=base_url,
        api_key=api_key,
        concurrency=concurrency,
        enforce_strategy_routing=enforce_strategy_routing,
        preflight_vllm=preflight_vllm,
        preflight_timeout_s=preflight_timeout_s,
        limit=limit,
    )
    bench_score_summary = score_benchmark(run_dir=bench_run_dir, output_dir=bench_report_dir)

    peer_run_summary: dict[str, Any] | None = None
    peer_score_summary: dict[str, Any] | None = None
    peer_score_report_path: Path | None = None
    if run_peer:
        peer_run_summary = run_peer_review_manifest(
            manifest_path=manifest,
            output_dir=peer_dir,
            run_paddle=run_paddle,
            max_pages=None if render_max_pages == 0 else render_max_pages,
            workers=workers,
            limit=limit,
        )
        peer_score_summary = aggregate_peer_scores(peer_dir)
        peer_report_dir.mkdir(parents=True, exist_ok=True)
        peer_score_report_path = peer_report_dir / "peer_score_report.json"
        write_json_atomic(peer_score_report_path, peer_score_summary)

    eval_summary = evaluate_clusters(
        clusters_path=clusters_dir / "cluster_manifest.jsonl",
        output_dir=strategy_eval_dir,
        bench_scores_path=bench_report_dir / "scores.jsonl",
        peer_score_report_path=peer_score_report_path,
    )

    by_route_payload = bench_run_summary.get("by_route")
    by_route = by_route_payload if isinstance(by_route_payload, dict) else {}
    throughput_payload = bench_run_summary.get("throughput")
    throughput = throughput_payload if isinstance(throughput_payload, dict) else {}
    by_status_payload = throughput.get("by_status")
    by_status = by_status_payload if isinstance(by_status_payload, dict) else {}
    total_tasks = int(bench_run_summary.get("tasks") or 0)
    fallback_count = int(by_route.get("paddle_to_vllm_fallback") or 0)
    error_count = int(by_status.get("error") or 0)
    route_metrics = {
        "fallback_count": fallback_count,
        "fallback_rate": round((fallback_count / total_tasks), 4) if total_tasks else 0.0,
        "error_count": error_count,
        "error_rate": round((error_count / total_tasks), 4) if total_tasks else 0.0,
    }

    pipeline_summary = {
        "status": "ok",
        "manifest": str(manifest),
        "output": str(output),
        "profile": profile_summary,
        "cluster": cluster_summary,
        "render": render_summary,
        "suite": suite_summary,
        "bench_run": bench_run_summary,
        "bench_score": bench_score_summary,
        "route_metrics": route_metrics,
        "peer_run": peer_run_summary,
        "peer_score": peer_score_summary,
        "strategy_eval": eval_summary,
    }
    write_json_atomic(output / "pipeline_summary.json", pipeline_summary)
    return pipeline_summary
```

Also add `Path` and `Any` to the imports at the top of `strategy.py` if not already present. `Any` from `typing` is already imported. Add `from pathlib import Path` if missing.

- [ ] **Step 2: Replace strategy_pipeline body in cli.py**

Replace the entire `strategy_pipeline` function body (lines 316–442) with:

```python
@strategy_app.command("pipeline")
def strategy_pipeline(
    manifest: Path = typer.Option(..., "--manifest", help="Input PDF manifest JSONL."),
    output: Path = typer.Option(..., help="Pipeline output root directory."),
    runner: str = typer.Option("qwen36_baseline", help="Bench runner/model alias."),
    base_url: str = typer.Option("http://127.0.0.1:8000/v1", help="OpenAI-compatible base URL."),
    api_key: str = typer.Option("dummy", help="OpenAI-compatible API key."),
    sample_per_bucket: int = typer.Option(20, "--sample-per-bucket", help="Max profile rows per theme/year/category bucket; 0 means all."),
    profile_max_pages: int = typer.Option(3, help="Pages to inspect for profiling; 0 means all."),
    render_max_pages: int = typer.Option(1, help="Pages to render per PDF; 0 means all selected."),
    cluster_sample_keys: int = typer.Option(20, "--cluster-sample-keys", help="Representative pdf keys to keep per cluster."),
    workers: int = typer.Option(1, help="Worker count for profile/peer stages."),
    render_workers: int = typer.Option(4, help="Worker count for render stage."),
    concurrency: int = typer.Option(4, help="Concurrent requests for bench run."),
    enforce_strategy_routing: bool = typer.Option(True, "--enforce-strategy-routing/--no-enforce-strategy-routing", help="Route tasks by strategy and apply fallback policies during bench run."),
    preflight_vllm: bool = typer.Option(True, "--preflight-vllm/--no-preflight-vllm", help="Check VLM endpoint reachability before benchmark execution."),
    preflight_timeout_s: float = typer.Option(5.0, "--preflight-timeout-s", help="Timeout in seconds for VLM endpoint preflight check."),
    run_peer: bool = typer.Option(True, "--peer/--no-peer", help="Run peer review stages."),
    run_paddle: bool = typer.Option(False, "--paddle/--no-paddle", help="Enable PaddleOCR peer."),
    limit: int | None = typer.Option(None, help="Optional row/task cap for quick runs."),
) -> None:
    """Run end-to-end strategy pipeline from profiling to strategy evaluation."""
    from gwanbo_ocr.strategy import run_pipeline

    pipeline_summary = run_pipeline(
        manifest=manifest,
        output=output,
        runner=runner,
        base_url=base_url,
        api_key=api_key,
        sample_per_bucket=sample_per_bucket,
        profile_max_pages=profile_max_pages,
        render_max_pages=render_max_pages,
        cluster_sample_keys=cluster_sample_keys,
        workers=workers,
        render_workers=render_workers,
        concurrency=concurrency,
        enforce_strategy_routing=enforce_strategy_routing,
        preflight_vllm=preflight_vllm,
        preflight_timeout_s=preflight_timeout_s,
        run_peer=run_peer,
        run_paddle=run_paddle,
        limit=limit,
    )
    _echo_summary(
        {
            "status": "ok",
            "output": str(output),
            "pipeline_summary": str(output / "pipeline_summary.json"),
            "evaluated_clusters": pipeline_summary.get("strategy_eval", {}).get("evaluated_clusters", 0),
            "bench_tasks": pipeline_summary.get("bench_run", {}).get("tasks", 0),
        }
    )
```

Remove the `from typing import Any` import in `strategy_pipeline` if it was only used there (cli.py still imports `Any` from typing at the top already — check line 6).

- [ ] **Step 3: Run tests**

```bash
pytest tests/test_strategy.py tests/test_cli.py -v
```
Expected: all tests pass.

- [ ] **Step 4: Commit**

```bash
git add src/gwanbo_ocr/strategy.py src/gwanbo_ocr/cli.py
git commit -m "refactor: move strategy_pipeline orchestration to strategy.run_pipeline()"
```

---

## Task 4: Create peers/ Package

**Files:**
- Create: `src/gwanbo_ocr/peers/_helpers.py`
- Create: `src/gwanbo_ocr/peers/native.py`
- Create: `src/gwanbo_ocr/peers/pdfplumber.py`
- Create: `src/gwanbo_ocr/peers/markitdown.py`
- Create: `src/gwanbo_ocr/peers/paddle.py`
- Create: `src/gwanbo_ocr/peers/vlm.py`
- Create: `src/gwanbo_ocr/peers/__init__.py`
- Modify: `src/gwanbo_ocr/strategy.py` (peer_review → peers import)
- Modify: `src/gwanbo_ocr/cli.py` (peer_review → peers import)
- Modify: `tests/test_peer_review.py` (update import block)
- Delete: `src/gwanbo_ocr/peer_review.py`

- [ ] **Step 1: Create peers/_helpers.py**

```python
# src/gwanbo_ocr/peers/_helpers.py
from __future__ import annotations

import re
import signal
from datetime import datetime
from pathlib import Path
from typing import Any

SAMPLE_CHARS_DEFAULT = 1200


def _skipped(reason: str, *, method: str) -> dict[str, Any]:
    return {
        "status": "skipped",
        "method": method,
        "skip_reason": reason,
        "text_extractable": False,
        "text_chars": 0,
        "sample_text": "",
        "error": None,
    }


def _error(message: str, *, method: str) -> dict[str, Any]:
    return {
        "status": "error",
        "method": method,
        "text_extractable": False,
        "text_chars": 0,
        "sample_text": "",
        "error": message,
    }


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _norm_for_sim(text: str) -> str:
    return re.sub(r"\W+", "", text).lower()[:2000]


def _with_timeout(fn: Any, timeout_seconds: int, *, method: str) -> dict[str, Any]:
    previous = None
    if timeout_seconds > 0:
        previous = signal.signal(signal.SIGALRM, _alarm_handler)
        signal.alarm(timeout_seconds)
    try:
        result = fn()
        return (
            result
            if isinstance(result, dict)
            else _error("method returned non-dict", method=method)
        )
    except Exception as exc:  # noqa: BLE001
        return _error(str(exc), method=method)
    finally:
        if timeout_seconds > 0:
            signal.alarm(0)
            if previous is not None:
                signal.signal(signal.SIGALRM, previous)


def _alarm_handler(_signum: int, _frame: Any) -> None:
    raise TimeoutError("peer review extraction timed out")


def _render_pages(
    pdf_path: Path,
    *,
    image_dir: Path,
    max_pages: int | None,
    dpi: int,
) -> dict[str, Any]:
    try:
        from gwanbo_ocr.render import page_count, render_page_to_png_bytes
    except ImportError:
        return {"status": "error", "error": "PyMuPDF (pymupdf) is not installed"}

    try:
        image_dir.mkdir(parents=True, exist_ok=True)
        total = page_count(pdf_path)
        pages_to_render = total if max_pages is None else min(total, max_pages)
        pages: list[dict[str, Any]] = []
        for idx in range(pages_to_render):
            png = render_page_to_png_bytes(pdf_path, page_index=idx, dpi=dpi)
            img_path = image_dir / f"page_{idx + 1:03d}.png"
            img_path.write_bytes(png)
            pages.append({"page_index": idx, "path": str(img_path)})
        return {"status": "ok", "page_count": total, "pages": pages}
    except Exception as exc:  # noqa: BLE001
        return {"status": "error", "error": str(exc)}


def _make_vlm_runner(base_url: str, model: str | None, api_key: str) -> Any:
    from gwanbo_ocr.runners.vllm import VllmChatRunner

    return VllmChatRunner(model=model or "default", base_url=base_url, api_key=api_key)


def _scope(max_pages: int | None) -> str:
    return "all_pages" if max_pages is None else f"first_{max_pages}_pages"


def _iso_now() -> str:
    return datetime.now().isoformat()
```

- [ ] **Step 2: Create peers/native.py**

```python
# src/gwanbo_ocr/peers/native.py
from __future__ import annotations

from pathlib import Path
from typing import Any

from ._helpers import SAMPLE_CHARS_DEFAULT


def extract_native_text(
    pdf_path: Path,
    *,
    max_pages: int | None = 1,
    sample_chars: int = SAMPLE_CHARS_DEFAULT,
    timeout_seconds: int = 30,
) -> dict[str, Any]:
    """Extract text using pypdf / builtin parser."""
    from gwanbo_ocr.pdf.text import analyze_pdf_text

    metadata = analyze_pdf_text(
        pdf_path,
        include_sample=True,
        sample_chars=sample_chars,
        max_pages=max_pages,
        timeout_seconds=timeout_seconds,
    )
    sample = str(metadata.get("sample_text") or "")
    return {
        "status": str(metadata.get("status") or "unknown"),
        "method": "pypdf.extract_text",
        "text_extractable": bool(metadata.get("text_extractable")),
        "pages": metadata.get("pages"),
        "scanned_pages": metadata.get("scanned_pages"),
        "text_chars": int(metadata.get("total_chars") or len(sample)),
        "sample_text": sample[:sample_chars],
        "error": metadata.get("error"),
    }
```

- [ ] **Step 3: Create peers/pdfplumber.py**

```python
# src/gwanbo_ocr/peers/pdfplumber.py
from __future__ import annotations

from pathlib import Path
from typing import Any

from ._helpers import SAMPLE_CHARS_DEFAULT, _normalize, _skipped, _with_timeout

_pdfplumber: Any
try:
    import pdfplumber as _pdfplumber  # type: ignore[import-not-found]
except ImportError:
    _pdfplumber = None


def extract_pdfplumber(
    pdf_path: Path,
    *,
    max_pages: int | None = 1,
    sample_chars: int = SAMPLE_CHARS_DEFAULT,
    timeout_seconds: int = 30,
) -> dict[str, Any]:
    """Extract text using pdfplumber.page.extract_text()."""
    plumber = _pdfplumber
    if plumber is None:
        return _skipped("pdfplumber is not installed", method="pdfplumber.extract_text")

    def _run() -> dict[str, Any]:
        with plumber.open(str(pdf_path)) as doc:
            page_count = len(doc.pages)
            pages_to_scan = page_count if max_pages is None else min(page_count, max_pages)
            parts: list[str] = []
            total_chars = 0
            page_errors: list[dict[str, Any]] = []
            for idx in range(pages_to_scan):
                try:
                    text = doc.pages[idx].extract_text() or ""
                    text = _normalize(text)
                    total_chars += len(text)
                    if text and len(" ".join(parts)) < sample_chars:
                        parts.append(text)
                except Exception as exc:  # noqa: BLE001
                    page_errors.append({"page_index": idx, "error": str(exc)})
            result: dict[str, Any] = {
                "status": "ok",
                "method": "pdfplumber.extract_text",
                "text_extractable": total_chars > 0,
                "pages": page_count,
                "scanned_pages": pages_to_scan,
                "text_chars": total_chars,
                "sample_text": " ".join(parts)[:sample_chars],
                "error": None,
            }
            if page_errors:
                result["page_errors"] = page_errors
                result["page_error_count"] = len(page_errors)
            return result

    return _with_timeout(_run, timeout_seconds, method="pdfplumber.extract_text")
```

- [ ] **Step 4: Create peers/markitdown.py**

```python
# src/gwanbo_ocr/peers/markitdown.py
from __future__ import annotations

from pathlib import Path
from typing import Any

from ._helpers import SAMPLE_CHARS_DEFAULT, _normalize, _skipped, _with_timeout

_MarkItDown: Any
try:
    from markitdown import MarkItDown as _MarkItDown  # type: ignore[import-not-found]
except ImportError:
    _MarkItDown = None


def extract_markitdown(
    pdf_path: Path,
    *,
    sample_chars: int = SAMPLE_CHARS_DEFAULT,
    timeout_seconds: int = 30,
) -> dict[str, Any]:
    """Extract text using MarkItDown.convert()."""
    MarkItDown = _MarkItDown
    if MarkItDown is None:
        return _skipped("markitdown is not installed", method="MarkItDown.convert")

    def _run() -> dict[str, Any]:
        converter = MarkItDown(enable_plugins=False)
        converted = converter.convert(str(pdf_path))
        text = _normalize(str(getattr(converted, "text_content", "") or ""))
        return {
            "status": "ok",
            "method": "MarkItDown.convert",
            "text_extractable": bool(text),
            "text_chars": len(text),
            "sample_text": text[:sample_chars],
            "error": None,
        }

    return _with_timeout(_run, timeout_seconds, method="MarkItDown.convert")
```

- [ ] **Step 5: Create peers/paddle.py**

```python
# src/gwanbo_ocr/peers/paddle.py
from __future__ import annotations

from pathlib import Path
from typing import Any

from ._helpers import (
    SAMPLE_CHARS_DEFAULT,
    _error,
    _normalize,
    _render_pages,
    _skipped,
    _with_timeout,
)


def extract_paddle_ocr(
    pdf_path: Path,
    *,
    image_dir: Path,
    max_pages: int | None = 1,
    sample_chars: int = SAMPLE_CHARS_DEFAULT,
    dpi: int = 200,
    lang: str = "korean",
    timeout_seconds: int = 60,
) -> dict[str, Any]:
    """Render pages with PyMuPDF and transcribe with PaddleOCR."""
    try:
        from gwanbo_ocr.runners.paddle import PaddleOcrRunner
    except ImportError:
        return _skipped("paddleocr is not installed", method="PaddleOCR")

    rendered = _render_pages(pdf_path, image_dir=image_dir, max_pages=max_pages, dpi=dpi)
    if rendered.get("status") != "ok":
        return _error(str(rendered.get("error", "render failed")), method="PaddleOCR")

    def _run() -> dict[str, Any]:
        runner = PaddleOcrRunner(lang=lang)
        parts: list[str] = []
        total_chars = 0
        page_results: list[dict[str, Any]] = []
        for page in rendered["pages"]:
            try:
                result = runner.transcribe(Path(page["path"]))
                text = _normalize(result.text)
                total_chars += len(text)
                if text and len(" ".join(parts)) < sample_chars:
                    parts.append(text)
                page_results.append(
                    {"page_index": page["page_index"], "status": "ok", "text_chars": len(text)}
                )
            except Exception as exc:  # noqa: BLE001
                page_results.append(
                    {"page_index": page["page_index"], "status": "error", "error": str(exc)}
                )
        return {
            "status": "ok",
            "method": "PaddleOCR",
            "text_extractable": total_chars > 0,
            "pages": rendered["page_count"],
            "scanned_pages": len(rendered["pages"]),
            "text_chars": total_chars,
            "sample_text": " ".join(parts)[:sample_chars],
            "dpi": dpi,
            "lang": lang,
            "page_results": page_results,
            "error": None,
        }

    return _with_timeout(_run, timeout_seconds, method="PaddleOCR")
```

- [ ] **Step 6: Create peers/vlm.py**

```python
# src/gwanbo_ocr/peers/vlm.py
from __future__ import annotations

from pathlib import Path
from typing import Any

from ._helpers import SAMPLE_CHARS_DEFAULT, _error, _normalize, _render_pages, _with_timeout


def extract_vlm_ocr(
    pdf_path: Path,
    *,
    image_dir: Path,
    runner: Any,
    max_pages: int | None = 1,
    sample_chars: int = SAMPLE_CHARS_DEFAULT,
    dpi: int = 200,
    timeout_seconds: int = 120,
) -> dict[str, Any]:
    """Render pages with PyMuPDF and transcribe with an OpenAI-compatible VLM."""
    rendered = _render_pages(pdf_path, image_dir=image_dir, max_pages=max_pages, dpi=dpi)
    if rendered.get("status") != "ok":
        return _error(str(rendered.get("error", "render failed")), method="VLM-OCR")

    method_label = f"VLM-OCR({getattr(runner, 'model', 'unknown')})"

    def _run() -> dict[str, Any]:
        parts: list[str] = []
        total_chars = 0
        page_results: list[dict[str, Any]] = []
        for page in rendered["pages"]:
            try:
                result = runner.transcribe(Path(page["path"]), page_number=page["page_index"] + 1)
                text = _normalize(result.text)
                total_chars += len(text)
                if text and len(" ".join(parts)) < sample_chars:
                    parts.append(text)
                page_results.append(
                    {
                        "page_index": page["page_index"],
                        "status": "ok",
                        "text_chars": len(text),
                        "latency_ms": result.data.get("latency_ms"),
                    }
                )
            except Exception as exc:  # noqa: BLE001
                page_results.append(
                    {"page_index": page["page_index"], "status": "error", "error": str(exc)}
                )
        return {
            "status": "ok",
            "method": method_label,
            "text_extractable": total_chars > 0,
            "pages": rendered["page_count"],
            "scanned_pages": len(rendered["pages"]),
            "text_chars": total_chars,
            "sample_text": " ".join(parts)[:sample_chars],
            "dpi": dpi,
            "model": getattr(runner, "model", None),
            "page_results": page_results,
            "error": None,
        }

    return _with_timeout(_run, timeout_seconds, method=method_label)
```

- [ ] **Step 7: Create peers/__init__.py**

```python
# src/gwanbo_ocr/peers/__init__.py
from __future__ import annotations

import concurrent.futures
import difflib
from pathlib import Path
from typing import Any

from ._helpers import SAMPLE_CHARS_DEFAULT, _iso_now, _make_vlm_runner, _norm_for_sim, _scope
from .markitdown import extract_markitdown
from .native import extract_native_text
from .paddle import extract_paddle_ocr
from .pdfplumber import extract_pdfplumber
from .vlm import extract_vlm_ocr

METHOD_PREFERENCE = {
    "vlm_ocr": 5,
    "paddle_ocr": 4,
    "pdfplumber": 3,
    "markitdown": 2,
    "native_text": 1,
}

__all__ = [
    "SAMPLE_CHARS_DEFAULT",
    "METHOD_PREFERENCE",
    "extract_markitdown",
    "extract_native_text",
    "extract_paddle_ocr",
    "extract_pdfplumber",
    "extract_vlm_ocr",
    "analyze_pdf_peer_review",
    "review_extraction_peers",
    "score_against_metadata",
    "decide_extraction",
    "run_peer_review_manifest",
    "aggregate_peer_scores",
]


def analyze_pdf_peer_review(
    pdf_path: Path,
    *,
    image_dir: Path,
    vlm_runner: Any | None = None,
    max_pages: int | None = 1,
    sample_chars: int = SAMPLE_CHARS_DEFAULT,
    dpi: int = 200,
    timeout_seconds: int = 60,
    run_paddle: bool = False,
    run_markitdown: bool = True,
    run_pdfplumber: bool = True,
    run_native_text: bool = True,
    ocr_lang: str = "korean",
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run all enabled peers and produce a complete peer-review report."""
    report: dict[str, Any] = {
        "path": str(pdf_path),
        "filename": pdf_path.name,
        "status": "ok",
        "analysis_scope": _scope(max_pages),
        "generated_at": _iso_now(),
        "peers": {},
        "review": {},
        "score": {},
        "decision": {},
    }

    peers = report["peers"]

    if run_native_text:
        peers["native_text"] = extract_native_text(
            pdf_path,
            max_pages=max_pages,
            sample_chars=sample_chars,
            timeout_seconds=timeout_seconds,
        )
    if run_pdfplumber:
        peers["pdfplumber"] = extract_pdfplumber(
            pdf_path,
            max_pages=max_pages,
            sample_chars=sample_chars,
            timeout_seconds=timeout_seconds,
        )
    if run_markitdown:
        peers["markitdown"] = extract_markitdown(
            pdf_path, sample_chars=sample_chars, timeout_seconds=timeout_seconds
        )
    if run_paddle:
        peers["paddle_ocr"] = extract_paddle_ocr(
            pdf_path,
            image_dir=image_dir / "paddle",
            max_pages=max_pages,
            sample_chars=sample_chars,
            dpi=dpi,
            lang=ocr_lang,
            timeout_seconds=timeout_seconds * 2,
        )
    if vlm_runner is not None:
        peers["vlm_ocr"] = extract_vlm_ocr(
            pdf_path,
            image_dir=image_dir / "vlm",
            runner=vlm_runner,
            max_pages=max_pages,
            sample_chars=sample_chars,
            dpi=dpi,
            timeout_seconds=timeout_seconds * 3,
        )

    report["review"] = review_extraction_peers(peers)
    if metadata:
        report["score"] = score_against_metadata(peers, metadata)
    report["decision"] = decide_extraction(peers, report["review"])

    if all(p.get("status") in {"error", "skipped"} for p in peers.values()):
        report["status"] = "error"
        report["error"] = "all extraction methods failed or were skipped"

    return report


def review_extraction_peers(peers: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Cross-compare all extraction peers."""
    summaries: dict[str, dict[str, Any]] = {}
    warnings: list[str] = []

    for name, peer in peers.items():
        status = str(peer.get("status") or "unknown")
        text_chars = int(peer.get("text_chars") or 0)
        summaries[name] = {
            "status": status,
            "text_chars": text_chars,
            "text_extractable": bool(peer.get("text_extractable")),
            "error": peer.get("error"),
        }
        if status == "error":
            warnings.append(f"{name} failed: {peer.get('error')}")
        elif status == "skipped":
            warnings.append(f"{name} skipped: {peer.get('skip_reason')}")

    best = _choose_best_method(peers)
    similarities = _pairwise_similarities(peers)

    if best:
        best_chars = int(peers[best].get("text_chars") or 0)
        for name, peer in peers.items():
            if name == best or peer.get("status") != "ok":
                continue
            chars = int(peer.get("text_chars") or 0)
            if best_chars and chars < best_chars * 0.3:
                warnings.append(f"{name} produced much less text than {best}")
    else:
        warnings.append("no successful extraction method produced text")

    return {
        "best_text_method": best,
        "peer_summaries": summaries,
        "pairwise_sample_similarity": similarities,
        "warnings": warnings,
    }


def score_against_metadata(
    peers: dict[str, dict[str, Any]],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """Score each peer's sample text against critical tokens from metadata."""
    from gwanbo_ocr.metrics import critical_token_f1, extract_critical_tokens

    token_fields = ("title", "category", "agency")
    reference_text = " ".join(str(metadata.get(f) or "") for f in token_fields if metadata.get(f))
    reference_tokens = extract_critical_tokens(reference_text)

    scores: dict[str, Any] = {"reference_tokens": sorted(reference_tokens)}
    for name, peer in peers.items():
        if peer.get("status") != "ok":
            scores[name] = {"status": peer.get("status"), "critical_token_f1": None}
            continue
        sample = str(peer.get("sample_text") or "")
        candidate_tokens = extract_critical_tokens(sample)
        f1 = critical_token_f1(reference_tokens, candidate_tokens)
        scores[name] = {
            "status": "ok",
            "critical_token_f1": round(f1.f1, 4),
            "precision": round(f1.precision, 4),
            "recall": round(f1.recall, 4),
        }

    ranked = [
        (name, scores[name]["critical_token_f1"])
        for name in scores
        if name != "reference_tokens" and scores[name].get("critical_token_f1") is not None
    ]
    ranked.sort(key=lambda item: item[1], reverse=True)
    scores["ranked_by_f1"] = [name for name, _ in ranked]
    return scores


def decide_extraction(
    peers: dict[str, dict[str, Any]],
    review: dict[str, Any],
) -> dict[str, Any]:
    """Recommend the best extraction strategy."""
    best_method = review.get("best_text_method")
    text_layer = ("native_text", "pdfplumber", "markitdown")
    ocr_methods = ("paddle_ocr", "vlm_ocr")

    text_layer_ok = any(
        peers.get(m, {}).get("status") == "ok" and int(peers.get(m, {}).get("text_chars") or 0) > 0
        for m in text_layer
    )
    ocr_ok = any(
        peers.get(m, {}).get("status") == "ok" and int(peers.get(m, {}).get("text_chars") or 0) > 0
        for m in ocr_methods
    )

    if text_layer_ok:
        return {
            "text_extractable": True,
            "preferred_text_source": best_method,
            "needs_ocr": False,
            "reason": "text layer produced usable text",
        }
    if ocr_ok:
        return {
            "text_extractable": True,
            "preferred_text_source": best_method,
            "needs_ocr": True,
            "reason": "image OCR produced text; no native text layer",
        }
    return {
        "text_extractable": False,
        "preferred_text_source": None,
        "needs_ocr": True,
        "reason": "no extraction method produced text",
    }


def run_peer_review_manifest(
    manifest_path: Path,
    output_dir: Path,
    *,
    vlm_base_url: str | None = None,
    vlm_model: str | None = None,
    vlm_api_key: str = "dummy",
    run_paddle: bool = False,
    run_markitdown: bool = True,
    run_pdfplumber: bool = True,
    run_native_text: bool = True,
    max_pages: int | None = 1,
    dpi: int = 200,
    workers: int = 1,
    limit: int | None = None,
    force: bool = False,
    timeout_seconds: int = 60,
    progress_every: int = 0,
) -> dict[str, Any]:
    """Process a JSONL manifest and write peer-review sidecars + index."""
    from gwanbo_ocr.pdf.io import read_jsonl, resolve_pdf_path, write_json_atomic

    vlm_runner = _make_vlm_runner(vlm_base_url, vlm_model, vlm_api_key) if vlm_base_url else None

    output_dir.mkdir(parents=True, exist_ok=True)
    items_dir = output_dir / "items"
    images_dir = output_dir / "images"

    summary: dict[str, Any] = {
        "manifest": str(manifest_path),
        "output_dir": str(output_dir),
        "started_at": _iso_now(),
        "total": 0,
        "processed": 0,
        "skipped_existing": 0,
        "errors": 0,
        "by_best_method": {},
        "by_decision": {},
        "settings": {
            "max_pages": max_pages,
            "dpi": dpi,
            "workers": workers,
            "run_native_text": run_native_text,
            "run_pdfplumber": run_pdfplumber,
            "run_markitdown": run_markitdown,
            "run_paddle": run_paddle,
            "vlm_model": vlm_model,
        },
    }

    work_items: list[dict[str, Any]] = []
    for row in read_jsonl(manifest_path):
        if limit is not None and summary["total"] >= limit:
            break
        summary["total"] += 1
        pdf_path_text = str(resolve_pdf_path(row, peti_root="/root/peti"))
        sample_id = str(row.get("sample_id") or row.get("id") or "unknown")
        sidecar_path = items_dir / f"{sample_id}.json"
        if sidecar_path.exists() and not force:
            summary["skipped_existing"] += 1
            continue
        work_items.append(
            {
                "row": row,
                "sample_id": sample_id,
                "pdf_path": pdf_path_text,
                "sidecar_path": str(sidecar_path),
                "image_dir": str(images_dir / sample_id),
                "vlm_runner": vlm_runner,
                "run_paddle": run_paddle,
                "run_markitdown": run_markitdown,
                "run_pdfplumber": run_pdfplumber,
                "run_native_text": run_native_text,
                "max_pages": max_pages,
                "dpi": dpi,
                "timeout_seconds": timeout_seconds,
            }
        )

    index: dict[str, Any] = {}
    for result in _bounded_process(work_items, workers):
        sample_id = result["sample_id"]
        report = result["report"]
        sidecar_path = Path(result["sidecar_path"])
        write_json_atomic(sidecar_path, report)
        summary["processed"] += 1
        if report.get("status") == "error":
            summary["errors"] += 1
        best = (report.get("review") or {}).get("best_text_method") or "none"
        summary["by_best_method"][best] = summary["by_best_method"].get(best, 0) + 1
        decision_key = (
            "needs_ocr" if (report.get("decision") or {}).get("needs_ocr") else "text_layer"
        )
        summary["by_decision"][decision_key] = summary["by_decision"].get(decision_key, 0) + 1
        index[sample_id] = _compact_index(report, sidecar_path)
        if progress_every and summary["processed"] % progress_every == 0:
            print(f"processed={summary['processed']} errors={summary['errors']}", flush=True)

    summary["completed_at"] = _iso_now()
    write_json_atomic(output_dir / "metadata.json", index)
    write_json_atomic(output_dir / "summary.json", summary)
    return summary


def aggregate_peer_scores(review_dir: Path) -> dict[str, Any]:
    """Aggregate method-level peer scores from peer-review sidecars."""
    from gwanbo_ocr.pdf.io import read_json

    index_path = review_dir / "metadata.json"
    index = read_json(index_path) or {}
    if not isinstance(index, dict) or not index:
        raise ValueError(f"No metadata.json found in {review_dir}")

    method_counts: dict[str, int] = {}
    decision_counts: dict[str, int] = {}
    f1_by_method: dict[str, list[float]] = {}

    for entry in index.values():
        if not isinstance(entry, dict):
            continue
        best = str(entry.get("best_text_method") or "none")
        method_counts[best] = method_counts.get(best, 0) + 1
        needs_ocr = bool((entry.get("decision") or {}).get("needs_ocr"))
        decision_key = "needs_ocr" if needs_ocr else "text_layer"
        decision_counts[decision_key] = decision_counts.get(decision_key, 0) + 1

        sidecar_path = Path(str(entry.get("sidecar_path") or ""))
        if not sidecar_path.is_file():
            continue
        sidecar = read_json(sidecar_path) or {}
        if not isinstance(sidecar, dict):
            continue
        score_payload = sidecar.get("score")
        score = score_payload if isinstance(score_payload, dict) else {}
        for method, metrics in score.items():
            if method in {"reference_tokens", "ranked_by_f1"}:
                continue
            if not isinstance(metrics, dict):
                continue
            value = metrics.get("critical_token_f1")
            try:
                if value is not None:
                    f1_by_method.setdefault(str(method), []).append(float(value))
            except (TypeError, ValueError):
                continue

    avg_f1 = {
        method: round(sum(values) / len(values), 4)
        for method, values in f1_by_method.items()
        if values
    }
    return {
        "total": len(index),
        "by_best_method": method_counts,
        "by_decision": decision_counts,
        "avg_critical_token_f1_by_method": avg_f1,
        "review_dir": str(review_dir),
    }


def _process_work_item(work_item: dict[str, Any]) -> dict[str, Any]:
    row = work_item["row"]
    pdf_path = Path(work_item["pdf_path"])
    metadata = {k: row.get(k) for k in ("title", "category", "agency", "id") if row.get(k)}

    report = analyze_pdf_peer_review(
        pdf_path,
        image_dir=Path(work_item["image_dir"]),
        vlm_runner=work_item.get("vlm_runner"),
        max_pages=work_item.get("max_pages"),
        dpi=int(work_item.get("dpi") or 200),
        timeout_seconds=int(work_item.get("timeout_seconds") or 60),
        run_paddle=bool(work_item.get("run_paddle")),
        run_markitdown=bool(work_item.get("run_markitdown", True)),
        run_pdfplumber=bool(work_item.get("run_pdfplumber", True)),
        run_native_text=bool(work_item.get("run_native_text", True)),
        metadata=metadata or None,
    )
    report.update(
        {
            "sample_id": work_item["sample_id"],
            "source": row.get("source") or row.get("theme"),
            "pdf_key": row.get("pdf_key") or row.get("id"),
        }
    )
    return {
        "sample_id": work_item["sample_id"],
        "sidecar_path": work_item["sidecar_path"],
        "report": report,
    }


def _bounded_process(
    work_items: list[dict[str, Any]],
    workers: int,
) -> Any:
    if workers <= 1:
        for item in work_items:
            yield _process_work_item(item)
        return

    max_pending = max(workers * 2, workers)
    iterator = iter(work_items)
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        pending: set[concurrent.futures.Future[dict[str, Any]]] = set()
        for _ in range(min(max_pending, len(work_items))):
            try:
                item = next(iterator)
            except StopIteration:
                break
            pending.add(executor.submit(_process_work_item, item))

        while pending:
            done, pending = concurrent.futures.wait(
                pending, return_when=concurrent.futures.FIRST_COMPLETED
            )
            for future in done:
                yield future.result()
                try:
                    item = next(iterator)
                except StopIteration:
                    continue
                pending.add(executor.submit(_process_work_item, item))


def _choose_best_method(peers: dict[str, dict[str, Any]]) -> str | None:
    candidates = [
        (name, int(peer.get("text_chars") or 0), METHOD_PREFERENCE.get(name, 0))
        for name, peer in peers.items()
        if peer.get("status") == "ok" and int(peer.get("text_chars") or 0) > 0
    ]
    if not candidates:
        return None
    candidates.sort(key=lambda t: (t[1], t[2]), reverse=True)
    return candidates[0][0]


def _pairwise_similarities(peers: dict[str, dict[str, Any]]) -> dict[str, float]:
    samples = {
        name: _norm_for_sim(str(peer.get("sample_text") or ""))
        for name, peer in peers.items()
        if peer.get("status") == "ok" and peer.get("sample_text")
    }
    result: dict[str, float] = {}
    names = sorted(samples)
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            ratio = difflib.SequenceMatcher(None, samples[left], samples[right]).ratio()
            result[f"{left}:{right}"] = round(ratio, 4)
    return result


def _compact_index(report: dict[str, Any], sidecar_path: Path) -> dict[str, Any]:
    review = report.get("review") or {}
    decision = report.get("decision") or {}
    score = report.get("score") or {}
    return {
        "status": report.get("status"),
        "sample_id": report.get("sample_id"),
        "pdf_key": report.get("pdf_key"),
        "analysis_scope": report.get("analysis_scope"),
        "best_text_method": review.get("best_text_method"),
        "decision": decision,
        "peer_summaries": review.get("peer_summaries"),
        "ranked_by_f1": score.get("ranked_by_f1"),
        "sidecar_path": str(sidecar_path),
        "generated_at": report.get("generated_at"),
        "error": report.get("error"),
    }
```


- [ ] **Step 8: Update strategy.py peer_review import**

In `src/gwanbo_ocr/strategy.py`, inside `run_pipeline()`, change:
```python
    from gwanbo_ocr.peer_review import aggregate_peer_scores, run_peer_review_manifest
```
to:
```python
    from gwanbo_ocr.peers import aggregate_peer_scores, run_peer_review_manifest
```

- [ ] **Step 9: Update cli.py peer_review imports**

In `src/gwanbo_ocr/cli.py`, change (in `peer_run` function):
```python
    from gwanbo_ocr.peer_review import run_peer_review_manifest
```
to:
```python
    from gwanbo_ocr.peers import run_peer_review_manifest
```

And in `peer_score` function:
```python
    from gwanbo_ocr.peer_review import aggregate_peer_scores
```
to:
```python
    from gwanbo_ocr.peers import aggregate_peer_scores
```

- [ ] **Step 10: Update tests/test_peer_review.py import block**

Replace:
```python
from gwanbo_ocr.peer_review import (
    aggregate_peer_scores,
    analyze_pdf_peer_review,
    decide_extraction,
    extract_markitdown,
    extract_native_text,
    extract_pdfplumber,
    extract_vlm_ocr,
    review_extraction_peers,
    run_peer_review_manifest,
    score_against_metadata,
)
```
with:
```python
from gwanbo_ocr.peers import (
    aggregate_peer_scores,
    analyze_pdf_peer_review,
    decide_extraction,
    extract_markitdown,
    extract_native_text,
    extract_pdfplumber,
    extract_vlm_ocr,
    review_extraction_peers,
    run_peer_review_manifest,
    score_against_metadata,
)
```

- [ ] **Step 11: Run tests before deleting peer_review.py**

```bash
pytest -v
```
Expected: all tests pass.

- [ ] **Step 12: Delete peer_review.py and run tests again**

```bash
rm src/gwanbo_ocr/peer_review.py
pytest -v
```
Expected: all tests pass.

- [ ] **Step 13: Commit**

```bash
git add src/gwanbo_ocr/peers/ src/gwanbo_ocr/strategy.py src/gwanbo_ocr/cli.py tests/test_peer_review.py
git rm src/gwanbo_ocr/peer_review.py
git commit -m "refactor: convert peer_review.py to peers/ package"
```

---

## Task 5: Restructure Test Files

**Files:**
- Create: `tests/peers/__init__.py`
- Create: `tests/peers/test_extract.py` (from test_peer_review.py: TestExtractNativeText, TestExtractPdfplumber, TestExtractMarkitdown, TestExtractVlmOcr)
- Create: `tests/peers/test_review.py` (from test_peer_review.py: TestReviewExtractionPeers, TestDecideExtraction, TestScoreAgainstMetadata)
- Create: `tests/peers/test_orchestration.py` (from test_peer_review.py: TestAnalyzePdfPeerReview, TestRunPeerReviewManifest, CLI peer tests, test_aggregate_peer_scores_*)
- Create: `tests/bench/test_report.py` (from test_bench.py: test_summarize_throughput_*, test_format_throughput_report_*)
- Create: `tests/bench/test_run.py` (from test_bench.py: test_resolve_runner_model_*, test_run_benchmark_*)
- Create: `tests/bench/test_score.py` (from test_bench.py: test_score_benchmark_*)
- Delete: `tests/test_peer_review.py`
- Delete: `tests/bench/test_bench.py`

- [ ] **Step 1: Create tests/peers/__init__.py**

```python
```
(empty file)

- [ ] **Step 2: Create tests/peers/test_extract.py**

Header for the new file:
```python
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from gwanbo_ocr.peers import (
    extract_markitdown,
    extract_native_text,
    extract_pdfplumber,
    extract_vlm_ocr,
)
```

Move these test classes from `tests/test_peer_review.py` into this file (keeping all test logic unchanged):
- `TestExtractNativeText` (lines 62–84)
- `TestExtractPdfplumber` (lines 85–107)
- `TestExtractMarkitdown` (lines 108–160)
- `TestExtractVlmOcr` (lines 161–227)

Also move the shared fixture `TEXT_PDF`, `_write_pdf()` helper, and any `import` statements needed by these classes.

- [ ] **Step 3: Create tests/peers/test_review.py**

Header:
```python
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from gwanbo_ocr.peers import (
    decide_extraction,
    review_extraction_peers,
    score_against_metadata,
)
```

Move from `tests/test_peer_review.py`:
- `TestReviewExtractionPeers` (test_chooses_highest_char_count, test_warns_about_failed_peers, etc.)
- `TestDecideExtraction`
- `TestScoreAgainstMetadata`

- [ ] **Step 4: Create tests/peers/test_orchestration.py**

Header:
```python
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from gwanbo_ocr.cli import app as cli_app
from gwanbo_ocr.peers import (
    aggregate_peer_scores,
    analyze_pdf_peer_review,
    run_peer_review_manifest,
)
```

Move from `tests/test_peer_review.py`:
- `TestAnalyzePdfPeerReview`
- `TestRunPeerReviewManifest`
- CLI peer tests (`test_peer_run_help`, `test_peer_score_help`, `test_peer_run_end_to_end`)
- `test_aggregate_peer_scores_reads_sidecar_scores`

Also move `TEXT_PDF` and `_write_pdf()` (or import from a shared conftest if preferred).

- [ ] **Step 5: Create tests/bench/test_report.py**

```python
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from gwanbo_ocr.bench import RunRecord, format_throughput_report, summarize_throughput
```

Move from `tests/bench/test_bench.py`:
- `test_summarize_throughput_uses_wall_elapsed_time`
- `test_format_throughput_report_is_markdown`

- [ ] **Step 6: Create tests/bench/test_run.py**

```python
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any
from urllib import error as url_error

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from gwanbo_ocr.bench import RunRecord, resolve_runner_model, run_benchmark
```

Move from `tests/bench/test_bench.py`:
- `test_resolve_runner_model_uses_model_config_alias`
- All `test_run_benchmark_*` tests (preflight, concurrency, strategy, fallback, etc.)

The monkeypatch targets in these tests were already updated in Task 1 to use `preflight_mod.request`. No further change needed.

- [ ] **Step 7: Create tests/bench/test_score.py**

```python
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from gwanbo_ocr.bench import score_benchmark
```

Move from `tests/bench/test_bench.py`:
- `test_score_benchmark_includes_route_summary`

- [ ] **Step 8: Run full suite before deleting old test files**

```bash
pytest -v
```
Expected: all tests pass (both old and new files exist simultaneously; no conflicts since class/function names don't overlap in the new files).

- [ ] **Step 9: Delete old test files**

```bash
rm tests/test_peer_review.py tests/bench/test_bench.py
```

- [ ] **Step 10: Run full suite after deletion**

```bash
pytest -v
```
Expected: all tests pass.

- [ ] **Step 11: Lint and type check**

```bash
ruff check src tests
mypy src
```
Expected: no errors.

- [ ] **Step 12: Commit**

```bash
git add tests/peers/ tests/bench/test_report.py tests/bench/test_run.py tests/bench/test_score.py
git rm tests/test_peer_review.py tests/bench/test_bench.py
git commit -m "refactor: restructure tests to match new bench/ and peers/ module layout"
```

---

## Verification

After all tasks complete:

```bash
# Full test suite
pytest --tb=short -q

# Lint
ruff check src tests
ruff format --check src tests

# Types
mypy src
```

Expected: all tests green, no ruff errors, no mypy errors. No behavior changes.

Final module structure check:
```bash
find src/gwanbo_ocr/bench src/gwanbo_ocr/peers src/gwanbo_ocr/runners/preflight.py -type f | sort
```
Expected output:
```
src/gwanbo_ocr/bench/__init__.py
src/gwanbo_ocr/bench/report.py
src/gwanbo_ocr/bench/run.py
src/gwanbo_ocr/bench/score.py
src/gwanbo_ocr/peers/__init__.py
src/gwanbo_ocr/peers/_helpers.py
src/gwanbo_ocr/peers/markitdown.py
src/gwanbo_ocr/peers/native.py
src/gwanbo_ocr/peers/paddle.py
src/gwanbo_ocr/peers/pdfplumber.py
src/gwanbo_ocr/peers/vlm.py
src/gwanbo_ocr/runners/preflight.py
```
