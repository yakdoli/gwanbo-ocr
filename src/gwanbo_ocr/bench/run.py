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
    paddle_service_url: str | None = None,
) -> dict[str, Any]:
    from gwanbo_ocr.runners.preflight import preflight_openai_endpoint

    output = Path(run_dir)
    output.mkdir(parents=True, exist_ok=True)
    tasks = _load_suite_tasks(suite)
    if limit is not None:
        tasks = tasks[:limit]

    model_id = resolve_runner_model(runner_name)
    effective_paddle_service_url = paddle_service_url or resolve_paddle_service_url()
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
                paddle_service_url=effective_paddle_service_url,
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
                        paddle_service_url=effective_paddle_service_url,
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
        "paddle_service_url": effective_paddle_service_url,
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
    paddle_service_url: str | None,
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
                result = _transcribe_with_paddle(
                    image_path,
                    page_number=page_number,
                    service_url=paddle_service_url,
                )
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


def resolve_paddle_service_url(
    *,
    config_path: str | Path = "configs/models.yaml",
) -> str | None:
    """Resolve the containerized PaddleOCR service URL from env/config defaults."""
    import os

    env_url = os.getenv("GWANBO_PADDLEOCR_SERVICE_URL")
    if env_url:
        return env_url
    config = Path(config_path)
    if not config.exists():
        return None
    try:
        import yaml

        data = yaml.safe_load(config.read_text(encoding="utf-8")) or {}
    except Exception:  # noqa: BLE001
        return None
    if not isinstance(data, Mapping):
        return None
    ocr = data.get("ocr")
    if isinstance(ocr, Mapping):
        paddleocr = ocr.get("paddleocr")
        if isinstance(paddleocr, Mapping) and paddleocr.get("service_url"):
            return str(paddleocr["service_url"])
    services = data.get("services")
    if isinstance(services, Mapping):
        paddle_api = services.get("paddleocr_api")
        if isinstance(paddle_api, Mapping) and paddle_api.get("base_url"):
            return str(paddle_api["base_url"])
    return None


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


def _transcribe_with_paddle(
    image_path: Any,
    *,
    page_number: int,
    service_url: str | None,
) -> Any:
    if service_url:
        from gwanbo_ocr.runners.paddle_service import PaddleOcrServiceRunner

        service_runner = PaddleOcrServiceRunner(service_url, lang="korean")
        return service_runner.transcribe(Path(str(image_path)), page_number=page_number)

    from gwanbo_ocr.runners.paddle import PaddleOcrRunner

    local_runner = PaddleOcrRunner(lang="korean")
    return local_runner.transcribe(image_path, page_number=page_number)
