from __future__ import annotations

import json
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from gwanbo_ocr.pdf.font_analysis import ResolutionTier
from gwanbo_ocr.prompts import (
    TRANSCRIPTION_JSON_SCHEMA,
    TRANSCRIPTION_SYSTEM_PROMPT,
    build_transcription_prompt,
)

from .report import _count_by, _duration, summarize_throughput, write_records_jsonl

_DEFAULT_TIER_RUNNERS = {
    "HIGH": "chandra_ocr_2_lemonade_vllm",
    "STANDARD": "lightonocr2_1b_lemonade_vllm",
    "LOW": "bizonai_ocr_lemonade_vllm",
}


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
    vlm_max_tokens: int | None = None,
    paddle_service_url: str | None = None,
    paddle_preprocess: bool = False,
    paddle_preprocess_max_chars: int = 4000,
    gold_manifest_path: str | Path | None = None,
) -> dict[str, Any]:
    from gwanbo_ocr.runners.preflight import preflight_openai_endpoint

    output = Path(run_dir)
    output.mkdir(parents=True, exist_ok=True)
    tasks = _load_suite_tasks(suite)
    if gold_manifest_path:
        tasks = _merge_gold_references(tasks, gold_manifest_path)
    if limit is not None:
        tasks = tasks[:limit]

    model_id = resolve_runner_model(runner_name)
    runner_config = resolve_runner_config(runner_name)
    if vlm_max_tokens is not None:
        runner_config = {**runner_config, "max_tokens": vlm_max_tokens}
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
                paddle_preprocess=paddle_preprocess,
                paddle_preprocess_max_chars=paddle_preprocess_max_chars,
                runner_config=runner_config,
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
                        paddle_preprocess=paddle_preprocess,
                        paddle_preprocess_max_chars=paddle_preprocess_max_chars,
                        runner_config=runner_config,
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
        "paddle_preprocess": paddle_preprocess,
        "paddle_preprocess_max_chars": paddle_preprocess_max_chars,
        "concurrency": worker_count,
        "enforce_strategy_routing": enforce_strategy_routing,
        "preflight": preflight,
        "vlm_max_tokens": runner_config.get("max_tokens"),
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


def run_benchmark_tiered(
    *,
    suite: str,
    run_dir: str | Path,
    base_url: str = "http://127.0.0.1:13305/api/v1",
    api_key: str = "dummy",
    concurrency: int = 4,
    limit: int | None = None,
    vlm_max_tokens: int | None = None,
    tier_runner_map: dict[str, str] | None = None,
    **kwargs,
) -> dict[str, Any]:
    output = Path(run_dir)
    output.mkdir(parents=True, exist_ok=True)
    tasks = _load_suite_tasks(suite)
    if limit is not None:
        tasks = tasks[:limit]

    effective_tier_runner_map = dict(_DEFAULT_TIER_RUNNERS)
    for tier, runner_name in (tier_runner_map or {}).items():
        effective_tier_runner_map[_resolution_tier_name(tier)] = str(runner_name)

    runner_models = {
        runner_name: resolve_runner_model(runner_name)
        for runner_name in set(effective_tier_runner_map.values())
    }
    runner_configs = {
        runner_name: resolve_runner_config(runner_name)
        for runner_name in set(effective_tier_runner_map.values())
    }
    if vlm_max_tokens is not None:
        runner_configs = {
            runner_name: {**runner_config, "max_tokens": vlm_max_tokens}
            for runner_name, runner_config in runner_configs.items()
        }

    worker_count = max(1, concurrency)
    if worker_count == 1:
        records = [
            _run_benchmark_tiered_task(
                task,
                base_url=base_url,
                api_key=api_key,
                tier_runner_map=effective_tier_runner_map,
                runner_models=runner_models,
                runner_configs=runner_configs,
            )
            for task in tasks
        ]
    else:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            records = list(
                executor.map(
                    lambda task: _run_benchmark_tiered_task(
                        task,
                        base_url=base_url,
                        api_key=api_key,
                        tier_runner_map=effective_tier_runner_map,
                        runner_models=runner_models,
                        runner_configs=runner_configs,
                    ),
                    tasks,
                )
            )

    results_path = write_records_jsonl(records, output / "results.jsonl")
    summary = {
        "status": "ok",
        "suite": suite,
        "base_url": base_url,
        "concurrency": worker_count,
        "vlm_max_tokens": vlm_max_tokens,
        "tasks": len(tasks),
        "results": str(results_path),
        "throughput": summarize_throughput(records),
        "by_tier": _count_by(records, "resolution_tier"),
        "tier_runner_map": effective_tier_runner_map,
    }
    if kwargs:
        summary["options"] = dict(kwargs)
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    return summary


def _run_benchmark_tiered_task(
    task: Mapping[str, Any],
    *,
    base_url: str,
    api_key: str,
    tier_runner_map: Mapping[str, str],
    runner_models: Mapping[str, str],
    runner_configs: Mapping[str, dict[str, Any]],
) -> dict[str, Any]:
    resolution_tier = _resolution_tier_name(task.get("resolution_tier", "STANDARD"))
    runner_name = tier_runner_map[resolution_tier]
    model_id = runner_models[runner_name]
    runner_config = runner_configs[runner_name]
    image_path = task.get("image_path")
    page_number = int(task.get("page_number") or 1)
    started_at = now_iso()
    record: dict[str, Any] = {
        "item_id": str(task.get("sample_id") or task.get("id") or task.get("pdf_key") or ""),
        "runner": runner_name,
        "model_id": model_id,
        "engine": "vllm-chat",
        "image_path": image_path,
        "resolution_tier": resolution_tier,
        "strategy": str(task.get("strategy") or ""),
        "cluster_id": task.get("cluster_id"),
        "strategy_confidence": task.get("strategy_confidence"),
        "reference_text": task.get("reference_text") or task.get("gold_text") or None,
        "started_at": started_at,
        "pages": 1,
        "bytes_processed": _file_size(image_path),
        "route": "vlm_tiered",
    }
    try:
        if not image_path:
            raise ValueError("task is missing image_path")
        result = _transcribe_with_vllm(
            image_path,
            page_number=page_number,
            model_id=model_id,
            base_url=base_url,
            api_key=api_key,
            runner_config=runner_config,
        )
        response_metadata = _extract_response_metadata(result)
        record.update(
            {
                "status": "ok",
                "ended_at": now_iso(),
                "text": result.text,
                "result": result.to_dict(),
            }
        )
        if response_metadata:
            record["response_metadata"] = response_metadata
            for field in (
                "finish_reason",
                "stop_reason",
                "prompt_tokens",
                "completion_tokens",
                "total_tokens",
            ):
                value = response_metadata.get(field)
                if value is not None:
                    record[field] = value
    except Exception as exc:  # noqa: BLE001
        record.update({"status": "error", "ended_at": now_iso(), "error": str(exc)})
    record["duration_s"] = _duration(record)
    return record


def _resolution_tier_name(value: Any) -> str:
    if isinstance(value, ResolutionTier):
        return value.name
    text = str(value or ResolutionTier.STANDARD.name).strip()
    if not text:
        return ResolutionTier.STANDARD.name
    upper_text = text.upper()
    if upper_text in ResolutionTier.__members__:
        return upper_text
    for tier in ResolutionTier:
        if text.lower() == tier.value:
            return tier.name
    return ResolutionTier.STANDARD.name


def _run_benchmark_task(
    task: Mapping[str, Any],
    *,
    runner_name: str,
    model_id: str,
    base_url: str,
    api_key: str,
    enforce_strategy_routing: bool,
    paddle_service_url: str | None,
    paddle_preprocess: bool = False,
    paddle_preprocess_max_chars: int = 4000,
    runner_config: dict[str, Any] | None = None,
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
        "reference_text": task.get("reference_text") or task.get("gold_text") or None,
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
        paddle_context_text: str | None = None
        if paddle_preprocess and not (enforce_strategy_routing and strategy == "ocr_paddle_simple"):
            preprocess = _build_paddle_preprocess_context(
                image_path,
                page_number=page_number,
                service_url=paddle_service_url,
                max_chars=paddle_preprocess_max_chars,
            )
            record["paddle_preprocess"] = preprocess
            if preprocess.get("status") == "ok":
                paddle_context_text = str(preprocess.get("text") or "")

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
                    paddle_preprocess_text=paddle_context_text,
                    runner_config=runner_config,
                )
        else:
            route = "vlm_primary"
            result = _transcribe_with_vllm(
                image_path,
                page_number=page_number,
                model_id=model_id,
                base_url=base_url,
                api_key=api_key,
                paddle_preprocess_text=paddle_context_text,
                runner_config=runner_config,
            )

        if enforce_strategy_routing and strategy == "peer_review_escalation":
            route = "vlm_escalation"

        response_metadata = _extract_response_metadata(result)
        record.update(
            {
                "status": "ok",
                "ended_at": now_iso(),
                "text": result.text,
                "result": result.to_dict(),
                "route": route,
            }
        )
        if response_metadata:
            record["response_metadata"] = response_metadata
            for field in (
                "finish_reason",
                "stop_reason",
                "prompt_tokens",
                "completion_tokens",
                "total_tokens",
            ):
                value = response_metadata.get(field)
                if value is not None:
                    record[field] = value
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


def resolve_runner_config(
    runner_name: str,
    *,
    config_path: str | Path = "configs/models.yaml",
) -> dict[str, Any]:
    """Read VLM request options from models.yaml for a runner alias."""
    _DEFAULT_TIMEOUT = 120
    _DEFAULT_RETRIES = 2
    _DEFAULT_MAX_TOKENS = 4096
    _DEFAULT_TEMPERATURE = 0.0
    defaults: dict[str, Any] = {
        "timeout_seconds": _DEFAULT_TIMEOUT,
        "max_retries": _DEFAULT_RETRIES,
        "max_tokens": _DEFAULT_MAX_TOKENS,
        "temperature": _DEFAULT_TEMPERATURE,
        "top_p": None,
        "json_mode": True,
        "user_prompt": None,
        "system_prompt": "default",
    }
    config = Path(config_path)
    if not config.exists():
        return defaults
    try:
        import yaml

        data = yaml.safe_load(config.read_text(encoding="utf-8")) or {}
    except Exception:  # noqa: BLE001
        return defaults
    models = data.get("vision_language_models")
    if not isinstance(models, Mapping):
        return defaults
    entry = models.get(runner_name)
    if not isinstance(entry, Mapping):
        return defaults
    temperature_value = entry.get("temperature", _DEFAULT_TEMPERATURE)
    top_p_value = entry.get("top_p")
    return {
        "timeout_seconds": int(entry.get("timeout_seconds") or _DEFAULT_TIMEOUT),
        "max_retries": int(entry.get("max_retries") or _DEFAULT_RETRIES),
        "max_tokens": int(entry.get("max_tokens") or _DEFAULT_MAX_TOKENS),
        "temperature": (
            _DEFAULT_TEMPERATURE if temperature_value is None else float(temperature_value)
        ),
        "top_p": None if top_p_value is None else float(top_p_value),
        "json_mode": bool(entry.get("json_mode", True)),
        "user_prompt": entry.get("user_prompt"),
        "system_prompt": entry.get("system_prompt", "default"),
    }


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
    paddle_preprocess_text: str | None = None,
    runner_config: dict[str, Any] | None = None,
) -> Any:
    from gwanbo_ocr.runners.vllm_chat import VllmChatRunner

    timeout = int((runner_config or {}).get("timeout_seconds") or 120)
    max_retries = int((runner_config or {}).get("max_retries") or 2)
    max_tokens = int((runner_config or {}).get("max_tokens") or 4096)
    temperature_value = (runner_config or {}).get("temperature", 0.0)
    top_p_value = (runner_config or {}).get("top_p")
    temperature = 0.0 if temperature_value is None else float(temperature_value)
    top_p = None if top_p_value is None else float(top_p_value)
    json_mode = bool((runner_config or {}).get("json_mode", True))
    user_prompt = (runner_config or {}).get("user_prompt")
    system_prompt = (runner_config or {}).get("system_prompt", "default")
    schema = TRANSCRIPTION_JSON_SCHEMA if json_mode else None
    if system_prompt == "default":
        system_prompt = TRANSCRIPTION_SYSTEM_PROMPT
    elif system_prompt is not None:
        system_prompt = str(system_prompt)
    effective_user_prompt = str(user_prompt) if user_prompt is not None else None
    if paddle_preprocess_text:
        base_prompt = effective_user_prompt or build_transcription_prompt(
            page_number=page_number,
            language_hint="ko,en",
            schema=schema,
        )
        effective_user_prompt = _append_paddle_preprocess_context(
            base_prompt,
            paddle_preprocess_text,
        )
    runner = VllmChatRunner(
        model=model_id,
        base_url=base_url,
        api_key=api_key,
        timeout=timeout,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        response_format={"type": "json_object"} if json_mode else None,
        strict_json=False,
    )
    del max_retries  # stored for future use when retry logic is added
    return runner.transcribe(
        image_path,
        page_number=page_number,
        language_hint="ko,en",
        schema=schema,
        user_prompt=effective_user_prompt,
        system_prompt=system_prompt,
    )


def _merge_gold_references(
    tasks: list[dict[str, Any]], gold_manifest_path: str | Path
) -> list[dict[str, Any]]:
    gold_path = Path(gold_manifest_path)
    if not gold_path.exists():
        return tasks
    gold: dict[str, str] = {}
    with gold_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            ref = row.get("reference_text") or row.get("gold_text")
            if not ref:
                continue
            key = row.get("sample_id") or row.get("pdf_key")
            if key:
                gold[str(key)] = str(ref)
    updated: list[dict[str, Any]] = []
    for task in tasks:
        if task.get("reference_text"):
            updated.append(task)
            continue
        key = task.get("sample_id") or task.get("pdf_key")
        ref = gold.get(str(key)) if key else None
        if ref:
            task = {**task, "reference_text": ref}
        updated.append(task)
    return updated


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


def _build_paddle_preprocess_context(
    image_path: Any,
    *,
    page_number: int,
    service_url: str | None,
    max_chars: int,
) -> dict[str, Any]:
    started_at = now_iso()
    started = datetime.now(UTC)
    try:
        result = _transcribe_with_paddle(
            image_path,
            page_number=page_number,
            service_url=service_url,
        )
    except Exception as exc:  # noqa: BLE001
        ended_at = now_iso()
        return {
            "status": "error",
            "started_at": started_at,
            "ended_at": ended_at,
            "duration_s": _seconds_since(started),
            "error": str(exc),
        }

    text = result.text or ""
    limited_text = _limit_text(text, max_chars=max_chars)
    ended_at = now_iso()
    return {
        "status": "ok",
        "started_at": started_at,
        "ended_at": ended_at,
        "duration_s": _seconds_since(started),
        "backend": result.backend,
        "chars": len(text),
        "truncated": len(limited_text) < len(text),
        "text": limited_text,
    }


def _append_paddle_preprocess_context(prompt: str, paddle_text: str) -> str:
    return "\n\n".join(
        [
            prompt,
            "PaddleOCR preliminary transcription follows. It may contain recognition errors; "
            "use it only as a reading-order and text clue, and prefer the image when they conflict.",
            paddle_text,
        ]
    )


def _extract_response_metadata(result: Any) -> dict[str, Any]:
    response = getattr(result, "raw_response", None)
    payload = _response_to_mapping(response)
    if not payload:
        return {}

    metadata: dict[str, Any] = {}
    for field in ("id", "model", "object"):
        value = payload.get(field)
        if value is not None:
            metadata[field] = value

    choice = _first_choice(payload)
    if choice:
        finish_reason = choice.get("finish_reason")
        stop_reason = choice.get("stop_reason")
        if finish_reason is not None:
            metadata["finish_reason"] = finish_reason
        if stop_reason is not None:
            metadata["stop_reason"] = stop_reason

    usage = payload.get("usage")
    if isinstance(usage, Mapping):
        usage_payload = {
            key: usage[key]
            for key in ("prompt_tokens", "completion_tokens", "total_tokens")
            if usage.get(key) is not None
        }
        if usage_payload:
            metadata["usage"] = usage_payload
            metadata.update(usage_payload)
    return metadata


def _response_to_mapping(response: Any) -> Mapping[str, Any]:
    if response is None:
        return {}
    if isinstance(response, Mapping):
        return response
    if hasattr(response, "model_dump"):
        dumped = response.model_dump()
        if isinstance(dumped, Mapping):
            return dumped
    return {
        field: getattr(response, field)
        for field in ("id", "model", "object", "choices", "usage")
        if getattr(response, field, None) is not None
    }


def _first_choice(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    choices = payload.get("choices")
    if isinstance(choices, list) and choices:
        choice = choices[0]
        if isinstance(choice, Mapping):
            return choice
    return {}


def _limit_text(text: str, *, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    marker = "..."
    if max_chars <= len(marker):
        return text[:max_chars]
    return text[: max_chars - len(marker)].rstrip() + marker


def _seconds_since(started: datetime) -> float:
    return round((datetime.now(UTC) - started).total_seconds(), 3)
