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
