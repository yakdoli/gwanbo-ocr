"""Document-to-Markdown conversion helpers used by CLI and services."""

from __future__ import annotations

import concurrent.futures
import re
from pathlib import Path
from typing import Any

from gwanbo_ocr.pdf.io import (
    iter_jsonl_records,
    resolve_pdf_path,
    write_json_atomic,
    write_jsonl_atomic,
)
from gwanbo_ocr.services import MarkItDownServiceClient

_MarkItDown: Any
try:
    from markitdown import MarkItDown as _MarkItDown  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - optional dependency
    _MarkItDown = None

_OpenAI: Any
try:
    from openai import OpenAI as _OpenAI  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - optional dependency
    _OpenAI = None


class ConversionError(RuntimeError):
    """Raised when a document cannot be converted."""


def convert_document(
    *,
    input_path: Path,
    output: Path,
    mode: str = "plain",
    service_url: str | None = None,
    llm_base_url: str | None = None,
    llm_model: str | None = None,
    llm_api_key: str = "dummy",
    llm_prompt: str | None = None,
    timeout_seconds: float = 120,
) -> dict[str, Any]:
    """Convert one document to Markdown and write a sidecar metadata file."""
    if mode not in {"plain", "ocr-llm"}:
        raise ConversionError(f"unsupported conversion mode: {mode}")
    source = input_path.expanduser()
    if not source.exists():
        raise ConversionError(f"input file not found: {source}")

    output_path = _document_output_path(source, output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if service_url:
        client = MarkItDownServiceClient(service_url, timeout=timeout_seconds)
        converted = client.convert_path(
            source,
            mode=mode,
            llm_base_url=llm_base_url,
            llm_model=llm_model,
            llm_api_key=llm_api_key,
            llm_prompt=llm_prompt,
        )
        text = _converted_text(converted)
    else:
        converter = _make_converter(
            mode=mode,
            llm_base_url=llm_base_url,
            llm_model=llm_model,
            llm_api_key=llm_api_key,
            llm_prompt=llm_prompt,
        )
        text = _converted_text(converter.convert(str(source)))

    output_path.write_text(text, encoding="utf-8")
    metadata = {
        "schema_version": "gwanbo-conversion/v1",
        "status": "ok",
        "mode": mode,
        "input_path": str(source),
        "output_path": str(output_path),
        "service_url": service_url,
        "llm_base_url": llm_base_url,
        "llm_model": llm_model,
        "text_chars": len(text),
    }
    metadata_path = output_path.with_suffix(".meta.json")
    write_json_atomic(metadata_path, metadata)
    return {**metadata, "metadata_path": str(metadata_path)}


def convert_manifest(
    *,
    manifest_path: Path,
    output_dir: Path,
    mode: str = "plain",
    key: str = "sample_id",
    pdf_path_field: str = "pdf_path",
    workers: int = 1,
    limit: int | None = None,
    skip_errors: bool = True,
    force: bool = False,
    peti_root: Path = Path("/root/peti"),
    service_url: str | None = None,
    llm_base_url: str | None = None,
    llm_model: str | None = None,
    llm_api_key: str = "dummy",
    llm_prompt: str | None = None,
    timeout_seconds: float = 120,
) -> dict[str, Any]:
    """Convert every PDF-like row in a manifest and write a conversion manifest."""
    output_dir.mkdir(parents=True, exist_ok=True)
    result_manifest = output_dir / "manifest.jsonl"

    work: list[tuple[int, dict[str, Any]]] = []
    malformed: list[dict[str, Any]] = []
    for line_number, payload in iter_jsonl_records(manifest_path):
        if limit is not None and len(work) + len(malformed) >= limit:
            break
        if isinstance(payload, dict) and payload.get("schema_version") == "jsonl-error/v1":
            malformed.append({"line_number": line_number, **payload})
            continue
        if isinstance(payload, dict):
            work.append((line_number, payload))
        else:
            malformed.append(
                {
                    "line_number": line_number,
                    "status": "error",
                    "error": "manifest row is not an object",
                }
            )

    def _convert_one(item: tuple[int, dict[str, Any]]) -> dict[str, Any]:
        line_number, row = item
        row_key = output_keys[line_number]
        output_path = output_dir / f"{row_key}.md"
        if output_path.exists() and not force:
            return {
                **row,
                "line_number": line_number,
                "status": "skipped",
                "reason": "output already exists",
                "markdown_path": str(output_path),
            }
        try:
            pdf_path = _manifest_pdf_path(
                row,
                field=pdf_path_field,
                base_dir=manifest_path.parent,
                peti_root=peti_root,
            )
            summary = convert_document(
                input_path=pdf_path,
                output=output_path,
                mode=mode,
                service_url=service_url,
                llm_base_url=llm_base_url,
                llm_model=llm_model,
                llm_api_key=llm_api_key,
                llm_prompt=llm_prompt,
                timeout_seconds=timeout_seconds,
            )
            return {
                **row,
                "line_number": line_number,
                "status": "ok",
                "markdown_path": summary["output_path"],
                "metadata_path": summary["metadata_path"],
                "mode": mode,
                "text_chars": summary["text_chars"],
            }
        except Exception as exc:  # noqa: BLE001
            if not skip_errors:
                raise
            return {
                **row,
                "line_number": line_number,
                "status": "error",
                "error": str(exc),
                "mode": mode,
            }

    output_keys = _unique_output_keys(work, key=key)

    if workers <= 1:
        rows = [_convert_one(item) for item in work]
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            rows = list(executor.map(_convert_one, work))
    rows.extend(malformed)
    rows.sort(key=lambda row: int(row.get("line_number") or 0))

    write_jsonl_atomic(result_manifest, rows)
    converted = sum(1 for row in rows if row.get("status") == "ok")
    skipped = sum(1 for row in rows if row.get("status") == "skipped")
    errors = sum(1 for row in rows if row.get("status") == "error")
    summary = {
        "schema_version": "gwanbo-conversion-manifest/v1",
        "status": "ok" if errors == 0 else "partial",
        "input_manifest": str(manifest_path),
        "output_dir": str(output_dir),
        "manifest_path": str(result_manifest),
        "mode": mode,
        "total_rows": len(rows),
        "converted": converted,
        "skipped": skipped,
        "errors": errors,
        "workers": workers,
    }
    write_json_atomic(output_dir / "summary.json", summary)
    return summary


def _make_converter(
    *,
    mode: str,
    llm_base_url: str | None,
    llm_model: str | None,
    llm_api_key: str,
    llm_prompt: str | None,
) -> Any:
    MarkItDown = _MarkItDown
    if MarkItDown is None:
        raise ConversionError("markitdown is not installed")
    if mode == "plain":
        return MarkItDown(enable_plugins=False)
    if _OpenAI is None:
        raise ConversionError("openai is required for MarkItDown OCR+LLM mode")
    if not llm_model:
        raise ConversionError("--llm-model is required for OCR+LLM mode")
    client_kwargs = {"api_key": llm_api_key}
    if llm_base_url:
        client_kwargs["base_url"] = llm_base_url
    kwargs: dict[str, Any] = {
        "enable_plugins": True,
        "llm_client": _OpenAI(**client_kwargs),
        "llm_model": llm_model,
    }
    if llm_prompt:
        kwargs["llm_prompt"] = llm_prompt
    return MarkItDown(**kwargs)


def _converted_text(converted: Any) -> str:
    if isinstance(converted, dict):
        for key in ("markdown_content", "text_content", "markdown", "text"):
            value = converted.get(key)
            if isinstance(value, str):
                return value
    text = getattr(converted, "text_content", None)
    if isinstance(text, str):
        return text
    return str(text or "")


def _document_output_path(input_path: Path, output: Path) -> Path:
    if output.suffix.lower() == ".md":
        return output
    return output / f"{input_path.stem}.md"


def _manifest_pdf_path(
    row: dict[str, Any],
    *,
    field: str,
    base_dir: Path,
    peti_root: Path,
) -> Path:
    if field != "pdf_path" and row.get(field):
        return Path(str(row[field])).expanduser()
    return resolve_pdf_path(row, base_dir=base_dir, peti_root=peti_root)


def _row_key(row: dict[str, Any], *, key: str, fallback: str) -> str:
    value = str(row.get(key) or row.get("id") or row.get("pdf_key") or fallback)
    cleaned = re.sub(r"[^0-9A-Za-z가-힣._-]+", "_", value).strip("._-")
    return cleaned or fallback


def _unique_output_keys(work: list[tuple[int, dict[str, Any]]], *, key: str) -> dict[int, str]:
    used: dict[str, int] = {}
    result: dict[int, str] = {}
    for line_number, row in work:
        base = _row_key(row, key=key, fallback=f"row_{line_number}")
        count = used.get(base, 0) + 1
        used[base] = count
        result[line_number] = base if count == 1 else f"{base}_{count}"
    return result
