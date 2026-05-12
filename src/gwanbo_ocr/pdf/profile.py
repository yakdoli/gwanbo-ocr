"""Build lightweight PDF feature profiles for layout clustering."""

from __future__ import annotations

import concurrent.futures
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from gwanbo_ocr.pdf.integrity import validate_pdf_integrity
from gwanbo_ocr.pdf.io import (
    iter_jsonl_records,
    pdf_key_from_row,
    read_json,
    resolve_pdf_path,
    write_json_atomic,
    write_jsonl_atomic,
)
from gwanbo_ocr.pdf.layout import analyze_pdf_layout
from gwanbo_ocr.pdf.text import analyze_pdf_text

PROFILE_SCHEMA_VERSION = "pdf-profile/v1"
PROFILE_MANIFEST_SCHEMA_VERSION = "pdf-profile-manifest/v1"
DEFAULT_PROFILE_STRATA = ("theme", "year", "category")


def profile_manifest(
    *,
    manifest_path: str | Path,
    output_dir: str | Path,
    max_pages: int | None = 3,
    workers: int = 1,
    sample_per_bucket: int | None = 20,
    table_strategy: str = "auto",
    timeout_seconds: int = 30,
    limit: int | None = None,
    only_missing: bool = False,
    force: bool = False,
) -> dict[str, Any]:
    """Profile manifest rows and write sidecars plus a compact JSONL manifest."""
    output = Path(output_dir)
    items_dir = output / "items"
    items_dir.mkdir(parents=True, exist_ok=True)

    rows = _load_manifest_rows(manifest_path, limit=limit)
    selected = _sample_rows(rows, sample_per_bucket=sample_per_bucket)
    work_items = []
    compact_rows: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()

    for index, row in enumerate(selected, start=1):
        pdf_key = pdf_key_from_row(row, fallback=f"row-{index}")
        sidecar_path = items_dir / f"{_safe_key(pdf_key)}.json"
        if sidecar_path.exists() and only_missing and not force:
            existing = read_json(sidecar_path)
            if isinstance(existing, dict):
                compact = compact_profile(existing, sidecar_path)
                compact_rows.append(compact)
                counts["skipped_existing"] += 1
                counts[str(compact.get("text_mode") or "unknown")] += 1
            continue
        work_items.append(
            {
                "row": row,
                "pdf_key": pdf_key,
                "sidecar_path": str(sidecar_path),
                "max_pages": max_pages,
                "table_strategy": table_strategy,
                "timeout_seconds": timeout_seconds,
            }
        )

    for result in _process_profile_items(work_items, workers):
        profile = result["profile"]
        sidecar_path = Path(result["sidecar_path"])
        write_json_atomic(sidecar_path, profile)
        compact = compact_profile(profile, sidecar_path)
        compact_rows.append(compact)
        counts["processed"] += 1
        counts[str(compact.get("text_mode") or "unknown")] += 1
        if profile.get("error"):
            counts["errors"] += 1

    compact_rows.sort(key=lambda item: str(item.get("pdf_key") or ""))
    manifest_out = output / "manifest.jsonl"
    write_jsonl_atomic(manifest_out, compact_rows)

    summary = {
        "status": "ok",
        "schema_version": PROFILE_MANIFEST_SCHEMA_VERSION,
        "input": str(manifest_path),
        "output_dir": str(output),
        "manifest": str(manifest_out),
        "total_input_rows": len(rows),
        "selected_rows": len(selected),
        "sample_per_bucket": sample_per_bucket,
        "max_pages": max_pages,
        "workers": workers,
        "table_strategy": table_strategy,
        "counts": dict(sorted(counts.items())),
        "by_theme_year_category": _bucket_summary(selected),
    }
    write_json_atomic(output / "summary.json", summary)
    return summary


def profile_row(
    row: Mapping[str, Any],
    *,
    max_pages: int | None = 3,
    table_strategy: str = "auto",
    timeout_seconds: int = 30,
) -> dict[str, Any]:
    """Return a `pdf-profile/v1` record for one manifest row."""
    pdf_path = resolve_pdf_path(row, peti_root="/root/peti")
    pdf_key = pdf_key_from_row(row, fallback=pdf_path.stem or "unknown")
    profile = _base_profile(row, pdf_key=pdf_key, pdf_path=pdf_path)

    if not _has_pdf_path(row) or not pdf_path.is_file():
        profile.update(
            {
                "pdf_exists": False,
                "integrity_status": "missing",
                "text_mode": "missing",
                "layout_class": _string(row.get("layout_class")) or "missing_pdf",
                "error": "file_missing",
            }
        )
        return profile

    try:
        integrity = validate_pdf_integrity(pdf_path, include_hashes=False, use_reader=False)
        profile["integrity_status"] = str(
            integrity.get("overall_status") or integrity.get("status") or ""
        )
        profile["size_bytes"] = integrity.get("size_bytes") or profile.get("size_bytes")
        if not integrity.get("valid", False):
            profile.update(
                {
                    "text_mode": "invalid",
                    "layout_class": "invalid_pdf",
                    "error": "integrity_failed",
                }
            )
            return profile

        text = analyze_pdf_text(
            pdf_path,
            include_sample=False,
            max_pages=max_pages,
            timeout_seconds=timeout_seconds,
        )
        profile["pages"] = _optional_int(text.get("pages")) or profile.get("pages")
        profile["text_extractable"] = bool(text.get("text_extractable"))
        profile["total_chars"] = _optional_int(text.get("total_chars")) or 0
        if text.get("status") == "error":
            profile.update({"text_mode": "error", "error": text.get("error") or "text_error"})
            return profile

        if not profile["text_extractable"]:
            profile.update({"text_mode": "image", "layout_class": "image_or_unextractable_pdf"})
            return profile

        profile["text_mode"] = "text"
        layout = analyze_pdf_layout(
            pdf_path,
            max_pages=max_pages,
            timeout_seconds=timeout_seconds,
            table_strategy=table_strategy,
        )
        _apply_layout(profile, layout)
        if layout.get("status") == "error":
            profile["error"] = layout.get("error")
    except Exception as exc:  # noqa: BLE001
        profile.update({"text_mode": "error", "error": str(exc)})

    return profile


def compact_profile(
    profile: Mapping[str, Any], sidecar_path: str | Path | None = None
) -> dict[str, Any]:
    """Return the profile fields used by clustering and downstream CLIs."""
    compact = {
        "schema_version": PROFILE_SCHEMA_VERSION,
        "pdf_key": profile.get("pdf_key"),
        "id": profile.get("id"),
        "theme": profile.get("theme"),
        "year": profile.get("year"),
        "category": profile.get("category"),
        "agency": profile.get("agency"),
        "pdf_path": profile.get("pdf_path"),
        "pdf_abs_path": profile.get("pdf_abs_path"),
        "pdf_exists": profile.get("pdf_exists"),
        "size_bytes": profile.get("size_bytes"),
        "pages": profile.get("pages"),
        "integrity_status": profile.get("integrity_status"),
        "text_extractable": profile.get("text_extractable"),
        "text_mode": profile.get("text_mode"),
        "total_chars": profile.get("total_chars"),
        "layout_class": profile.get("layout_class"),
        "table_count": profile.get("table_count"),
        "table_text_ratio": profile.get("table_text_ratio"),
        "form_score": profile.get("form_score"),
        "text_quality": profile.get("text_quality"),
        "error": profile.get("error"),
    }
    if sidecar_path is not None:
        compact["sidecar_path"] = str(sidecar_path)
    return compact


def _process_profile_items(
    work_items: list[dict[str, Any]], workers: int
) -> Iterable[dict[str, Any]]:
    if workers <= 1:
        for item in work_items:
            yield _process_profile_item(item)
        return

    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        yield from executor.map(_process_profile_item, work_items)


def _process_profile_item(item: dict[str, Any]) -> dict[str, Any]:
    profile = profile_row(
        item["row"],
        max_pages=item["max_pages"],
        table_strategy=item["table_strategy"],
        timeout_seconds=item["timeout_seconds"],
    )
    return {"profile": profile, "sidecar_path": item["sidecar_path"]}


def _load_manifest_rows(path: str | Path, *, limit: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, payload in iter_jsonl_records(path):
        if limit is not None and len(rows) >= limit:
            break
        if isinstance(payload, dict):
            if payload.get("schema_version") == "jsonl-error/v1":
                rows.append(
                    {
                        "schema_version": "manifest-row-error/v1",
                        "line_number": line_number,
                        "error": payload.get("error"),
                    }
                )
            else:
                rows.append(payload)
    return rows


def _sample_rows(
    rows: Sequence[dict[str, Any]],
    *,
    sample_per_bucket: int | None,
) -> list[dict[str, Any]]:
    if sample_per_bucket is None or sample_per_bucket <= 0:
        return list(rows)

    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(_metadata_bucket(row), []).append(row)

    selected: list[dict[str, Any]] = []
    for key in sorted(groups):
        group = sorted(groups[key], key=lambda row: pdf_key_from_row(row))
        selected.extend(group[:sample_per_bucket])
    return selected


def _metadata_bucket(row: Mapping[str, Any]) -> tuple[str, str, str]:
    return (
        _string(row.get("theme") or row.get("source")) or "unknown",
        _year(row),
        _string(row.get("category")) or "uncategorized",
    )


def _bucket_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        counts["|".join(_metadata_bucket(row))] += 1
    return dict(sorted(counts.items()))


def _base_profile(row: Mapping[str, Any], *, pdf_key: str, pdf_path: Path) -> dict[str, Any]:
    pages = _optional_int(row.get("pages"))
    size_bytes = _optional_int(row.get("size_bytes"))
    text_extractable = row.get("text_extractable")
    text_extractable_value = text_extractable if isinstance(text_extractable, bool) else None
    return {
        "schema_version": PROFILE_SCHEMA_VERSION,
        "pdf_key": pdf_key,
        "id": row.get("id"),
        "theme": row.get("theme") or row.get("source"),
        "year": _year(row),
        "category": row.get("category"),
        "agency": row.get("agency"),
        "pdf_path": row.get("pdf_path") or str(pdf_path),
        "pdf_abs_path": str(pdf_path) if pdf_path else row.get("pdf_abs_path"),
        "pdf_exists": pdf_path.is_file() if pdf_path else bool(row.get("pdf_exists")),
        "size_bytes": size_bytes,
        "pages": pages,
        "integrity_status": "",
        "text_extractable": text_extractable_value,
        "text_mode": "unknown",
        "total_chars": _optional_int(row.get("total_chars")) or 0,
        "layout_class": _string(row.get("layout_class")) or "unknown_text",
        "table_count": _optional_int(row.get("table_count")) or 0,
        "table_text_ratio": 0.0,
        "form_score": 0.0,
        "text_quality": "",
        "error": None,
        "source_row": dict(row),
    }


def _apply_layout(profile: dict[str, Any], layout: Mapping[str, Any]) -> None:
    layout_payload = layout.get("layout")
    layout_dict = layout_payload if isinstance(layout_payload, Mapping) else {}
    metrics_payload = layout_dict.get("metrics")
    metrics = metrics_payload if isinstance(metrics_payload, Mapping) else {}
    profile["layout_class"] = _string(layout_dict.get("document_class")) or "unknown_text"
    profile["table_count"] = _optional_int(metrics.get("table_count")) or len(
        layout.get("tables") or []
    )
    profile["table_text_ratio"] = _optional_float(metrics.get("table_text_ratio")) or 0.0
    profile["form_score"] = _optional_float(metrics.get("form_score")) or 0.0
    profile["text_quality"] = _string(metrics.get("text_quality"))


def _safe_key(pdf_key: str) -> str:
    return pdf_key.replace("/", "__").replace("\\", "__") or "unknown"


def _has_pdf_path(row: Mapping[str, Any]) -> bool:
    return any(str(row.get(key) or "").strip() for key in ("pdf_abs_path", "pdf_path", "path"))


def _year(row: Mapping[str, Any]) -> str:
    year = _string(row.get("year"))
    if year:
        return year
    return _string(row.get("date"))[:4] or "unknown"


def _string(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _optional_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
