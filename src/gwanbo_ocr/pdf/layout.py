"""PDF layout classification and table extraction helpers."""

from __future__ import annotations

import contextlib
import json
import re
import signal
import threading
from collections import Counter
from collections.abc import Iterable, Iterator
from datetime import datetime
from pathlib import Path
from typing import Any

from .io import (
    iter_pdf_paths,
    limited,
    normalize_text,
    pdf_key_from_path,
    read_json,
    read_jsonl,
    resolve_path,
    resolve_pdf_path,
    write_json_atomic,
    write_jsonl_atomic,
)

try:
    import pdfplumber  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - default in the kata environment.
    pdfplumber = None  # type: ignore[assignment]


LAYOUT_CLASSES = {
    "table_heavy",
    "table_with_body",
    "multi_column_text",
    "body_text",
    "form_like",
    "sparse_text",
    "unknown_text",
    "error",
}


def analyze_pdf_layout(
    pdf_path: Path | str,
    *,
    max_pages: int | None = None,
    timeout_seconds: int = 30,
    table_strategy: str = "auto",
    table_settings: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Classify PDF text layout and extract tables when pdfplumber exists."""
    path = Path(pdf_path)
    result: dict[str, Any] = {
        "path": str(path),
        "filename": path.name,
        "status": "ok",
        "text_extractable": False,
        "extraction_method": "pdfplumber",
        "generated_at": iso_now(),
        "layout": empty_layout("unknown_text"),
        "tables": [],
    }

    try:
        result["size_bytes"] = path.stat().st_size
    except OSError as exc:
        return error_result(result, str(exc))

    if pdfplumber is None:
        return error_result(result, "pdfplumber is not installed")

    with _alarm_timeout(timeout_seconds, "PDF layout analysis timed out"):
        try:
            with pdfplumber.open(str(path)) as pdf:
                page_count = len(pdf.pages)
                result["pages"] = page_count
                pages_to_scan = page_count if max_pages is None else min(page_count, max_pages)
                page_metrics: list[dict[str, Any]] = []
                tables: list[dict[str, Any]] = []
                page_errors: list[dict[str, Any]] = []

                for page_index, page in enumerate(pdf.pages[:pages_to_scan]):
                    try:
                        metric, page_tables = analyze_page_layout(
                            page,
                            page_index,
                            table_strategy=table_strategy,
                            table_settings=table_settings,
                        )
                    except Exception as exc:  # noqa: BLE001
                        page_errors.append({"page_index": page_index, "error": str(exc)})
                        continue
                    page_metrics.append(metric)
                    tables.extend(page_tables)

                result["scanned_pages"] = pages_to_scan
                result["text_extractable"] = sum_metric(page_metrics, "text_chars") > 0
                result["layout"] = classify_layout(page_metrics, tables)
                result["tables"] = tables
                if page_errors:
                    result["page_errors"] = page_errors
                    result["page_error_count"] = len(page_errors)
        except Exception as exc:  # noqa: BLE001
            return error_result(result, str(exc))

    return result


def analyze_page_layout(
    page: Any,
    page_index: int,
    *,
    table_strategy: str = "auto",
    table_settings: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    raw_text = safe_call(page.extract_text) or ""
    text = normalize_text(raw_text)
    words = safe_call(page.extract_words) or []
    if not isinstance(words, list):
        words = []

    page_tables = extract_page_tables(page, page_index, table_strategy, table_settings)
    table_chars = sum(int(table.get("text_chars") or 0) for table in page_tables)
    text_chars = len(text)
    metric: dict[str, Any] = {
        "page_index": page_index,
        "width": number_or_none(getattr(page, "width", None)),
        "height": number_or_none(getattr(page, "height", None)),
        "text_chars": text_chars,
        "word_count": len(words),
        "line_count": count_lines(raw_text),
        "table_count": len(page_tables),
        "table_chars": table_chars,
        "table_text_ratio": ratio(table_chars, text_chars),
        "estimated_columns": estimate_columns(words),
        "text_quality": classify_text_quality(text),
        "form_score": estimate_form_score(raw_text),
    }
    return metric, page_tables


def extract_page_tables(
    page: Any,
    page_index: int,
    table_strategy: str = "auto",
    table_settings: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    tables: list[dict[str, Any]] = []
    strategy_profiles = table_strategy_profiles(table_strategy, table_settings)
    if hasattr(page, "find_tables"):
        for strategy_name, settings in strategy_profiles:
            found_tables = (
                page.find_tables(table_settings=settings) if settings else page.find_tables()
            )
            for table in found_tables or []:
                rows = safe_call(table.extract) or []
                tables.append(
                    table_rows_to_json(
                        rows, page_index, 0, getattr(table, "bbox", None), strategy_name
                    )
                )
        return renumber_tables(dedupe_tables(tables), page_index)

    if hasattr(page, "extract_tables"):
        for strategy_name, settings in strategy_profiles:
            found_rows = (
                page.extract_tables(table_settings=settings) if settings else page.extract_tables()
            )
            for rows in found_rows or []:
                tables.append(table_rows_to_json(rows, page_index, 0, None, strategy_name))
    return renumber_tables(dedupe_tables(tables), page_index)


def table_rows_to_json(
    rows: Iterable[Iterable[Any]],
    page_index: int = 0,
    table_index: int = 0,
    bbox: Iterable[Any] | None = None,
    extraction_strategy: str | None = None,
) -> dict[str, Any]:
    """Convert raw table rows into stable column-keyed records."""
    raw_rows = clean_table_rows(rows)
    column_count = max((len(row) for row in raw_rows), default=0)
    padded_rows = [pad_row(row, column_count) for row in raw_rows]
    header = padded_rows[0] if padded_rows else []
    data_rows = padded_rows[1:] if len(padded_rows) > 1 else []
    columns = [
        {"key": f"col_{index + 1}", "label": header[index] if index < len(header) else ""}
        for index in range(column_count)
    ]
    records = [
        {
            column["key"]: row[index] if index < len(row) else ""
            for index, column in enumerate(columns)
        }
        for row in data_rows
    ]
    text_chars = table_text_chars(padded_rows)
    cell_count = sum(len(row) for row in padded_rows)
    nonempty_cells = sum(1 for row in padded_rows for cell in row if cell)
    return {
        "table_id": f"p{page_index + 1:03d}-t{table_index + 1:03d}",
        "page_index": page_index,
        "extraction_strategy": extraction_strategy or "unknown",
        "bbox": normalize_bbox(bbox),
        "row_count": len(padded_rows),
        "column_count": column_count,
        "nonempty_cell_count": nonempty_cells,
        "cell_density": round(nonempty_cells / cell_count, 4) if cell_count else 0.0,
        "columns": columns,
        "records": records,
        "raw_rows": padded_rows,
        "text_chars": text_chars,
    }


def normalize_table_rows(rows: Iterable[Iterable[Any]]) -> list[list[str]]:
    return clean_table_rows(rows)


def table_strategy_profiles(
    table_strategy: str,
    table_settings: dict[str, Any] | None = None,
) -> list[tuple[str, dict[str, Any] | None]]:
    """Return pdfplumber table settings inspired by line/text parser families."""
    if table_settings is not None:
        return [("custom", table_settings)]
    if table_strategy == "lines":
        return [("lines", table_settings_for_strategy("lines"))]
    if table_strategy == "lines-strict":
        return [("lines_strict", table_settings_for_strategy("lines_strict"))]
    if table_strategy == "text":
        return [("text", table_settings_for_strategy("text"))]
    if table_strategy != "auto":
        raise ValueError(f"unsupported table_strategy: {table_strategy}")
    return [
        ("lines", table_settings_for_strategy("lines")),
        ("lines_strict", table_settings_for_strategy("lines_strict")),
        ("text", table_settings_for_strategy("text")),
    ]


def table_settings_for_strategy(strategy: str) -> dict[str, Any]:
    settings: dict[str, Any] = {
        "vertical_strategy": strategy,
        "horizontal_strategy": strategy,
        "snap_tolerance": 3,
        "join_tolerance": 3,
        "intersection_tolerance": 3,
    }
    if strategy == "text":
        settings.update(
            {
                "min_words_vertical": 3,
                "min_words_horizontal": 1,
                "text_tolerance": 3,
            }
        )
    return settings


def dedupe_tables(tables: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for table in tables:
        if not is_valid_table_candidate(table):
            continue
        duplicate_index = find_duplicate_table(selected, table)
        if duplicate_index is None:
            selected.append(table)
            continue
        existing = selected[duplicate_index]
        strategies = sorted(
            {
                str(existing.get("extraction_strategy") or ""),
                str(table.get("extraction_strategy") or ""),
                *[str(value) for value in existing.get("alternate_strategies", [])],
                *[str(value) for value in table.get("alternate_strategies", [])],
            }
            - {""}
        )
        winner = table if table_quality(table) > table_quality(existing) else existing
        winner["alternate_strategies"] = strategies
        selected[duplicate_index] = winner
    return selected


def is_valid_table_candidate(table: dict[str, Any]) -> bool:
    row_count = int(table.get("row_count") or 0)
    column_count = int(table.get("column_count") or 0)
    if row_count < 2 or column_count < 2:
        return False
    if table_nonempty_cell_count(table) < 2:
        return False
    if str(table.get("extraction_strategy") or "") == "text" and table_cell_density(table) < 0.35:
        return False
    return True


def table_cell_density(table: dict[str, Any]) -> float:
    rows = table.get("raw_rows")
    if not isinstance(rows, list):
        return 0.0
    cell_count = sum(len(row) for row in rows if isinstance(row, list))
    return table_nonempty_cell_count(table) / cell_count if cell_count else 0.0


def table_nonempty_cell_count(table: dict[str, Any]) -> int:
    rows = table.get("raw_rows")
    if not isinstance(rows, list):
        return 0
    return sum(1 for row in rows if isinstance(row, list) for cell in row if cell)


def find_duplicate_table(tables: list[dict[str, Any]], candidate: dict[str, Any]) -> int | None:
    for index, existing in enumerate(tables):
        if table_iou(existing.get("bbox"), candidate.get("bbox")) >= 0.9:
            return index
        if table_signature(existing) and table_signature(existing) == table_signature(candidate):
            return index
    return None


def table_quality(table: dict[str, Any]) -> int:
    return (
        int(table.get("text_chars") or 0)
        + int(table.get("row_count") or 0) * 10
        + int(table.get("column_count") or 0) * 5
    )


def table_signature(table: dict[str, Any]) -> str:
    rows = table.get("raw_rows")
    if not isinstance(rows, list):
        return ""
    return json.dumps(rows[:3], ensure_ascii=False, sort_keys=True)


def table_iou(first: Any, second: Any) -> float:
    if (
        not isinstance(first, list)
        or not isinstance(second, list)
        or len(first) != 4
        or len(second) != 4
    ):
        return 0.0
    x1 = max(float(first[0]), float(second[0]))
    y1 = max(float(first[1]), float(second[1]))
    x2 = min(float(first[2]), float(second[2]))
    y2 = min(float(first[3]), float(second[3]))
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if intersection <= 0:
        return 0.0
    first_area = max(0.0, float(first[2]) - float(first[0])) * max(
        0.0, float(first[3]) - float(first[1])
    )
    second_area = max(0.0, float(second[2]) - float(second[0])) * max(
        0.0, float(second[3]) - float(second[1])
    )
    union = first_area + second_area - intersection
    return intersection / union if union else 0.0


def renumber_tables(tables: list[dict[str, Any]], page_index: int) -> list[dict[str, Any]]:
    for table_index, table in enumerate(tables):
        table["table_id"] = f"p{page_index + 1:03d}-t{table_index + 1:03d}"
    return tables


def classify_layout(
    page_metrics: list[dict[str, Any]], tables: list[dict[str, Any]]
) -> dict[str, Any]:
    """Classify document layout from page metrics and extracted tables."""
    if not page_metrics:
        return empty_layout("unknown_text")

    pages_scanned = len(page_metrics)
    total_chars = sum_metric(page_metrics, "text_chars")
    table_chars = sum_metric(page_metrics, "table_chars")
    table_count = len(tables)
    line_count = sum_metric(page_metrics, "line_count")
    word_count = sum_metric(page_metrics, "word_count")
    estimated_columns = max(int(metric.get("estimated_columns") or 0) for metric in page_metrics)
    table_text_ratio = ratio(table_chars, total_chars)
    form_score = ratio(
        sum(float(metric.get("form_score") or 0.0) for metric in page_metrics), pages_scanned
    )
    text_quality = aggregate_text_quality(page_metrics)
    page_classes = [classify_page(metric) for metric in page_metrics]
    table_strategies = sorted(
        {str(table.get("extraction_strategy") or "unknown") for table in tables}
    )

    document_class = "unknown_text"
    confidence = 0.5
    if total_chars < 30:
        document_class = "sparse_text"
        confidence = 0.75
    elif table_count and table_text_ratio >= 0.5:
        document_class = "table_heavy"
        confidence = min(0.95, 0.68 + table_text_ratio * 0.27)
    elif table_count:
        document_class = "table_with_body"
        confidence = 0.72
    elif form_score >= 0.35:
        document_class = "form_like"
        confidence = min(0.9, 0.65 + form_score * 0.25)
    elif estimated_columns >= 2:
        document_class = "multi_column_text"
        confidence = 0.7
    elif total_chars > 0:
        document_class = "body_text"
        confidence = 0.68

    return {
        "document_class": document_class,
        "confidence": round(confidence, 2),
        "metrics": {
            "pages_scanned": pages_scanned,
            "table_count": table_count,
            "table_text_ratio": round(table_text_ratio, 4),
            "estimated_columns": estimated_columns,
            "text_quality": text_quality,
            "total_chars": total_chars,
            "table_chars": table_chars,
            "line_count": line_count,
            "word_count": word_count,
            "form_score": round(form_score, 4),
            "table_strategies": table_strategies,
        },
        "page_classes": page_classes,
    }


def generate_layout_sidecars(
    pdf_dir: Path | str,
    output_dir: Path | str,
    *,
    limit: int | None = None,
    max_pages: int | None = 3,
    table_strategy: str = "auto",
    timeout_seconds: int = 30,
) -> dict[str, Any]:
    """Analyze PDFs and write layout sidecars without reading item JSON."""
    root = Path(pdf_dir)
    output = Path(output_dir)
    item_output = output / "items"
    item_output.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {
        "pdf_dir": str(root),
        "output_dir": str(output),
        "started_at": iso_now(),
        "processed": 0,
        "errors": 0,
        "tables": 0,
        "table_strategy": table_strategy,
    }
    index: dict[str, dict[str, Any]] = {}
    layout_counts: Counter[str] = Counter()

    for pdf_path in limited(iter_pdf_paths(root), limit):
        key = pdf_key_from_path(pdf_path, root)
        metadata = analyze_pdf_layout(
            pdf_path,
            max_pages=max_pages,
            table_strategy=table_strategy,
            timeout_seconds=timeout_seconds,
        )
        metadata["pdf_key"] = key
        metadata["pdf_path"] = str(pdf_path)
        sidecar_path = item_output / f"{key}.json"
        write_json_atomic(sidecar_path, metadata)
        compact = compact_index_metadata(metadata, sidecar_path)
        index[key] = compact
        summary["processed"] += 1
        summary["tables"] += len(metadata.get("tables") or [])
        if metadata.get("status") == "error":
            summary["errors"] += 1
        layout_counts[
            str((metadata.get("layout") or {}).get("document_class") or "unknown_text")
        ] += 1

    summary["by_layout_class"] = dict(sorted(layout_counts.items()))
    summary["completed_at"] = iso_now()
    write_json_atomic(output / "metadata.json", index)
    write_json_atomic(output / "summary.json", summary)
    return summary


def generate_layout_manifest(
    *,
    classification_manifest: Path | str,
    output_dir: Path | str,
    max_pages: int | None = 3,
    table_strategy: str = "auto",
    workers: int = 1,
    only_missing: bool = False,
    force: bool = False,
    limit: int | None = None,
) -> dict[str, Any]:
    """Generate layout sidecars for rows marked layout-eligible."""
    del workers
    output = Path(output_dir)
    items_dir = output / "items"
    items_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {
        "status": "ok",
        "classification_manifest": str(classification_manifest),
        "output_dir": str(output),
        "processed": 0,
        "eligible": 0,
        "skipped_not_eligible": 0,
        "skipped_existing": 0,
        "errors": 0,
        "tables": 0,
        "table_strategy": table_strategy,
    }
    layout_counts: Counter[str] = Counter()

    for index, row in enumerate(read_jsonl(classification_manifest), start=1):
        if limit is not None and index > limit:
            break
        if not isinstance(row, dict):
            continue
        if not bool(row.get("layout_eligible") or row.get("text_extractable")):
            summary["skipped_not_eligible"] += 1
            continue
        summary["eligible"] += 1
        pdf_key = str(
            row.get("pdf_key") or row.get("id") or Path(str(row.get("pdf_path") or "unknown")).stem
        )
        sidecar_path = items_dir / f"{pdf_key}.json"
        if sidecar_path.exists() and only_missing and not force:
            summary["skipped_existing"] += 1
            existing = read_json(sidecar_path)
            if isinstance(existing, dict):
                rows.append(compact_index_metadata(existing, sidecar_path))
            continue
        metadata = analyze_pdf_layout(
            resolve_pdf_path(row, peti_root="/root/peti"),
            max_pages=max_pages,
            table_strategy=table_strategy,
        )
        metadata.update(
            {
                "schema_version": "text-pdf-layout/v1",
                "pdf_key": pdf_key,
                "classification_sidecar": row.get("sidecar_path"),
                "pdf_path": row.get("pdf_path"),
            }
        )
        write_json_atomic(sidecar_path, metadata)
        compact = compact_index_metadata(metadata, sidecar_path)
        rows.append(compact)
        summary["processed"] += 1
        summary["tables"] += len(metadata.get("tables") or [])
        if metadata.get("status") == "error":
            summary["errors"] += 1
        layout_counts[
            str((metadata.get("layout") or {}).get("document_class") or "unknown_text")
        ] += 1

    summary["by_layout_class"] = dict(sorted(layout_counts.items()))
    manifest_path = output / "manifest.jsonl"
    write_jsonl_atomic(manifest_path, rows)
    summary["manifest"] = str(manifest_path)
    write_json_atomic(output / "summary.json", summary)
    return summary


def load_layout_sidecar(
    path_text: str | Path, base_dir: Path | str | None = None
) -> dict[str, Any] | None:
    """Resolve and read a layout sidecar JSON file."""
    payload = read_json(resolve_path(path_text, base_dir))
    return payload if isinstance(payload, dict) else None


def compact_index_metadata(
    metadata: dict[str, Any], sidecar_path: Path | str | None = None
) -> dict[str, Any]:
    layout = metadata.get("layout") if isinstance(metadata.get("layout"), dict) else {}
    compact = {
        "status": metadata.get("status"),
        "pdf_key": metadata.get("pdf_key"),
        "pdf_path": metadata.get("pdf_path"),
        "text_extractable": metadata.get("text_extractable"),
        "layout": layout,
        "table_count": len(metadata.get("tables") or []),
        "generated_at": metadata.get("generated_at"),
        "error": metadata.get("error"),
    }
    if sidecar_path is not None:
        compact["sidecar_path"] = str(sidecar_path)
    return compact


def clean_table_rows(rows: Iterable[Iterable[Any]]) -> list[list[str]]:
    clean_rows: list[list[str]] = []
    for row in rows or []:
        if row is None:
            row = []
        cells = [normalize_cell(cell) for cell in row]
        if any(cells):
            clean_rows.append(cells)
    return clean_rows


def normalize_cell(value: Any) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def pad_row(row: list[str], column_count: int) -> list[str]:
    if len(row) >= column_count:
        return row[:column_count]
    return [*row, *([""] * (column_count - len(row)))]


def normalize_bbox(bbox: Iterable[Any] | None) -> list[float] | None:
    if bbox is None:
        return None
    normalized: list[float] = []
    for value in bbox:
        number = number_or_none(value)
        if number is None:
            return None
        normalized.append(round(number, 2))
    return normalized


def table_text_chars(rows: list[list[str]]) -> int:
    return sum(len(cell) for row in rows for cell in row)


def classify_page(metric: dict[str, Any]) -> dict[str, Any]:
    text_chars = int(metric.get("text_chars") or 0)
    table_count = int(metric.get("table_count") or 0)
    table_text_ratio = float(metric.get("table_text_ratio") or 0.0)
    if text_chars < 30:
        page_class = "sparse_text"
    elif table_count and table_text_ratio >= 0.5:
        page_class = "table_heavy"
    elif table_count:
        page_class = "table_with_body"
    elif float(metric.get("form_score") or 0.0) >= 0.35:
        page_class = "form_like"
    elif int(metric.get("estimated_columns") or 0) >= 2:
        page_class = "multi_column_text"
    else:
        page_class = "body_text"
    return {
        "page_index": metric.get("page_index"),
        "page_class": page_class,
        "table_count": table_count,
        "text_chars": text_chars,
        "table_text_ratio": round(table_text_ratio, 4),
        "estimated_columns": metric.get("estimated_columns"),
        "text_quality": metric.get("text_quality"),
    }


def classify_text_quality(text: str) -> str:
    compact = re.sub(r"\s+", "", text)
    if len(compact) < 20:
        return "empty_or_sparse"
    if "(cid:" in compact.lower():
        return "suspect_or_encoded"
    suspicious = sum(1 for char in compact if is_suspicious_char(char))
    hangul = sum(1 for char in compact if "\uac00" <= char <= "\ud7a3")
    ascii_alnum = sum(1 for char in compact if char.isascii() and char.isalnum())
    suspicious_ratio = suspicious / len(compact)
    readable_ratio = (hangul + ascii_alnum) / len(compact)
    if suspicious_ratio >= 0.2 and readable_ratio < 0.55:
        return "suspect_or_encoded"
    return "readable"


def is_suspicious_char(char: str) -> bool:
    if char.isascii():
        return False
    if "\uac00" <= char <= "\ud7a3":
        return False
    if "\u1100" <= char <= "\u11ff" or "\u3130" <= char <= "\u318f":
        return False
    if "\u4e00" <= char <= "\u9fff":
        return False
    if char in "·ㆍ℃㎞㎡㎥％":
        return False
    return not char.isspace()


def aggregate_text_quality(page_metrics: list[dict[str, Any]]) -> str:
    qualities = [str(metric.get("text_quality") or "unknown") for metric in page_metrics]
    if any(quality == "suspect_or_encoded" for quality in qualities):
        return "suspect_or_encoded"
    if qualities and all(quality == "empty_or_sparse" for quality in qualities):
        return "empty_or_sparse"
    if any(quality == "readable" for quality in qualities):
        return "readable"
    return "unknown"


def estimate_columns(words: list[Any]) -> int:
    starts: list[float] = []
    for word in words:
        if not isinstance(word, dict):
            continue
        x0 = number_or_none(word.get("x0"))
        if x0 is not None:
            starts.append(x0)
    if len(starts) < 8:
        return 1 if starts else 0
    starts.sort()
    clusters: list[list[float]] = []
    for value in starts:
        if not clusters or value - clusters[-1][-1] > 80:
            clusters.append([value])
        else:
            clusters[-1].append(value)
    meaningful = [cluster for cluster in clusters if len(cluster) >= 4]
    return max(1, min(4, len(meaningful) or 1))


def estimate_form_score(text: str) -> float:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return 0.0
    form_lines = 0
    for line in lines:
        if re.search(r"[:：]\s*\S+", line) or re.search(r"^[0-9가-힣A-Za-z][.)]\s+", line):
            form_lines += 1
        elif " - " in line or "ㆍ" in line or "·" in line:
            form_lines += 1
    return form_lines / len(lines)


def empty_layout(document_class: str) -> dict[str, Any]:
    return {
        "document_class": document_class,
        "confidence": 0.0,
        "metrics": {
            "pages_scanned": 0,
            "table_count": 0,
            "table_text_ratio": 0.0,
            "estimated_columns": 0,
            "text_quality": "unknown",
        },
        "page_classes": [],
    }


def error_result(result: dict[str, Any], error: str) -> dict[str, Any]:
    result.update({"status": "error", "error": error, "text_extractable": False})
    result["layout"] = empty_layout("error")
    return result


def count_lines(text: str) -> int:
    return len([line for line in text.splitlines() if line.strip()])


def ratio(numerator: float, denominator: float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def sum_metric(page_metrics: list[dict[str, Any]], key: str) -> int:
    return sum(int(metric.get(key) or 0) for metric in page_metrics)


def number_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def safe_call(func: Any, *args: Any, **kwargs: Any) -> Any:
    try:
        return func(*args, **kwargs)
    except Exception:  # noqa: BLE001
        return None


def iso_now() -> str:
    return datetime.now().isoformat()


@contextlib.contextmanager
def _alarm_timeout(seconds: int, message: str) -> Iterator[None]:
    if (
        seconds <= 0
        or not hasattr(signal, "SIGALRM")
        or threading.current_thread() is not threading.main_thread()
    ):
        yield
        return

    previous_handler = signal.getsignal(signal.SIGALRM)

    def raise_timeout(_signum: int, _frame: Any) -> None:
        raise TimeoutError(message)

    signal.signal(signal.SIGALRM, raise_timeout)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous_handler)
