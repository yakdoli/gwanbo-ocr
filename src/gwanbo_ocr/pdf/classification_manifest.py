"""Streaming manifest adapter for PDF classification."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from .classification import classify_pdf_document
from .io import (
    UnsafePathError,
    iter_jsonl_records,
    pdf_key_from_row,
    resolve_pdf_path,
    safe_output_path,
    write_json_atomic,
    write_jsonl_atomic,
)
from .limits import DEFAULT_MAX_FILE_BYTES


def classify_manifest(
    *,
    manifest_path: Path | str,
    output_dir: Path | str,
    max_pages: int | None = 3,
    method: str = "extract-text",
    only_missing: bool = False,
    force: bool = False,
    limit: int | None = None,
    peti_root: Path | str | None = None,
    timeout_seconds: float = 30,
    max_file_bytes: int = DEFAULT_MAX_FILE_BYTES,
) -> dict[str, Any]:
    """Classify PDF rows from a JSONL manifest and write sidecars."""
    output = Path(output_dir)
    items_dir = output / "items"
    items_dir.mkdir(parents=True, exist_ok=True)
    counts = _initial_counts()
    seen_pdf_keys: set[str] = set()
    trusted_root = Path(peti_root) if peti_root is not None else Path(manifest_path).parent
    manifest_output = output / "manifest.jsonl"

    def classified_rows() -> Iterator[dict[str, Any]]:
        for line_number, row in iter_jsonl_records(manifest_path):
            if limit is not None and counts["total"] >= limit:
                break
            counts["total"] += 1
            if not isinstance(row, dict) or row.get("status") == "error":
                counts["error"] += 1
                yield _manifest_error_row(line_number, "malformed_jsonl")
                continue
            try:
                pdf_key = pdf_key_from_row(row)
                if pdf_key in seen_pdf_keys:
                    counts["error"] += 1
                    yield _manifest_error_row(line_number, "duplicate_pdf_key")
                    continue
                seen_pdf_keys.add(pdf_key)
                sidecar_path = safe_output_path(items_dir, f"{pdf_key}.json")
                if sidecar_path.exists() and only_missing and not force:
                    counts["skipped_existing"] += 1
                    yield _compact_row(_read_json_dict(sidecar_path) or {}, sidecar_path)
                    continue
                pdf_path = resolve_pdf_path(
                    row,
                    base_dir=Path(manifest_path).parent,
                    peti_root=peti_root,
                    trusted_root=trusted_root,
                )
            except UnsafePathError:
                counts["error"] += 1
                yield _manifest_error_row(line_number, "unsafe_path")
                continue

            text_metadata = _text_metadata(row) if method == "manifest-metadata" else None
            classification = classify_pdf_document(
                pdf_path,
                text_metadata=text_metadata,
                analyze_text=method != "manifest-metadata",
                max_pages=max_pages,
                timeout_seconds=timeout_seconds,
                max_file_bytes=max_file_bytes,
            )
            doc_class = str(classification.get("document_class") or "error")
            counts["processed"] += 1
            counts[doc_class] = counts.get(doc_class, 0) + 1
            sidecar = _sidecar(row, pdf_key, pdf_path, classification)
            write_json_atomic(sidecar_path, sidecar)
            yield _compact_row(sidecar, sidecar_path)

    write_jsonl_atomic(manifest_output, classified_rows())
    summary = {
        "status": "error" if counts["error"] else "ok",
        "input": str(manifest_path),
        "output_dir": str(output),
        "method": method,
        "max_pages": max_pages,
        "peti_root": str(peti_root) if peti_root is not None else None,
        "counts": counts,
        "manifest": str(manifest_output),
    }
    write_json_atomic(output / "summary.json", summary)
    return summary


def _initial_counts() -> dict[str, int]:
    return {
        "total": 0,
        "processed": 0,
        "skipped_existing": 0,
        "text_pdf": 0,
        "image_or_unextractable_pdf": 0,
        "invalid_pdf": 0,
        "missing_pdf": 0,
        "error": 0,
    }


def _text_metadata(row: dict[str, Any]) -> dict[str, Any] | None:
    if "text_extractable" not in row:
        return None
    return {
        "status": "ok",
        "text_extractable": bool(row.get("text_extractable")),
        "pages": row.get("pages"),
        "text_pages": row.get("text_pages"),
        "total_chars": row.get("total_chars") or 0,
        "scanned_pages": row.get("scanned_pages"),
    }


def _sidecar(
    row: dict[str, Any], pdf_key: str, pdf_path: Path, classification: dict[str, Any]
) -> dict[str, Any]:
    doc_class = str(classification.get("document_class") or "error")
    return {
        "schema_version": "pdf-classification/v1",
        "pdf_key": pdf_key,
        "source": row.get("theme") or row.get("source"),
        "id": row.get("id"),
        "metadata_path": row.get("source_metadata_path") or row.get("metadata_path"),
        "pdf_path": str(pdf_path),
        "manifest_row": row,
        "status": "ok" if doc_class != "error" else "error",
        "integrity": classification.get("integrity"),
        "native_text": classification.get("text"),
        "decision": _decision(classification),
        "classification": classification,
    }


def _decision(classification: dict[str, Any]) -> dict[str, Any]:
    doc_class = str(classification.get("document_class") or "error")
    text_extractable = bool(classification.get("text_extractable"))
    return {
        "document_kind": doc_class,
        "text_extractable": text_extractable,
        "needs_ocr": doc_class
        in {"image_or_unextractable_pdf", "missing_pdf", "invalid_pdf", "error"},
        "layout_eligible": doc_class == "text_pdf" and text_extractable,
        "preferred_text_source": "pdf_text" if text_extractable else None,
        "confidence": classification.get("confidence", 0.0),
        "reason": "; ".join(str(reason) for reason in classification.get("reasons") or []),
    }


def _compact_row(sidecar: dict[str, Any], sidecar_path: Path) -> dict[str, Any]:
    decision_payload = sidecar.get("decision")
    decision: dict[str, Any] = decision_payload if isinstance(decision_payload, dict) else {}
    native_text_payload = sidecar.get("native_text")
    native_text: dict[str, Any] = (
        native_text_payload if isinstance(native_text_payload, dict) else {}
    )
    classification_payload = sidecar.get("classification")
    classification: dict[str, Any] = (
        classification_payload if isinstance(classification_payload, dict) else {}
    )
    return {
        "schema_version": "pdf-classification-manifest/v1",
        "pdf_key": sidecar.get("pdf_key"),
        "source": sidecar.get("source"),
        "id": sidecar.get("id"),
        "pdf_path": sidecar.get("pdf_path"),
        "sidecar_path": str(sidecar_path),
        "status": sidecar.get("status"),
        "document_kind": decision.get("document_kind"),
        "text_extractable": decision.get("text_extractable"),
        "needs_ocr": decision.get("needs_ocr"),
        "layout_eligible": decision.get("layout_eligible"),
        "pages": native_text.get("pages"),
        "total_chars": native_text.get("total_chars"),
        "error": classification.get("error"),
    }


def _read_json_dict(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _manifest_error_row(line_number: int, error: str) -> dict[str, Any]:
    return {
        "schema_version": "pdf-classification-manifest/v1",
        "status": "error",
        "line_number": line_number,
        "error": error,
    }
