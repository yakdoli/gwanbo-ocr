"""High-level PDF classification helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .integrity import validate_pdf_integrity
from .io import (
    pdf_key_from_row,
    read_jsonl,
    resolve_pdf_path,
    write_json_atomic,
    write_jsonl_atomic,
)
from .text import analyze_pdf_text

PDF_CLASSES = {
    "missing_pdf",
    "invalid_pdf",
    "text_pdf",
    "image_or_unextractable_pdf",
    "error",
}


def classify_pdf_document(
    pdf_path: Path | str,
    *,
    text_metadata: dict[str, Any] | None = None,
    integrity_metadata: dict[str, Any] | None = None,
    analyze_text: bool = True,
    include_integrity_hashes: bool = False,
    max_pages: int | None = 3,
) -> dict[str, Any]:
    """Classify a PDF by integrity and text-layer extractability."""
    path = Path(pdf_path)
    if not path.exists():
        return {
            "path": str(path),
            "filename": path.name,
            "document_class": "missing_pdf",
            "confidence": 1.0,
            "valid_pdf": False,
            "text_extractable": False,
            "reasons": ["file_missing"],
        }

    integrity = integrity_metadata or validate_pdf_integrity(
        path,
        include_hashes=include_integrity_hashes,
        use_reader=False,
    )
    if not integrity.get("valid"):
        return {
            "path": str(path),
            "filename": path.name,
            "document_class": "invalid_pdf",
            "confidence": 0.95,
            "valid_pdf": False,
            "text_extractable": False,
            "reasons": _failed_checks(integrity),
            "integrity": integrity,
        }

    text = text_metadata
    if text is None and analyze_text:
        text = analyze_pdf_text(path, max_pages=max_pages)

    if text and text.get("status") == "error":
        return {
            "path": str(path),
            "filename": path.name,
            "document_class": "error",
            "confidence": 0.6,
            "valid_pdf": True,
            "text_extractable": False,
            "reasons": [str(text.get("error") or "text_analysis_error")],
            "integrity": integrity,
            "text": text,
        }

    text_extractable = bool((text or {}).get("text_extractable"))
    total_chars = int((text or {}).get("total_chars") or 0)
    if text_extractable:
        confidence = 0.95 if total_chars >= 30 else 0.82
        document_class = "text_pdf"
        reasons = ["extractable_text_layer"]
    else:
        confidence = 0.76 if text is not None else 0.55
        document_class = "image_or_unextractable_pdf"
        reasons = ["no_extractable_text_detected" if text is not None else "text_not_analyzed"]

    return {
        "path": str(path),
        "filename": path.name,
        "document_class": document_class,
        "confidence": confidence,
        "valid_pdf": True,
        "text_extractable": text_extractable,
        "total_chars": total_chars,
        "reasons": reasons,
        "integrity": integrity,
        "text": text,
    }


def classify_pdf_from_metadata(
    *,
    integrity: dict[str, Any] | None = None,
    text: dict[str, Any] | None = None,
    layout: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Classify from precomputed metadata without touching the filesystem."""
    reasons: list[str] = []
    if integrity is not None and not integrity.get(
        "valid", integrity.get("overall_status") == "pass"
    ):
        return {
            "document_class": "invalid_pdf",
            "confidence": 0.95,
            "reasons": _failed_checks(integrity),
        }
    if text is not None and text.get("status") == "error":
        return {
            "document_class": "error",
            "confidence": 0.6,
            "reasons": [str(text.get("error") or "text_error")],
        }
    if text is not None and text.get("text_extractable"):
        layout_class = _layout_class(layout)
        if layout_class:
            reasons.append(f"layout:{layout_class}")
        reasons.append("extractable_text_layer")
        layout_confidence = layout.get("confidence") if isinstance(layout, dict) else None
        return {
            "document_class": layout_class or "text_pdf",
            "confidence": float(layout_confidence or 0.9) if layout_class else 0.9,
            "text_extractable": True,
            "reasons": reasons,
        }
    if text is not None:
        return {
            "document_class": "image_or_unextractable_pdf",
            "confidence": 0.76,
            "text_extractable": False,
            "reasons": ["no_extractable_text_detected"],
        }
    return {"document_class": "unknown", "confidence": 0.0, "reasons": ["insufficient_metadata"]}


def _failed_checks(integrity: dict[str, Any]) -> list[str]:
    checks_payload = integrity.get("checks")
    checks: dict[str, Any] = checks_payload if isinstance(checks_payload, dict) else {}
    reasons = [
        name
        for name, check in checks.items()
        if isinstance(check, dict) and check.get("status") == "fail"
    ]
    return reasons or ["integrity_failed"]


def _layout_class(layout: dict[str, Any] | None) -> str:
    if not isinstance(layout, dict):
        return ""
    nested_payload = layout.get("layout")
    nested: dict[str, Any] = nested_payload if isinstance(nested_payload, dict) else layout
    value = str(nested.get("document_class") or "")
    return "" if value in {"", "unknown_text", "error"} else value


def classify_manifest(
    *,
    manifest_path: Path | str,
    output_dir: Path | str,
    max_pages: int | None = 3,
    workers: int = 1,
    method: str = "extract-text",
    only_missing: bool = False,
    force: bool = False,
    limit: int | None = None,
) -> dict[str, Any]:
    """Classify PDF rows from a JSONL manifest and write sidecars.

    ``workers`` is accepted for CLI/API stability; the current implementation is
    deterministic and sequential because PDF parser timeouts are signal-based.
    """
    del workers
    output = Path(output_dir)
    items_dir = output / "items"
    items_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    counts: dict[str, int] = {
        "total": 0,
        "processed": 0,
        "skipped_existing": 0,
        "text_pdf": 0,
        "image_or_unextractable_pdf": 0,
        "invalid_pdf": 0,
        "missing_pdf": 0,
        "error": 0,
    }

    for index, row in enumerate(read_jsonl(manifest_path), start=1):
        if limit is not None and index > limit:
            break
        if not isinstance(row, dict):
            continue
        counts["total"] += 1
        pdf_key = _row_pdf_key(row)
        sidecar_path = items_dir / f"{pdf_key}.json"
        if sidecar_path.exists() and only_missing and not force:
            counts["skipped_existing"] += 1
            existing = _read_json_dict(sidecar_path)
            rows.append(_compact_classification_manifest_row(existing or {}, sidecar_path))
            continue

        pdf_path = _row_pdf_path(row)
        text_metadata = None
        if method == "manifest-metadata" and "text_extractable" in row:
            text_metadata = {
                "status": "ok",
                "text_extractable": bool(row.get("text_extractable")),
                "pages": row.get("pages"),
                "text_pages": row.get("text_pages"),
                "total_chars": row.get("total_chars") or 0,
                "scanned_pages": row.get("scanned_pages"),
            }
        classification = classify_pdf_document(
            pdf_path,
            text_metadata=text_metadata,
            analyze_text=method != "manifest-metadata",
            max_pages=max_pages,
        )
        doc_class = str(classification.get("document_class") or "error")
        counts["processed"] += 1
        counts[doc_class] = counts.get(doc_class, 0) + 1

        sidecar = {
            "schema_version": "pdf-classification/v1",
            "pdf_key": pdf_key,
            "source": row.get("theme") or row.get("source"),
            "id": row.get("id"),
            "metadata_path": row.get("source_metadata_path") or row.get("metadata_path"),
            "pdf_path": str(pdf_path),
            "manifest_row": row,
            "status": "ok" if classification.get("document_class") not in {"error"} else "error",
            "integrity": classification.get("integrity"),
            "native_text": classification.get("text"),
            "decision": _decision_from_classification(classification),
            "classification": classification,
        }
        write_json_atomic(sidecar_path, sidecar)
        rows.append(_compact_classification_manifest_row(sidecar, sidecar_path))

    output.mkdir(parents=True, exist_ok=True)
    write_jsonl_atomic(output / "manifest.jsonl", rows)
    summary = {
        "status": "ok",
        "input": str(manifest_path),
        "output_dir": str(output),
        "method": method,
        "max_pages": max_pages,
        "counts": counts,
        "manifest": str(output / "manifest.jsonl"),
    }
    write_json_atomic(output / "summary.json", summary)
    return summary


def _decision_from_classification(classification: dict[str, Any]) -> dict[str, Any]:
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


def _compact_classification_manifest_row(
    sidecar: dict[str, Any], sidecar_path: Path
) -> dict[str, Any]:
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


def _row_pdf_path(row: dict[str, Any]) -> Path:
    return resolve_pdf_path(row, peti_root="/root/peti")


def _row_pdf_key(row: dict[str, Any]) -> str:
    return pdf_key_from_row(row)


def _read_json_dict(path: Path) -> dict[str, Any] | None:
    import json

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None
