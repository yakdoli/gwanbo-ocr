"""Streaming manifest adapter for PDF integrity validation."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

from .integrity import validate_pdf_integrity
from .io import (
    UnsafePathError,
    iter_jsonl_records,
    resolve_pdf_path,
    write_json_atomic,
    write_jsonl_atomic,
)
from .limits import DEFAULT_MAX_FILE_BYTES


def validate_manifest(
    *,
    manifest_path: Path | str,
    output_dir: Path | str,
    limit: int | None = None,
    peti_root: Path | str | None = None,
    max_file_bytes: int = DEFAULT_MAX_FILE_BYTES,
    timeout_seconds: float = 30,
) -> dict[str, Any]:
    """Validate PDF files referenced by a manifest JSONL."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    counts = {"total": 0, "passed": 0, "failed": 0}
    report_path = output / "manifest.jsonl"
    trusted_root = Path(peti_root) if peti_root is not None else Path(manifest_path).parent

    def integrity_rows() -> Iterator[dict[str, Any]]:
        for line_number, row in iter_jsonl_records(manifest_path):
            if limit is not None and counts["total"] >= limit:
                break
            counts["total"] += 1
            if not isinstance(row, dict) or row.get("status") == "error":
                counts["failed"] += 1
                yield _error_row(line_number, "malformed_jsonl")
                continue
            try:
                pdf_path = resolve_pdf_path(
                    row,
                    base_dir=Path(manifest_path).parent,
                    peti_root=peti_root,
                    trusted_root=trusted_root,
                )
            except UnsafePathError:
                counts["failed"] += 1
                yield _error_row(line_number, "unsafe_path")
                continue
            integrity = validate_pdf_integrity(
                pdf_path,
                include_hashes=True,
                use_reader=False,
                max_file_bytes=max_file_bytes,
                timeout_seconds=timeout_seconds,
            )
            if integrity.get("overall_status") == "pass":
                counts["passed"] += 1
            else:
                counts["failed"] += 1
            yield {
                "id": row.get("id"),
                "pdf_key": row.get("pdf_key") or row.get("metadata_key"),
                "pdf_path": str(pdf_path),
                "integrity": integrity,
            }

    write_jsonl_atomic(report_path, integrity_rows())
    summary = {
        "status": "error" if counts["failed"] else "ok",
        "input": str(manifest_path),
        "output_dir": str(output),
        "manifest": str(report_path),
        "counts": counts,
    }
    write_json_atomic(output / "summary.json", summary)
    return summary


def _error_row(line_number: int, error: str) -> dict[str, Any]:
    return {"status": "error", "line_number": line_number, "error": error}
