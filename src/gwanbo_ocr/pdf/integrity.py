"""PDF integrity checks adapted from the Peti validator."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from typing import Any

from .io import (
    file_md5,
    file_sha256,
    iter_pdf_paths,
    read_jsonl,
    write_json_atomic,
    write_jsonl_atomic,
)

try:  # Prefer the maintained package when it exists.
    from pypdf import PdfReader as _PdfReader  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - depends on optional environment.
    try:
        from PyPDF2 import PdfReader as _PdfReader  # type: ignore[import-not-found]
    except ImportError:  # pragma: no cover - default in the kata environment.
        _PdfReader = None  # type: ignore[assignment]


EOF_TAIL_BYTES = 4096


def is_pdf_header(data: bytes) -> bool:
    """Return true when *data* starts with a PDF header."""
    return data.startswith(b"%PDF-")


def has_pdf_eof(data: bytes, *, tail_bytes: int = EOF_TAIL_BYTES) -> bool:
    """Return true when a PDF EOF marker appears near the end of *data*."""
    return b"%%EOF" in data[-tail_bytes:]


def is_complete_pdf_bytes(data: bytes) -> bool:
    """Return true for bytes that look like a complete PDF transfer."""
    return is_pdf_header(data) and has_pdf_eof(data)


def validate_pdf_integrity(
    pdf_path: Path | str,
    *,
    include_hashes: bool = True,
    use_reader: bool = True,
    min_size_bytes: int = 1,
) -> dict[str, Any]:
    """Validate a single PDF file without requiring optional PDF libraries."""
    path = Path(pdf_path)
    result: dict[str, Any] = {
        "path": str(path),
        "filename": path.name,
        "timestamp": datetime.now().isoformat(),
        "checks": {},
    }

    exists = path.exists()
    result["checks"]["exists"] = {
        "status": "pass" if exists else "fail",
        "message": "file exists" if exists else "file missing",
    }
    if not exists:
        result.update({"valid": False, "overall_status": "fail", "status": "error"})
        return result

    try:
        size_bytes = path.stat().st_size
    except OSError as exc:
        result["checks"]["stat"] = {"status": "fail", "message": str(exc)}
        result.update({"valid": False, "overall_status": "fail", "status": "error"})
        return result

    result["size_bytes"] = size_bytes
    result["size_mb"] = round(size_bytes / (1024 * 1024), 4)
    result["checks"]["file_size"] = {
        "status": "pass" if size_bytes >= min_size_bytes else "fail",
        "value": size_bytes,
        "message": f"file has at least {min_size_bytes} byte(s)",
    }

    try:
        data = path.read_bytes()
        result["checks"]["readable"] = {"status": "pass", "message": "file is readable"}
    except OSError as exc:
        result["checks"]["readable"] = {"status": "fail", "message": str(exc)}
        result.update({"valid": False, "overall_status": "fail", "status": "error"})
        return result

    header = data[:8]
    header_ok = is_pdf_header(data)
    result["checks"]["pdf_header"] = {
        "status": "pass" if header_ok else "fail",
        "value": header.decode("ascii", errors="ignore"),
        "message": "PDF header present" if header_ok else "PDF header missing",
    }
    if header_ok:
        result["pdf_version"] = header[5:8].decode("ascii", errors="ignore")

    eof_ok = has_pdf_eof(data)
    result["checks"]["pdf_eof"] = {
        "status": "pass" if eof_ok else "fail",
        "message": "EOF marker present near file end" if eof_ok else "EOF marker missing near file end",
    }

    structure_ok = header_ok and eof_ok and size_bytes >= min_size_bytes
    result["checks"]["pdf_structure"] = {
        "status": "pass" if structure_ok else "fail",
        "message": "basic PDF structure present" if structure_ok else "basic PDF structure incomplete",
    }

    if include_hashes:
        result["md5"] = file_md5(path)
        result["sha256"] = file_sha256(path)

    if use_reader:
        result["checks"]["reader_validation"] = _reader_check(path)

    all_passed = all(check.get("status") in {"pass", "skip"} for check in result["checks"].values())
    result["valid"] = all_passed
    result["overall_status"] = "pass" if all_passed else "fail"
    result["status"] = "ok" if all_passed else "error"
    return result


def validate_pdf_directory(
    pdf_dir: Path | str,
    *,
    limit: int | None = None,
    include_hashes: bool = True,
    use_reader: bool = True,
) -> dict[str, Any]:
    """Validate PDFs below a directory and return a compact summary."""
    root = Path(pdf_dir)
    paths = list(iter_pdf_paths(root))
    if limit is not None:
        paths = paths[:limit]

    results = [
        validate_pdf_integrity(path, include_hashes=include_hashes, use_reader=use_reader)
        for path in paths
    ]
    passed = sum(1 for item in results if item.get("overall_status") == "pass")
    failed = sum(1 for item in results if item.get("overall_status") == "fail")
    return {
        "status": "success" if root.exists() else "error",
        "pdf_dir": str(root),
        "timestamp": datetime.now().isoformat(),
        "summary": {
            "total_files": len(results),
            "passed": passed,
            "failed": failed,
            "pass_rate": round(passed / len(results), 4) if results else 0.0,
        },
        "results": results,
    }


class PDFValidator:
    """Compatibility wrapper around the functional integrity API."""

    def __init__(self, pdf_directory: str | Path = "artifacts/pdfs"):
        self.pdf_dir = Path(pdf_directory)
        self.validation_results: list[dict[str, Any]] = []

    def validate_pdf(self, pdf_path: Path | str) -> dict[str, Any]:
        return validate_pdf_integrity(pdf_path)

    def validate_all_pdfs(self) -> dict[str, Any]:
        report = validate_pdf_directory(self.pdf_dir)
        self.validation_results = list(report["results"])
        return report

    def save_report(self, output_file: str | Path = "artifacts/validation_report.json") -> bool:
        try:
            payload = {
                "validation_timestamp": datetime.now().isoformat(),
                "summary": {
                    "total_files": len(self.validation_results),
                    "passed": sum(1 for item in self.validation_results if item.get("overall_status") == "pass"),
                    "failed": sum(1 for item in self.validation_results if item.get("overall_status") == "fail"),
                },
                "results": self.validation_results,
            }
            write_json_atomic(output_file, payload)
            return True
        except OSError:
            return False


def summarize_integrity(results: Iterable[dict[str, Any]]) -> dict[str, int]:
    """Return pass/fail counts for existing validation results."""
    rows = list(results)
    return {
        "total": len(rows),
        "passed": sum(1 for row in rows if row.get("overall_status") == "pass"),
        "failed": sum(1 for row in rows if row.get("overall_status") == "fail"),
    }


def validate_manifest(
    *,
    manifest_path: Path | str,
    output_dir: Path | str,
    limit: int | None = None,
) -> dict[str, Any]:
    """Validate PDF files referenced by a manifest JSONL."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    counts = {"total": 0, "passed": 0, "failed": 0}
    for index, row in enumerate(read_jsonl(manifest_path), start=1):
        if limit is not None and index > limit:
            break
        if not isinstance(row, dict):
            continue
        counts["total"] += 1
        pdf_path = _row_pdf_path(row)
        integrity = validate_pdf_integrity(pdf_path, include_hashes=True, use_reader=False)
        if integrity.get("overall_status") == "pass":
            counts["passed"] += 1
        else:
            counts["failed"] += 1
        rows.append(
            {
                "id": row.get("id"),
                "pdf_key": row.get("pdf_key") or row.get("metadata_key"),
                "pdf_path": str(pdf_path),
                "integrity": integrity,
            }
        )
    report_path = output / "manifest.jsonl"
    write_jsonl_atomic(report_path, rows)
    summary = {
        "status": "ok",
        "input": str(manifest_path),
        "output_dir": str(output),
        "manifest": str(report_path),
        "counts": counts,
    }
    write_json_atomic(output / "summary.json", summary)
    return summary


def _reader_check(path: Path) -> dict[str, Any]:
    if _PdfReader is None:
        return {"status": "skip", "message": "pypdf/PyPDF2 is not installed"}
    try:
        reader = _PdfReader(str(path))
        return {
            "status": "pass",
            "pages": len(reader.pages),
            "message": "PDF reader opened file",
        }
    except Exception as exc:  # noqa: BLE001 - optional parser reports many exception types.
        return {"status": "fail", "message": str(exc)}


def _row_pdf_path(row: dict[str, Any]) -> Path:
    for key in ("pdf_abs_path", "pdf_path", "path"):
        value = str(row.get(key) or "").strip()
        if value:
            return Path(value)
    return Path("")
