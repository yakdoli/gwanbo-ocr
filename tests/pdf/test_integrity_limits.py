from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from gwanbo_ocr.pdf import integrity as pdf_integrity
from gwanbo_ocr.pdf.integrity import validate_pdf_integrity
from gwanbo_ocr.pdf.integrity_manifest import validate_manifest


def test_integrity_rejects_oversized_pdf_before_hashing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pdf_path = tmp_path / "large.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n%%EOF")

    def fail_hash(_path: Path) -> str:
        raise AssertionError("hashing must not run")

    monkeypatch.setattr(pdf_integrity, "file_md5", fail_hash)
    monkeypatch.setattr(pdf_integrity, "file_sha256", fail_hash)

    result = validate_pdf_integrity(pdf_path, max_file_bytes=1)

    assert (result["status"], result["checks"]["file_size"]["status"]) == ("error", "fail")


def test_integrity_hashing_enforces_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pdf_path = tmp_path / "slow.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n%%EOF")

    def slow_hash(_path: Path) -> str:
        time.sleep(1)
        return "unused"

    monkeypatch.setattr(pdf_integrity, "file_md5", slow_hash)

    result = validate_pdf_integrity(pdf_path, timeout_seconds=0.01)

    assert (result["status"], result["checks"]["resource_limit"]["status"]) == (
        "error",
        "fail",
    )


def test_integrity_manifest_propagates_file_size_limit(tmp_path: Path) -> None:
    pdf_path = tmp_path / "large.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n%%EOF")
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps({"id": "doc", "pdf_abs_path": str(pdf_path)}) + "\n")

    summary = validate_manifest(
        manifest_path=manifest,
        output_dir=tmp_path / "integrity",
        max_file_bytes=1,
    )

    assert (summary["status"], summary["counts"]["failed"]) == ("error", 1)
