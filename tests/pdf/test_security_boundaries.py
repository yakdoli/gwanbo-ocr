from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if "pymupdf" not in sys.modules:
    try:
        import pymupdf  # noqa: F401
    except ModuleNotFoundError:
        sys.modules["pymupdf"] = types.ModuleType("pymupdf")

from gwanbo_ocr.pdf.classification import classify_manifest
from gwanbo_ocr.pdf.content import PdfResourceLimitError, inspect_pdf_content
from gwanbo_ocr.pdf.integrity_manifest import validate_manifest
from gwanbo_ocr.pdf.io import UnsafePathError, pdf_key_from_row, resolve_pdf_path
from gwanbo_ocr.pdf.limits import PdfTimeoutError

TEXT_PDF = b"""%PDF-1.4
1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj
2 0 obj<</Type/Pages/Count 1/Kids[3 0 R]>>endobj
3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 300 144]/Contents 4 0 R/Resources<</Font<</F1 5 0 R>>>>>>endobj
4 0 obj<</Length 43>>stream
BT /F1 24 Tf 72 72 Td (Hello PDF Text) Tj ET
endstream endobj
5 0 obj<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>endobj
xref
0 6
0000000000 65535 f\x20
0000000010 00000 n\x20
0000000060 00000 n\x20
0000000117 00000 n\x20
0000000243 00000 n\x20
0000000346 00000 n\x20
trailer<</Size 6/Root 1 0 R>>
startxref
416
%%EOF
"""


def test_pdf_key_rejects_output_path_traversal() -> None:
    with pytest.raises(UnsafePathError):
        pdf_key_from_row({"id": "../../outside"})


def test_resolve_pdf_path_rejects_path_outside_trusted_root(tmp_path: Path) -> None:
    trusted_root = tmp_path / "corpus"
    trusted_root.mkdir()
    outside = tmp_path / "outside.pdf"
    outside.write_bytes(b"%PDF-1.4\n%%EOF")

    with pytest.raises(UnsafePathError):
        resolve_pdf_path({"pdf_abs_path": str(outside)}, trusted_root=trusted_root)


def test_resolve_pdf_path_rejects_symlink_escape(tmp_path: Path) -> None:
    trusted_root = tmp_path / "corpus"
    trusted_root.mkdir()
    outside = tmp_path / "outside.pdf"
    outside.write_bytes(b"%PDF-1.4\n%%EOF")
    link = trusted_root / "linked.pdf"
    link.symlink_to(outside)

    with pytest.raises(UnsafePathError):
        resolve_pdf_path({"pdf_abs_path": str(link)}, trusted_root=trusted_root)


def test_content_inspection_rejects_file_over_size_limit(tmp_path: Path) -> None:
    pdf_path = tmp_path / "oversized.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n%%EOF")

    with pytest.raises(PdfResourceLimitError):
        inspect_pdf_content(pdf_path, max_file_bytes=1)


def test_content_inspection_enforces_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pdf_path = tmp_path / "slow.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n%%EOF")

    def slow_open(_path: Path) -> None:
        time.sleep(1)

    monkeypatch.setattr("gwanbo_ocr.pdf.content.pymupdf.open", slow_open)

    with pytest.raises(PdfTimeoutError):
        inspect_pdf_content(pdf_path, timeout_seconds=0.01)


def test_classification_keeps_processing_after_malformed_jsonl(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        "{not-json}\n"
        + json.dumps({"id": "missing", "pdf_abs_path": str(tmp_path / "x.pdf")})
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "classification"

    summary = classify_manifest(manifest_path=manifest, output_dir=output)

    rows = [json.loads(line) for line in (output / "manifest.jsonl").read_text().splitlines()]
    assert summary["status"] == "error"
    assert summary["counts"]["error"] == 1
    assert len(rows) == 2
    assert rows[0]["status"] == "error"


def test_classification_manifest_resolves_relative_pdf_path_from_manifest_parent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_dir = tmp_path / "batch"
    manifest_dir.mkdir()
    (manifest_dir / "pdfs").mkdir()
    (manifest_dir / "cwd" / "pdfs").mkdir(parents=True)
    manifest = manifest_dir / "manifest.jsonl"
    manifest.write_text(
        json.dumps({"id": "doc", "pdf_path": "pdfs/doc.pdf"}) + "\n", encoding="utf-8"
    )

    (manifest_dir / "pdfs" / "doc.pdf").write_bytes(TEXT_PDF)
    (manifest_dir / "cwd" / "pdfs" / "doc.pdf").write_bytes(TEXT_PDF)

    monkeypatch.chdir(manifest_dir / "cwd")
    monkeypatch.setattr(
        "gwanbo_ocr.pdf.classification_manifest.classify_pdf_document",
        lambda pdf_path, **_: {"document_class": "text_pdf", "integrity": {}, "text": {}},
    )

    summary = classify_manifest(manifest_path=manifest, output_dir=tmp_path / "classification")
    row = json.loads((tmp_path / "classification" / "manifest.jsonl").read_text().splitlines()[0])

    assert summary["counts"]["processed"] == 1
    assert row["pdf_path"] == str(manifest_dir / "pdfs" / "doc.pdf")


def test_classification_rejects_duplicate_pdf_keys_without_overwriting_sidecar(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = tmp_path / "first.pdf"
    second = tmp_path / "second.pdf"
    first.write_bytes(TEXT_PDF)
    second.write_bytes(TEXT_PDF)
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        "\n".join(
            (
                json.dumps({"id": "same", "source": "first", "pdf_abs_path": str(first)}),
                json.dumps({"id": "same", "source": "second", "pdf_abs_path": str(second)}),
            )
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "gwanbo_ocr.pdf.classification_manifest.classify_pdf_document",
        lambda pdf_path, **_: {"document_class": "text_pdf", "integrity": {}, "text": {}},
    )

    summary = classify_manifest(manifest_path=manifest, output_dir=tmp_path / "classification")

    sidecar = json.loads(
        (tmp_path / "classification" / "items" / "same.json").read_text(encoding="utf-8")
    )
    assert (
        summary["status"],
        summary["counts"]["processed"],
        summary["counts"]["error"],
        sidecar["source"],
    ) == ("error", 1, 1, "first")


def test_integrity_manifest_reports_failed_batch_status(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("{not-json}\n", encoding="utf-8")

    summary = validate_manifest(manifest_path=manifest, output_dir=tmp_path / "integrity")

    assert (summary["status"], summary["counts"]["failed"]) == ("error", 1)


def test_integrity_manifest_resolves_relative_pdf_path_from_manifest_parent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_dir = tmp_path / "batch"
    manifest_dir.mkdir()
    (manifest_dir / "pdfs").mkdir()
    (manifest_dir / "cwd" / "pdfs").mkdir(parents=True)
    manifest = manifest_dir / "manifest.jsonl"
    manifest.write_text(
        json.dumps({"id": "doc", "pdf_path": "pdfs/doc.pdf"}) + "\n", encoding="utf-8"
    )

    (manifest_dir / "pdfs" / "doc.pdf").write_bytes(TEXT_PDF)
    (manifest_dir / "cwd" / "pdfs" / "doc.pdf").write_bytes(TEXT_PDF)

    monkeypatch.chdir(manifest_dir / "cwd")
    monkeypatch.setattr(
        "gwanbo_ocr.pdf.integrity_manifest.validate_pdf_integrity",
        lambda pdf_path, **_: {"overall_status": "pass"},
    )

    summary = validate_manifest(manifest_path=manifest, output_dir=tmp_path / "integrity")
    row = json.loads((tmp_path / "integrity" / "manifest.jsonl").read_text().splitlines()[0])

    assert summary["counts"]["passed"] == 1
    assert row["pdf_path"] == str(manifest_dir / "pdfs" / "doc.pdf")


def test_correction_module_imports_without_pdf_extra() -> None:
    source_root = Path(__file__).resolve().parents[2] / "src"
    script = """
import builtins
original_import = builtins.__import__
def guarded_import(name, *args, **kwargs):
    if name == "pymupdf":
        raise ImportError("pymupdf intentionally unavailable")
    return original_import(name, *args, **kwargs)
builtins.__import__ = guarded_import
from gwanbo_ocr.pdf.correction import correct_ocr_text
assert correct_ocr_text("관보").corrected_text == "관보"
"""
    environment = {**os.environ, "PYTHONPATH": str(source_root)}

    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert result.returncode == 0, result.stderr
