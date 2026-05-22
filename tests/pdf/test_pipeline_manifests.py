from __future__ import annotations

import json
from pathlib import Path

from gwanbo_ocr.pdf.classification import classify_manifest
from gwanbo_ocr.pdf.integrity import validate_manifest, validate_pdf_integrity
from gwanbo_ocr.pdf.layout import generate_layout_manifest

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


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_classify_manifest_writes_sidecars_and_respects_only_missing(tmp_path: Path) -> None:
    valid_pdf = tmp_path / "valid.pdf"
    valid_pdf.write_bytes(TEXT_PDF)
    missing_pdf = tmp_path / "missing.pdf"

    manifest_path = tmp_path / "manifest.jsonl"
    _write_jsonl(
        manifest_path,
        [
            {
                "id": "doc-valid",
                "pdf_key": "doc-valid",
                "pdf_abs_path": str(valid_pdf),
                "text_extractable": True,
                "total_chars": 64,
            },
            {
                "id": "doc-missing",
                "pdf_key": "doc-missing",
                "pdf_abs_path": str(missing_pdf),
                "text_extractable": False,
                "total_chars": 0,
            },
        ],
    )

    output_dir = tmp_path / "classification"
    first = classify_manifest(
        manifest_path=manifest_path,
        output_dir=output_dir,
        method="manifest-metadata",
        workers=8,
    )
    second = classify_manifest(
        manifest_path=manifest_path,
        output_dir=output_dir,
        method="manifest-metadata",
        only_missing=True,
    )

    assert first["counts"]["total"] == 2
    assert first["counts"]["processed"] == 2
    assert first["counts"]["text_pdf"] == 1
    assert first["counts"]["missing_pdf"] == 1
    assert second["counts"]["skipped_existing"] == 2
    assert (output_dir / "items" / "doc-valid.json").exists()
    assert (output_dir / "items" / "doc-missing.json").exists()


def test_validate_pdf_integrity_detects_structure_failures(tmp_path: Path) -> None:
    no_header = tmp_path / "no-header.pdf"
    no_header.write_bytes(b"NOTPDF\n%%EOF")
    no_eof = tmp_path / "no-eof.pdf"
    no_eof.write_bytes(b"%PDF-1.4\njust body")

    header_result = validate_pdf_integrity(no_header, use_reader=False, include_hashes=False)
    eof_result = validate_pdf_integrity(no_eof, use_reader=False, include_hashes=False)

    assert header_result["checks"]["pdf_header"]["status"] == "fail"
    assert header_result["checks"]["pdf_structure"]["status"] == "fail"
    assert eof_result["checks"]["pdf_eof"]["status"] == "fail"
    assert eof_result["checks"]["pdf_structure"]["status"] == "fail"


def test_validate_manifest_reports_pass_and_fail_rows(tmp_path: Path) -> None:
    valid_pdf = tmp_path / "ok.pdf"
    valid_pdf.write_bytes(TEXT_PDF)
    invalid_pdf = tmp_path / "bad.pdf"
    invalid_pdf.write_bytes(b"%PDF-1.4\nmissing eof marker")

    manifest_path = tmp_path / "manifest.jsonl"
    _write_jsonl(
        manifest_path,
        [
            {"id": "ok", "pdf_key": "ok", "pdf_abs_path": str(valid_pdf)},
            {"id": "bad", "pdf_key": "bad", "pdf_abs_path": str(invalid_pdf)},
        ],
    )

    summary = validate_manifest(manifest_path=manifest_path, output_dir=tmp_path / "integrity")

    assert summary["counts"] == {"total": 2, "passed": 1, "failed": 1}
    rows = (tmp_path / "integrity" / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(rows) == 2


def test_generate_layout_manifest_processes_only_eligible_rows(tmp_path: Path, monkeypatch) -> None:
    valid_pdf = tmp_path / "for-layout.pdf"
    valid_pdf.write_bytes(TEXT_PDF)
    classification_manifest = tmp_path / "classification-manifest.jsonl"
    _write_jsonl(
        classification_manifest,
        [
            {
                "pdf_key": "eligible",
                "pdf_abs_path": str(valid_pdf),
                "pdf_path": str(valid_pdf),
                "layout_eligible": True,
                "text_extractable": True,
                "sidecar_path": "dummy-sidecar.json",
            },
            {
                "pdf_key": "skip-me",
                "pdf_abs_path": str(valid_pdf),
                "pdf_path": str(valid_pdf),
                "layout_eligible": False,
                "text_extractable": False,
            },
        ],
    )

    def fake_analyze_pdf_layout(*_args, **_kwargs):
        return {
            "status": "ok",
            "text_extractable": True,
            "generated_at": "2026-05-13T00:00:00",
            "layout": {"document_class": "table_with_body", "confidence": 0.72},
            "tables": [{"table_id": "p001-t001"}],
        }

    monkeypatch.setattr("gwanbo_ocr.pdf.layout.analyze_pdf_layout", fake_analyze_pdf_layout)

    output_dir = tmp_path / "layout"
    summary = generate_layout_manifest(
        classification_manifest=classification_manifest,
        output_dir=output_dir,
        workers=4,
        table_strategy="text",
    )

    assert summary["eligible"] == 1
    assert summary["processed"] == 1
    assert summary["skipped_not_eligible"] == 1
    assert summary["tables"] == 1
    assert summary["by_layout_class"]["table_with_body"] == 1
    assert (output_dir / "items" / "eligible.json").exists()
