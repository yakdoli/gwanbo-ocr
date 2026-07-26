from __future__ import annotations

import json
from pathlib import Path

import pymupdf
from typer.testing import CliRunner

from gwanbo_ocr.cli import app

runner = CliRunner()


def test_pdf_classify_writes_manifest(tmp_path: Path) -> None:
    pdf_path = tmp_path / "artifacts" / "pety" / "pdfs" / "native.pdf"
    pdf_path.parent.mkdir(parents=True)
    with pymupdf.open() as document:
        page = document.new_page()
        page.insert_text((40, 80), "native text")
        document.save(pdf_path)
    manifest_path = tmp_path / "manifest.jsonl"
    manifest_path.write_text(
        json.dumps({"id": "native", "pdf_path": "artifacts/pety/pdfs/native.pdf"}) + "\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "classification"

    result = runner.invoke(
        app,
        [
            "pdf",
            "classify",
            "--input",
            str(manifest_path),
            "--output",
            str(output_dir),
            "--peti-root",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 0
    row = json.loads((output_dir / "manifest.jsonl").read_text(encoding="utf-8"))
    assert row["document_kind"] == "text_pdf"
    assert row["needs_ocr"] is False


def test_pdf_classify_help_does_not_advertise_unused_workers() -> None:
    result = runner.invoke(app, ["pdf", "classify", "--help"])

    assert "--workers" not in result.output


def test_pdf_correct_preserves_raw_text_and_writes_audit(tmp_path: Path) -> None:
    input_path = tmp_path / "ocr.jsonl"
    input_path.write_text(
        json.dumps({"document_id": "doc-1", "raw_text": "관보  1,234원"}, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )
    output_path = tmp_path / "corrected.jsonl"

    result = runner.invoke(
        app,
        ["pdf", "correct", "--input", str(input_path), "--output", str(output_path)],
    )

    assert result.exit_code == 0
    row = json.loads(output_path.read_text(encoding="utf-8"))
    assert row["raw_text"] == "관보  1,234원"
    assert row["ocr_correction"]["corrected_text"] == "관보 1,234원"
    assert row["ocr_correction"]["protected_values"] == ["1,234"]


def test_pdf_correct_applies_metadata_consistency_gate(tmp_path: Path) -> None:
    # Given
    input_path = tmp_path / "ocr.jsonl"
    input_path.write_text(
        json.dumps(
            {
                "document_id": "doc-1",
                "raw_text": "관보",
                "document_metadata": {
                    "document_id": "doc-1",
                    "title": "OCR 제목",
                    "date": "2024/01/03",
                    "page_count": 1,
                },
                "source_metadata": {
                    "id": "doc-1",
                    "title": "관보 공고",
                    "date": "2024-01-02",
                    "pdf_text": {"pages": 2},
                },
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    output_path = tmp_path / "corrected.jsonl"

    # When
    result = runner.invoke(
        app,
        ["pdf", "correct", "--input", str(input_path), "--output", str(output_path)],
    )

    # Then
    assert result.exit_code == 0
    row = json.loads(output_path.read_text(encoding="utf-8"))
    audit = row["metadata_correction"]
    assert audit["status"] == "corrected"
    assert audit["can_publish"] is True
    assert audit["original_metadata"]["title"] == "OCR 제목"
    assert audit["corrected_metadata"]["title"] == "관보 공고"
    assert audit["corrected_metadata"]["page_count"] == 2


def test_pdf_correct_blocks_rejected_metadata_at_output_boundary(tmp_path: Path) -> None:
    input_path = tmp_path / "ocr.jsonl"
    input_path.write_text(
        json.dumps(
            {
                "document_id": "wrong-document",
                "raw_text": "관보",
                "document_metadata": {
                    "document_id": "wrong-document",
                    "title": "관보 공고",
                    "date": "2024-01-02",
                },
                "source_metadata": {
                    "id": "doc-1",
                    "title": "관보 공고",
                    "date": "2024-01-02",
                },
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    output_path = tmp_path / "corrected.jsonl"

    result = runner.invoke(
        app,
        ["pdf", "correct", "--input", str(input_path), "--output", str(output_path)],
    )

    row = json.loads(output_path.read_text(encoding="utf-8"))
    summary = json.loads(result.output)
    assert (row["status"], row["document_id"], summary["processed"], summary["errors"]) == (
        "rejected",
        "doc-1",
        0,
        1,
    )


def test_pdf_correct_requires_crop_root_for_vlm(tmp_path: Path) -> None:
    input_path = tmp_path / "ocr.jsonl"
    input_path.write_text('{"raw_text":"관보"}\n', encoding="utf-8")

    result = runner.invoke(
        app,
        [
            "pdf",
            "correct",
            "--input",
            str(input_path),
            "--output",
            str(tmp_path / "corrected.jsonl"),
            "--vlm-base-url",
            "http://localhost:8000/v1",
            "--vlm-model",
            "qwen-vl",
        ],
    )

    assert result.exit_code != 0
    assert "--vlm-crop-root" in result.output


def test_pdf_correct_help_does_not_allow_arbitrary_secret_environment_selection() -> None:
    result = runner.invoke(app, ["pdf", "correct", "--help"])

    assert "--vlm-api-key-env" not in result.output
