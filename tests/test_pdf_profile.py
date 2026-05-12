from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from typer.testing import CliRunner

from gwanbo_ocr.cli import app
from gwanbo_ocr.pdf.profile import compact_profile, profile_manifest, profile_row

runner = CliRunner()


def test_profile_row_handles_metadata_only_missing_pdf(tmp_path: Path) -> None:
    row = {
        "id": "item-001",
        "theme": "searchThema",
        "date": "2024-01-02",
        "category": "고시",
        "pdf_abs_path": str(tmp_path / "missing.pdf"),
    }

    profile = profile_row(row)

    assert profile["schema_version"] == "pdf-profile/v1"
    assert profile["pdf_key"] == "item-001"
    assert profile["year"] == "2024"
    assert profile["integrity_status"] == "missing"
    assert profile["text_mode"] == "missing"
    assert profile["error"] == "file_missing"


def test_profile_row_merges_text_and_layout_features(tmp_path: Path) -> None:
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"%PDF-1.4\n%%EOF")
    row = {
        "id": "item-002",
        "theme": "pety",
        "year": "2001",
        "category": "공고",
        "pdf_abs_path": str(pdf),
    }

    with (
        patch(
            "gwanbo_ocr.pdf.profile.validate_pdf_integrity",
            return_value={"valid": True, "overall_status": "pass", "size_bytes": 14},
        ),
        patch(
            "gwanbo_ocr.pdf.profile.analyze_pdf_text",
            return_value={
                "status": "ok",
                "pages": 2,
                "text_extractable": True,
                "total_chars": 100,
            },
        ),
        patch(
            "gwanbo_ocr.pdf.profile.analyze_pdf_layout",
            return_value={
                "status": "ok",
                "layout": {
                    "document_class": "table_heavy",
                    "metrics": {
                        "table_count": 3,
                        "table_text_ratio": 0.72,
                        "form_score": 0.1,
                        "text_quality": "clean",
                    },
                },
                "tables": [{}, {}, {}],
            },
        ),
    ):
        profile = profile_row(row)

    assert profile["text_mode"] == "text"
    assert profile["layout_class"] == "table_heavy"
    assert profile["table_count"] == 3
    assert profile["table_text_ratio"] == 0.72
    assert profile["text_quality"] == "clean"
    assert compact_profile(profile)["pdf_key"] == "item-002"


def test_profile_manifest_samples_per_metadata_bucket(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    rows = [
        {"id": "a1", "theme": "searchThema", "year": "2024", "category": "고시"},
        {"id": "a2", "theme": "searchThema", "year": "2024", "category": "고시"},
        {"id": "b1", "theme": "searchThema", "year": "2024", "category": "공고"},
    ]
    manifest.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "profiles"

    summary = profile_manifest(
        manifest_path=manifest,
        output_dir=output,
        sample_per_bucket=1,
        workers=1,
    )

    profile_rows = [
        json.loads(line)
        for line in (output / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert summary["selected_rows"] == 2
    assert {row["id"] for row in profile_rows} == {"a1", "b1"}


def test_pdf_profile_cli_help() -> None:
    result = runner.invoke(app, ["pdf", "profile", "--help"])

    assert result.exit_code == 0
    assert "--sample-per-bucket" in result.output
