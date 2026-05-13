from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any
from unittest.mock import patch

from typer.testing import CliRunner

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from gwanbo_ocr.cli import app as cli_app
from gwanbo_ocr.peers import (
    aggregate_peer_scores,
    analyze_pdf_peer_review,
    run_peer_review_manifest,
)

# Minimal valid PDF bytes with one text page.
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


def _write_pdf(path: Path, payload: bytes = TEXT_PDF) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


# ---------------------------------------------------------------------------
# Orchestration: analyze_pdf_peer_review
# ---------------------------------------------------------------------------


class TestAnalyzePdfPeerReview:
    def test_runs_enabled_peers_and_produces_report(self, tmp_path: Path) -> None:
        pdf = _write_pdf(tmp_path / "doc.pdf")

        def fake_native(pdf_path: Path, **_kwargs: Any) -> dict[str, Any]:
            return {
                "status": "ok",
                "method": "pypdf.extract_text",
                "text_extractable": True,
                "text_chars": 14,
                "sample_text": "Hello PDF Text",
                "error": None,
            }

        with (
            patch("gwanbo_ocr.peers.extract_native_text", fake_native),
            patch(
                "gwanbo_ocr.peers.extract_pdfplumber",
                return_value={
                    "status": "skipped",
                    "text_chars": 0,
                    "skip_reason": "off",
                    "text_extractable": False,
                    "sample_text": "",
                    "error": None,
                },
            ),
            patch(
                "gwanbo_ocr.peers.extract_markitdown",
                return_value={
                    "status": "skipped",
                    "text_chars": 0,
                    "skip_reason": "off",
                    "text_extractable": False,
                    "sample_text": "",
                    "error": None,
                },
            ),
        ):
            report = analyze_pdf_peer_review(
                pdf,
                image_dir=tmp_path / "images",
                run_native_text=True,
                run_pdfplumber=False,
                run_markitdown=False,
                run_paddle=False,
            )

        assert report["status"] == "ok"
        assert "native_text" in report["peers"]
        assert "review" in report
        assert "decision" in report
        assert report["decision"]["text_extractable"] is True

    def test_includes_score_when_metadata_provided(self, tmp_path: Path) -> None:
        pdf = _write_pdf(tmp_path / "doc.pdf")

        fake_peer = {
            "status": "ok",
            "method": "test",
            "text_extractable": True,
            "text_chars": 10,
            "sample_text": "공보처 예산",
            "error": None,
        }

        with (
            patch("gwanbo_ocr.peers.extract_native_text", return_value=fake_peer),
            patch(
                "gwanbo_ocr.peers.extract_pdfplumber",
                return_value={
                    "status": "skipped",
                    "text_chars": 0,
                    "skip_reason": "off",
                    "text_extractable": False,
                    "sample_text": "",
                    "error": None,
                },
            ),
            patch(
                "gwanbo_ocr.peers.extract_markitdown",
                return_value={
                    "status": "skipped",
                    "text_chars": 0,
                    "skip_reason": "off",
                    "text_extractable": False,
                    "sample_text": "",
                    "error": None,
                },
            ),
        ):
            report = analyze_pdf_peer_review(
                pdf,
                image_dir=tmp_path / "images",
                run_native_text=True,
                run_pdfplumber=False,
                run_markitdown=False,
                run_paddle=False,
                metadata={"title": "예산 공고", "agency": "공보처"},
            )

        assert "score" in report
        assert "reference_tokens" in report["score"]

    def test_report_status_error_when_all_peers_fail(self, tmp_path: Path) -> None:
        pdf = _write_pdf(tmp_path / "doc.pdf")
        failed = {
            "status": "error",
            "text_chars": 0,
            "text_extractable": False,
            "sample_text": "",
            "error": "oops",
            "method": "x",
        }

        with (
            patch("gwanbo_ocr.peers.extract_native_text", return_value=failed),
            patch(
                "gwanbo_ocr.peers.extract_pdfplumber",
                return_value={**failed, "status": "skipped", "skip_reason": "off"},
            ),
            patch(
                "gwanbo_ocr.peers.extract_markitdown",
                return_value={**failed, "status": "skipped", "skip_reason": "off"},
            ),
        ):
            report = analyze_pdf_peer_review(
                pdf,
                image_dir=tmp_path / "images",
                run_native_text=True,
                run_pdfplumber=False,
                run_markitdown=False,
                run_paddle=False,
            )

        assert report["status"] == "error"

    def test_can_run_paddle_vl_without_classic_paddle(self, tmp_path: Path) -> None:
        pdf = _write_pdf(tmp_path / "doc.pdf")
        skipped = {
            "status": "skipped",
            "text_chars": 0,
            "skip_reason": "off",
            "text_extractable": False,
            "sample_text": "",
            "error": None,
        }
        paddle_vl_peer = {
            "status": "ok",
            "method": "PaddleOCR-VL",
            "text_extractable": True,
            "text_chars": 12,
            "sample_text": "관보 결과",
            "error": None,
        }

        with (
            patch("gwanbo_ocr.peers.extract_native_text", return_value=skipped),
            patch("gwanbo_ocr.peers.extract_pdfplumber", return_value=skipped),
            patch("gwanbo_ocr.peers.extract_markitdown", return_value=skipped),
            patch("gwanbo_ocr.peers.extract_paddle_ocr_vl", return_value=paddle_vl_peer),
            patch(
                "gwanbo_ocr.peers.extract_paddle_ocr",
                side_effect=AssertionError("classic PaddleOCR should not run"),
            ),
        ):
            report = analyze_pdf_peer_review(
                pdf,
                image_dir=tmp_path / "images",
                run_native_text=True,
                run_pdfplumber=False,
                run_markitdown=False,
                run_paddle=False,
                run_paddle_vl=True,
                paddle_vl_server_url="http://127.0.0.1:8000/v1",
                paddle_vl_model="PaddlePaddle/PaddleOCR-VL-1.5",
            )

        assert "paddle_ocr_vl" in report["peers"]
        assert "paddle_ocr" not in report["peers"]
        assert report["decision"]["preferred_text_source"] == "paddle_ocr_vl"


# ---------------------------------------------------------------------------
# Batch manifest processing
# ---------------------------------------------------------------------------


class TestRunPeerReviewManifest:
    def _write_manifest(self, path: Path, rows: list[dict[str, Any]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def test_processes_manifest_and_writes_index(self, tmp_path: Path) -> None:
        pdf = _write_pdf(tmp_path / "pdfs" / "doc.pdf")
        manifest = tmp_path / "manifest.jsonl"
        self._write_manifest(manifest, [{"id": "item-001", "pdf_path": str(pdf), "title": "공고"}])
        output_dir = tmp_path / "peer_output"

        fake_peer = {
            "status": "ok",
            "method": "test",
            "text_extractable": True,
            "text_chars": 10,
            "sample_text": "공보",
            "error": None,
        }
        with (
            patch("gwanbo_ocr.peers.extract_native_text", return_value=fake_peer),
            patch(
                "gwanbo_ocr.peers.extract_pdfplumber",
                return_value={**fake_peer, "status": "skipped", "skip_reason": "off"},
            ),
            patch(
                "gwanbo_ocr.peers.extract_markitdown",
                return_value={**fake_peer, "status": "skipped", "skip_reason": "off"},
            ),
            patch("gwanbo_ocr.peers.extract_paddle_ocr_vl", return_value=fake_peer),
        ):
            summary = run_peer_review_manifest(
                manifest_path=manifest,
                output_dir=output_dir,
                run_native_text=True,
                run_pdfplumber=False,
                run_markitdown=False,
                run_paddle=False,
                run_paddle_vl=True,
                paddle_vl_server_url="http://127.0.0.1:8000/v1",
                paddle_vl_model="PaddlePaddle/PaddleOCR-VL-1.5",
                workers=1,
            )

        assert summary["processed"] == 1
        assert summary["total"] == 1
        assert summary["settings"]["run_paddle_vl"] is True
        assert summary["settings"]["paddle_vl_model"] == "PaddlePaddle/PaddleOCR-VL-1.5"
        assert (output_dir / "metadata.json").exists()
        assert (output_dir / "summary.json").exists()
        assert (output_dir / "samples" / "item-001" / "source.json").exists()
        assert (output_dir / "samples" / "item-001" / "peer_samples.json").exists()
        assert (output_dir / "samples" / "item-001" / "diff_summary.json").exists()
        assert (output_dir / "samples" / "item-001" / "peer_samples.md").exists()

        index = json.loads((output_dir / "metadata.json").read_text(encoding="utf-8"))
        assert "item-001" in index
        assert index["item-001"]["sample_artifact_dir"].endswith("samples/item-001")
        assert index["item-001"]["sample_artifact_paths"]["source_json"].endswith(
            "samples/item-001/source.json"
        )

        source = json.loads(
            (output_dir / "samples" / "item-001" / "source.json").read_text(encoding="utf-8")
        )
        diff_summary = json.loads(
            (output_dir / "samples" / "item-001" / "diff_summary.json").read_text(encoding="utf-8")
        )
        assert source["manifest_path"] == str(manifest)
        assert source["sidecar_path"].endswith("items/item-001.json")
        assert source["artifact_paths"]["peer_samples_json"].endswith(
            "samples/item-001/peer_samples.json"
        )
        assert diff_summary["artifact_paths"]["source_json"].endswith(
            "samples/item-001/source.json"
        )

    def test_skips_existing_sidecars_when_not_forced(self, tmp_path: Path) -> None:
        pdf = _write_pdf(tmp_path / "doc.pdf")
        manifest = tmp_path / "manifest.jsonl"
        self._write_manifest(manifest, [{"id": "skip-me", "pdf_path": str(pdf)}])
        output_dir = tmp_path / "out"

        # Pre-create sidecar so it's already present.
        sidecar = output_dir / "items" / "skip-me.json"
        sidecar.parent.mkdir(parents=True)
        sidecar.write_text("{}", encoding="utf-8")

        with patch(
            "gwanbo_ocr.peers.extract_native_text",
            side_effect=AssertionError("should not be called"),
        ):
            summary = run_peer_review_manifest(
                manifest_path=manifest,
                output_dir=output_dir,
                run_native_text=True,
                run_pdfplumber=False,
                run_markitdown=False,
                run_paddle=False,
                force=False,
            )

        assert summary["skipped_existing"] == 1
        assert summary["processed"] == 0

    def test_limit_restricts_rows_processed(self, tmp_path: Path) -> None:
        pdfs = [_write_pdf(tmp_path / f"pdf{i}" / "doc.pdf") for i in range(5)]
        manifest = tmp_path / "manifest.jsonl"
        self._write_manifest(
            manifest, [{"id": f"item-{i}", "pdf_path": str(p)} for i, p in enumerate(pdfs)]
        )

        fake_peer = {
            "status": "error",
            "text_chars": 0,
            "text_extractable": False,
            "sample_text": "",
            "error": "x",
            "method": "t",
        }
        with (
            patch("gwanbo_ocr.peers.extract_native_text", return_value=fake_peer),
            patch(
                "gwanbo_ocr.peers.extract_pdfplumber",
                return_value={**fake_peer, "status": "skipped", "skip_reason": "off"},
            ),
            patch(
                "gwanbo_ocr.peers.extract_markitdown",
                return_value={**fake_peer, "status": "skipped", "skip_reason": "off"},
            ),
        ):
            summary = run_peer_review_manifest(
                manifest_path=manifest,
                output_dir=tmp_path / "out",
                run_native_text=True,
                run_pdfplumber=False,
                run_markitdown=False,
                run_paddle=False,
                limit=2,
            )

        assert summary["total"] == 2


# ---------------------------------------------------------------------------
# CLI peer commands
# ---------------------------------------------------------------------------


def test_peer_run_help() -> None:
    result = CliRunner().invoke(cli_app, ["peer", "run", "--help"])
    assert result.exit_code == 0
    assert "--manifest" in result.output
    assert "--vlm-base-url" in result.output
    assert "--paddle-vl" in result.output
    assert "MarkItDown OCR" in result.output


def test_peer_run_forwards_service_peer_options(tmp_path: Path, monkeypatch: Any) -> None:
    captured: dict[str, Any] = {}

    def fake_run_peer_review_manifest(**kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return {"status": "ok"}

    import gwanbo_ocr.peers as peers

    monkeypatch.setattr(peers, "run_peer_review_manifest", fake_run_peer_review_manifest)

    result = CliRunner().invoke(
        cli_app,
        [
            "peer",
            "run",
            "--manifest",
            str(tmp_path / "manifest.jsonl"),
            "--output",
            str(tmp_path / "out"),
            "--markitdown-ocr-llm",
            "--markitdown-service-url",
            "http://127.0.0.1:8081",
            "--markitdown-llm-base-url",
            "http://127.0.0.1:8000/v1",
            "--markitdown-llm-model",
            "Qwen/Qwen3.6-35B-A3B-FP8",
            "--paddle-service-url",
            "http://127.0.0.1:8082",
            "--paddle-vl-service-url",
            "http://127.0.0.1:8082",
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["run_markitdown_ocr_llm"] is True
    assert captured["markitdown_service_url"] == "http://127.0.0.1:8081"
    assert captured["markitdown_llm_base_url"] == "http://127.0.0.1:8000/v1"
    assert captured["markitdown_llm_model"] == "Qwen/Qwen3.6-35B-A3B-FP8"
    assert captured["paddle_service_url"] == "http://127.0.0.1:8082"
    assert captured["paddle_vl_service_url"] == "http://127.0.0.1:8082"


def test_peer_score_help() -> None:
    result = CliRunner().invoke(cli_app, ["peer", "score", "--help"])
    assert result.exit_code == 0
    assert "--review-dir" in result.output


def test_peer_run_end_to_end(tmp_path: Path) -> None:
    pdf = _write_pdf(tmp_path / "doc.pdf")
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps({"id": "t1", "pdf_path": str(pdf)}) + "\n", encoding="utf-8")
    output = tmp_path / "out"

    fake_peer = {
        "status": "ok",
        "method": "m",
        "text_extractable": True,
        "text_chars": 5,
        "sample_text": "hi",
        "error": None,
    }
    with (
        patch("gwanbo_ocr.peers.extract_native_text", return_value=fake_peer),
        patch(
            "gwanbo_ocr.peers.extract_pdfplumber",
            return_value={**fake_peer, "status": "skipped", "skip_reason": "off"},
        ),
        patch(
            "gwanbo_ocr.peers.extract_markitdown",
            return_value={**fake_peer, "status": "skipped", "skip_reason": "off"},
        ),
    ):
        result = CliRunner().invoke(
            cli_app,
            [
                "peer",
                "run",
                "--manifest",
                str(manifest),
                "--output",
                str(output),
                "--skip-pdfplumber",
                "--skip-markitdown",
            ],
        )

    assert result.exit_code == 0, result.output
    assert (output / "summary.json").exists()


def test_aggregate_peer_scores_reads_sidecar_scores(tmp_path: Path) -> None:
    review_dir = tmp_path / "review"
    items_dir = review_dir / "items"
    items_dir.mkdir(parents=True)
    sidecar = items_dir / "a.json"
    sidecar.write_text(
        json.dumps(
            {
                "score": {
                    "native_text": {"critical_token_f1": 0.9},
                    "vlm_ocr": {"critical_token_f1": 0.7},
                    "ranked_by_f1": ["native_text", "vlm_ocr"],
                    "reference_tokens": ["x"],
                }
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (review_dir / "metadata.json").write_text(
        json.dumps(
            {
                "a": {
                    "best_text_method": "native_text",
                    "decision": {"needs_ocr": False},
                    "sidecar_path": str(sidecar),
                }
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    report = aggregate_peer_scores(review_dir)
    assert report["total"] == 1
    assert report["by_best_method"]["native_text"] == 1
    assert report["avg_critical_token_f1_by_method"]["native_text"] == 0.9
