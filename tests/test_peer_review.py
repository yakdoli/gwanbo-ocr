"""Tests for the peer review extraction module."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gwanbo_ocr.cli import app as cli_app
from gwanbo_ocr.peer_review import (
    analyze_pdf_peer_review,
    decide_extraction,
    extract_markitdown,
    extract_native_text,
    extract_pdfplumber,
    extract_vlm_ocr,
    review_extraction_peers,
    run_peer_review_manifest,
    score_against_metadata,
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
# Individual peer extractors
# ---------------------------------------------------------------------------


class TestExtractNativeText:
    def test_extracts_text_from_valid_pdf(self, tmp_path: Path) -> None:
        pdf = _write_pdf(tmp_path / "doc.pdf")
        result = extract_native_text(pdf, max_pages=1)

        assert result["status"] == "ok"
        assert result["method"] == "pypdf.extract_text"
        assert isinstance(result["text_chars"], int)
        assert isinstance(result["text_extractable"], bool)

    def test_returns_error_for_missing_file(self, tmp_path: Path) -> None:
        result = extract_native_text(tmp_path / "nonexistent.pdf", max_pages=1)
        assert result["status"] in {"error", "ok"}

    def test_sample_text_capped_at_sample_chars(self, tmp_path: Path) -> None:
        pdf = _write_pdf(tmp_path / "doc.pdf")
        result = extract_native_text(pdf, max_pages=1, sample_chars=5)
        assert len(result["sample_text"]) <= 5


class TestExtractPdfplumber:
    def test_extracts_text_when_pdfplumber_available(self, tmp_path: Path) -> None:
        try:
            import pdfplumber  # noqa: F401
        except ImportError:
            pytest.skip("pdfplumber not installed")

        pdf = _write_pdf(tmp_path / "doc.pdf")
        result = extract_pdfplumber(pdf, max_pages=1)

        assert result["method"] == "pdfplumber.extract_text"
        assert result["status"] in {"ok", "error"}
        assert isinstance(result["text_chars"], int)

    def test_skipped_when_pdfplumber_missing(self, tmp_path: Path) -> None:
        pdf = _write_pdf(tmp_path / "doc.pdf")
        with patch("gwanbo_ocr.peer_review._pdfplumber", None):
            result = extract_pdfplumber(pdf)
        assert result["status"] == "skipped"
        assert "pdfplumber is not installed" in result["skip_reason"]


class TestExtractMarkitdown:
    def test_skipped_when_markitdown_missing(self, tmp_path: Path) -> None:
        pdf = _write_pdf(tmp_path / "doc.pdf")
        with patch("gwanbo_ocr.peer_review._MarkItDown", None):
            result = extract_markitdown(pdf)
        assert result["status"] == "skipped"

    def test_calls_markitdown_when_available(self, tmp_path: Path) -> None:
        pdf = _write_pdf(tmp_path / "doc.pdf")

        class FakeResult:
            text_content = "관보 제목 내용"

        class FakeMarkItDown:
            def __init__(self, **_kwargs: Any) -> None:
                pass

            def convert(self, _path: str) -> FakeResult:
                return FakeResult()

        with patch("gwanbo_ocr.peer_review._MarkItDown", FakeMarkItDown):
            result = extract_markitdown(pdf, sample_chars=100)

        assert result["status"] == "ok"
        assert result["text_chars"] > 0
        assert "관보" in result["sample_text"]

    def test_records_error_on_exception(self, tmp_path: Path) -> None:
        pdf = _write_pdf(tmp_path / "doc.pdf")

        class FailingMarkItDown:
            def __init__(self, **_kwargs: Any) -> None:
                pass

            def convert(self, _path: str) -> None:
                raise RuntimeError("conversion failed")

        with patch("gwanbo_ocr.peer_review._MarkItDown", FailingMarkItDown):
            result = extract_markitdown(pdf)

        assert result["status"] == "error"
        assert "conversion failed" in result["error"]


class TestExtractVlmOcr:
    def _make_runner(self, text: str = "VLM transcribed text") -> Any:
        from gwanbo_ocr.runners.base import TranscriptionResult

        runner = MagicMock()
        runner.model = "test-vlm"
        runner.transcribe.return_value = TranscriptionResult.from_payload(
            {"text": text}, backend="vlm"
        )
        return runner

    def test_renders_and_transcribes(self, tmp_path: Path) -> None:
        try:
            import fitz  # noqa: F401
        except ImportError:
            pytest.skip("PyMuPDF not installed")

        pdf = _write_pdf(tmp_path / "doc.pdf")
        runner = self._make_runner("관보 VLM 결과")
        result = extract_vlm_ocr(
            pdf,
            image_dir=tmp_path / "images",
            runner=runner,
            max_pages=1,
            dpi=72,
        )

        assert result["status"] == "ok"
        assert result["text_chars"] > 0
        assert "관보" in result["sample_text"]
        assert result["model"] == "test-vlm"

    def test_error_when_render_fails(self, tmp_path: Path) -> None:
        pdf = tmp_path / "missing.pdf"
        runner = self._make_runner()
        with patch("gwanbo_ocr.peer_review._render_pages", return_value={"status": "error", "error": "render failed"}):
            result = extract_vlm_ocr(pdf, image_dir=tmp_path / "img", runner=runner)
        assert result["status"] == "error"


# ---------------------------------------------------------------------------
# Review and decision logic
# ---------------------------------------------------------------------------


class TestReviewExtractionPeers:
    def _peers(self) -> dict[str, dict[str, Any]]:
        return {
            "native_text": {"status": "ok", "text_chars": 50, "text_extractable": True, "sample_text": "native text sample"},
            "pdfplumber": {"status": "ok", "text_chars": 60, "text_extractable": True, "sample_text": "pdfplumber sample text"},
            "markitdown": {"status": "error", "text_chars": 0, "text_extractable": False, "error": "bad convert"},
            "vlm_ocr": {"status": "ok", "text_chars": 200, "text_extractable": True, "sample_text": "vlm ocr output from the document"},
        }

    def test_chooses_highest_char_count(self) -> None:
        review = review_extraction_peers(self._peers())
        assert review["best_text_method"] == "vlm_ocr"

    def test_warns_about_failed_peers(self) -> None:
        review = review_extraction_peers(self._peers())
        assert any("markitdown failed" in w for w in review["warnings"])

    def test_pairwise_similarity_computed(self) -> None:
        review = review_extraction_peers(self._peers())
        assert isinstance(review["pairwise_sample_similarity"], dict)

    def test_warns_when_no_text_produced(self) -> None:
        peers = {
            "native_text": {"status": "error", "text_chars": 0, "error": "fail"},
        }
        review = review_extraction_peers(peers)
        assert any("no successful" in w for w in review["warnings"])

    def test_peer_summaries_include_all_methods(self) -> None:
        review = review_extraction_peers(self._peers())
        assert set(review["peer_summaries"]) == {"native_text", "pdfplumber", "markitdown", "vlm_ocr"}


class TestDecideExtraction:
    def test_text_layer_preferred_when_available(self) -> None:
        peers = {
            "native_text": {"status": "ok", "text_chars": 100, "text_extractable": True},
            "vlm_ocr": {"status": "ok", "text_chars": 200, "text_extractable": True},
        }
        review = {"best_text_method": "vlm_ocr"}
        decision = decide_extraction(peers, review)
        assert decision["text_extractable"] is True
        assert decision["needs_ocr"] is False

    def test_ocr_needed_when_no_text_layer(self) -> None:
        peers = {
            "native_text": {"status": "ok", "text_chars": 0, "text_extractable": False},
            "vlm_ocr": {"status": "ok", "text_chars": 150, "text_extractable": True},
        }
        review = {"best_text_method": "vlm_ocr"}
        decision = decide_extraction(peers, review)
        assert decision["needs_ocr"] is True
        assert decision["text_extractable"] is True

    def test_not_extractable_when_all_fail(self) -> None:
        peers = {
            "native_text": {"status": "error", "text_chars": 0},
            "vlm_ocr": {"status": "error", "text_chars": 0},
        }
        review = {"best_text_method": None}
        decision = decide_extraction(peers, review)
        assert decision["text_extractable"] is False
        assert decision["needs_ocr"] is True


class TestScoreAgainstMetadata:
    def test_critical_token_f1_computed(self) -> None:
        peers = {
            "native_text": {
                "status": "ok",
                "sample_text": "기획재정부 예산 공고",
                "text_chars": 10,
            },
        }
        metadata = {"title": "예산 공고", "agency": "기획재정부", "category": "공고"}
        scores = score_against_metadata(peers, metadata)

        assert "native_text" in scores
        assert scores["native_text"]["status"] == "ok"
        assert scores["native_text"]["critical_token_f1"] is not None
        assert 0.0 <= scores["native_text"]["critical_token_f1"] <= 1.0

    def test_ranked_by_f1_orders_methods(self) -> None:
        peers = {
            "method_a": {"status": "ok", "sample_text": "기획재정부 공고", "text_chars": 10},
            "method_b": {"status": "ok", "sample_text": "무관한 텍스트 irrelevant text", "text_chars": 20},
        }
        metadata = {"title": "기획재정부 예산 공고", "agency": "기획재정부"}
        scores = score_against_metadata(peers, metadata)
        assert scores["ranked_by_f1"][0] == "method_a"

    def test_error_peer_has_null_f1(self) -> None:
        peers = {
            "failed_method": {"status": "error", "text_chars": 0},
        }
        scores = score_against_metadata(peers, {"title": "test"})
        assert scores["failed_method"]["critical_token_f1"] is None


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
            patch("gwanbo_ocr.peer_review.extract_native_text", fake_native),
            patch("gwanbo_ocr.peer_review.extract_pdfplumber", return_value={"status": "skipped", "text_chars": 0, "skip_reason": "off", "text_extractable": False, "sample_text": "", "error": None}),
            patch("gwanbo_ocr.peer_review.extract_markitdown", return_value={"status": "skipped", "text_chars": 0, "skip_reason": "off", "text_extractable": False, "sample_text": "", "error": None}),
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
            patch("gwanbo_ocr.peer_review.extract_native_text", return_value=fake_peer),
            patch("gwanbo_ocr.peer_review.extract_pdfplumber", return_value={"status": "skipped", "text_chars": 0, "skip_reason": "off", "text_extractable": False, "sample_text": "", "error": None}),
            patch("gwanbo_ocr.peer_review.extract_markitdown", return_value={"status": "skipped", "text_chars": 0, "skip_reason": "off", "text_extractable": False, "sample_text": "", "error": None}),
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
        failed = {"status": "error", "text_chars": 0, "text_extractable": False, "sample_text": "", "error": "oops", "method": "x"}

        with (
            patch("gwanbo_ocr.peer_review.extract_native_text", return_value=failed),
            patch("gwanbo_ocr.peer_review.extract_pdfplumber", return_value={**failed, "status": "skipped", "skip_reason": "off"}),
            patch("gwanbo_ocr.peer_review.extract_markitdown", return_value={**failed, "status": "skipped", "skip_reason": "off"}),
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
            patch("gwanbo_ocr.peer_review.extract_native_text", return_value=fake_peer),
            patch("gwanbo_ocr.peer_review.extract_pdfplumber", return_value={**fake_peer, "status": "skipped", "skip_reason": "off"}),
            patch("gwanbo_ocr.peer_review.extract_markitdown", return_value={**fake_peer, "status": "skipped", "skip_reason": "off"}),
        ):
            summary = run_peer_review_manifest(
                manifest_path=manifest,
                output_dir=output_dir,
                run_native_text=True,
                run_pdfplumber=False,
                run_markitdown=False,
                run_paddle=False,
                workers=1,
            )

        assert summary["processed"] == 1
        assert summary["total"] == 1
        assert (output_dir / "metadata.json").exists()
        assert (output_dir / "summary.json").exists()

        index = json.loads((output_dir / "metadata.json").read_text(encoding="utf-8"))
        assert "item-001" in index

    def test_skips_existing_sidecars_when_not_forced(self, tmp_path: Path) -> None:
        pdf = _write_pdf(tmp_path / "doc.pdf")
        manifest = tmp_path / "manifest.jsonl"
        self._write_manifest(manifest, [{"id": "skip-me", "pdf_path": str(pdf)}])
        output_dir = tmp_path / "out"

        # Pre-create sidecar so it's already present.
        sidecar = output_dir / "items" / "skip-me.json"
        sidecar.parent.mkdir(parents=True)
        sidecar.write_text("{}", encoding="utf-8")

        with patch("gwanbo_ocr.peer_review.extract_native_text", side_effect=AssertionError("should not be called")):
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
        self._write_manifest(manifest, [{"id": f"item-{i}", "pdf_path": str(p)} for i, p in enumerate(pdfs)])

        fake_peer = {"status": "error", "text_chars": 0, "text_extractable": False, "sample_text": "", "error": "x", "method": "t"}
        with (
            patch("gwanbo_ocr.peer_review.extract_native_text", return_value=fake_peer),
            patch("gwanbo_ocr.peer_review.extract_pdfplumber", return_value={**fake_peer, "status": "skipped", "skip_reason": "off"}),
            patch("gwanbo_ocr.peer_review.extract_markitdown", return_value={**fake_peer, "status": "skipped", "skip_reason": "off"}),
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


class TestPeerCli:
    def test_peer_run_help(self) -> None:
        result = CliRunner().invoke(cli_app, ["peer", "run", "--help"])
        assert result.exit_code == 0
        assert "--manifest" in result.output
        assert "--vlm-base-url" in result.output

    def test_peer_score_help(self) -> None:
        result = CliRunner().invoke(cli_app, ["peer", "score", "--help"])
        assert result.exit_code == 0
        assert "--review-dir" in result.output

    def test_peer_run_end_to_end(self, tmp_path: Path) -> None:
        pdf = _write_pdf(tmp_path / "doc.pdf")
        manifest = tmp_path / "manifest.jsonl"
        manifest.write_text(json.dumps({"id": "t1", "pdf_path": str(pdf)}) + "\n", encoding="utf-8")
        output = tmp_path / "out"

        fake_peer = {"status": "ok", "method": "m", "text_extractable": True, "text_chars": 5, "sample_text": "hi", "error": None}
        with (
            patch("gwanbo_ocr.peer_review.extract_native_text", return_value=fake_peer),
            patch("gwanbo_ocr.peer_review.extract_pdfplumber", return_value={**fake_peer, "status": "skipped", "skip_reason": "off"}),
            patch("gwanbo_ocr.peer_review.extract_markitdown", return_value={**fake_peer, "status": "skipped", "skip_reason": "off"}),
        ):
            result = CliRunner().invoke(
                cli_app,
                ["peer", "run", "--manifest", str(manifest), "--output", str(output),
                 "--skip-pdfplumber", "--skip-markitdown"],
            )

        assert result.exit_code == 0, result.output
        assert (output / "summary.json").exists()
