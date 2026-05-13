from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from gwanbo_ocr.peers import (
    decide_extraction,
    review_extraction_peers,
    score_against_metadata,
)

# ---------------------------------------------------------------------------
# Review and decision logic
# ---------------------------------------------------------------------------


class TestReviewExtractionPeers:
    def _peers(self) -> dict[str, dict]:
        return {
            "native_text": {
                "status": "ok",
                "text_chars": 50,
                "text_extractable": True,
                "sample_text": "native text sample",
            },
            "pdfplumber": {
                "status": "ok",
                "text_chars": 60,
                "text_extractable": True,
                "sample_text": "pdfplumber sample text",
            },
            "markitdown": {
                "status": "error",
                "text_chars": 0,
                "text_extractable": False,
                "error": "bad convert",
            },
            "vlm_ocr": {
                "status": "ok",
                "text_chars": 200,
                "text_extractable": True,
                "sample_text": "vlm ocr output from the document",
            },
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
        assert set(review["peer_summaries"]) == {
            "native_text",
            "pdfplumber",
            "markitdown",
            "vlm_ocr",
        }


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
            "method_b": {
                "status": "ok",
                "sample_text": "무관한 텍스트 irrelevant text",
                "text_chars": 20,
            },
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
