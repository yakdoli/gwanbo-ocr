from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from gwanbo_ocr.pdf.classification import classify_pdf_document
from gwanbo_ocr.pdf.integrity import is_complete_pdf_bytes, validate_pdf_integrity
from gwanbo_ocr.pdf.layout import (
    classify_layout,
    extract_page_tables,
    normalize_table_rows,
    table_rows_to_json,
)
from gwanbo_ocr.pdf.text import analyze_pdf_text

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


class FakeTable:
    bbox = (0, 100, 300, 200)

    def extract(self):
        return [
            ["Name", "Name", ""],
            ["Alice", "Seoul", "10"],
            ["Bob", "Busan", "20"],
        ]


class FakePage:
    width = 300
    height = 200

    def extract_text(self):
        return "Name Name\nAlice Seoul 10\nBob Busan 20"

    def extract_words(self):
        return [
            {"x0": 10},
            {"x0": 12},
            {"x0": 14},
            {"x0": 16},
            {"x0": 140},
            {"x0": 142},
            {"x0": 144},
            {"x0": 146},
        ]

    def find_tables(self, table_settings=None):
        return [FakeTable()]


class PdfSessionBTests(unittest.TestCase):
    def test_text_pdf_extracts_without_required_optional_dependency(self) -> None:
        pdf_path = self._write_pdf("text.pdf", TEXT_PDF)

        result = analyze_pdf_text(pdf_path, include_sample=True, include_sha256=True)

        self.assertEqual(result["status"], "ok")
        self.assertTrue(result["text_extractable"])
        self.assertEqual(result["text_pages"], 1)
        self.assertIn("Hello PDF Text", result["sample_text"])
        self.assertEqual(len(result["sha256"]), 64)

    def test_integrity_checks_header_eof_and_hashes(self) -> None:
        pdf_path = self._write_pdf("complete.pdf", TEXT_PDF)
        broken_path = self._write_pdf("broken.pdf", b"%PDF-1.4\nno eof")

        valid = validate_pdf_integrity(pdf_path)
        broken = validate_pdf_integrity(broken_path, use_reader=False)

        self.assertTrue(is_complete_pdf_bytes(TEXT_PDF))
        self.assertEqual(valid["overall_status"], "pass")
        self.assertTrue(valid["valid"])
        self.assertEqual(valid["checks"]["pdf_header"]["status"], "pass")
        self.assertEqual(valid["checks"]["pdf_eof"]["status"], "pass")
        self.assertEqual(len(valid["sha256"]), 64)
        self.assertFalse(broken["valid"])
        self.assertEqual(broken["checks"]["pdf_eof"]["status"], "fail")

    def test_table_normalization_uses_stable_columns(self) -> None:
        rows = [[" Name\n", "Name", None], ["Alice", " Seoul ", 10], [None, None, None], ["Bob", "Busan"]]

        self.assertEqual(
            normalize_table_rows(rows),
            [["Name", "Name", ""], ["Alice", "Seoul", "10"], ["Bob", "Busan"]],
        )

        table = table_rows_to_json(rows, page_index=0, table_index=0, bbox=(0, 1, 2, 3))

        self.assertEqual(
            table["columns"],
            [
                {"key": "col_1", "label": "Name"},
                {"key": "col_2", "label": "Name"},
                {"key": "col_3", "label": ""},
            ],
        )
        self.assertEqual(table["records"][0], {"col_1": "Alice", "col_2": "Seoul", "col_3": "10"})
        self.assertEqual(table["records"][1], {"col_1": "Bob", "col_2": "Busan", "col_3": ""})
        self.assertEqual(table["bbox"], [0.0, 1.0, 2.0, 3.0])

    def test_layout_classification_and_fake_table_extraction(self) -> None:
        base_metric = {
            "page_index": 0,
            "text_chars": 120,
            "line_count": 10,
            "word_count": 20,
            "estimated_columns": 1,
            "text_quality": "readable",
            "form_score": 0.0,
            "table_count": 0,
            "table_chars": 0,
            "table_text_ratio": 0.0,
        }

        heavy = classify_layout([{**base_metric, "table_count": 1, "table_chars": 80}], [{"table_id": "t1"}])
        body = classify_layout([base_metric], [])
        columns = classify_layout([{**base_metric, "estimated_columns": 2}], [])
        tables = extract_page_tables(FakePage(), page_index=0, table_strategy="auto")

        self.assertEqual(heavy["document_class"], "table_heavy")
        self.assertEqual(body["document_class"], "body_text")
        self.assertEqual(columns["document_class"], "multi_column_text")
        self.assertEqual(len(tables), 1)
        self.assertEqual(tables[0]["records"][0], {"col_1": "Alice", "col_2": "Seoul", "col_3": "10"})
        self.assertEqual(tables[0]["alternate_strategies"], ["lines", "lines_strict", "text"])

    def test_document_classification_uses_integrity_and_text_metadata(self) -> None:
        pdf_path = self._write_pdf("classified.pdf", TEXT_PDF)

        result = classify_pdf_document(
            pdf_path,
            text_metadata={"status": "ok", "text_extractable": True, "total_chars": 42},
        )

        self.assertEqual(result["document_class"], "text_pdf")
        self.assertTrue(result["valid_pdf"])
        self.assertTrue(result["text_extractable"])

    def _write_pdf(self, name: str, payload: bytes) -> Path:
        path = Path(self._testMethodName)
        path.mkdir(exist_ok=True)
        pdf_path = path / name
        pdf_path.write_bytes(payload)
        self.addCleanup(lambda: pdf_path.unlink(missing_ok=True))
        self.addCleanup(lambda: path.rmdir() if path.exists() and not any(path.iterdir()) else None)
        return pdf_path


if __name__ == "__main__":
    unittest.main()
