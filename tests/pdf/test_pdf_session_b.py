from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from gwanbo_ocr.pdf.classification import classify_pdf_document
from gwanbo_ocr.pdf.integrity import is_complete_pdf_bytes, validate_pdf_integrity
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


class PdfSessionBTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)
        self._pdf_dir = Path(self._tmpdir.name)

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
        pdf_path = self._pdf_dir / name
        pdf_path.parent.mkdir(parents=True, exist_ok=True)
        pdf_path.write_bytes(payload)
        return pdf_path


if __name__ == "__main__":
    unittest.main()
