"""Tests for render.py that do not require PyMuPDF to be installed.

Rendering functions that call PyMuPDF (_import_fitz) are exercised only
when the optional [pdf] extra is available; otherwise they are skipped.
"""

from __future__ import annotations

import base64
import io
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gwanbo_ocr.render import RenderedPage, _resolve_page_index

# ---------------------------------------------------------------------------
# RenderedPage — no external deps
# ---------------------------------------------------------------------------


def _fake_image(width: int = 100, height: int = 80) -> MagicMock:
    """Return a MagicMock that behaves like a minimal Pillow image for tests."""
    img = MagicMock()
    img.size = (width, height)

    def _save(buf: io.BytesIO, format: str) -> None:
        buf.write(b"\x89PNG\r\n\x1a\n" + b"\x00" * 20)

    img.save = _save
    return img


class TestRenderedPage:
    def test_page_number_is_one_based(self) -> None:
        page = RenderedPage(image=_fake_image(), page_index=0, width=100, height=80, dpi=200)
        assert page.page_number == 1

        page2 = RenderedPage(image=_fake_image(), page_index=4, width=100, height=80, dpi=200)
        assert page2.page_number == 5

    def test_to_png_bytes_returns_bytes(self) -> None:
        page = RenderedPage(image=_fake_image(), page_index=0, width=100, height=80, dpi=200)
        result = page.to_png_bytes()
        assert isinstance(result, bytes)
        assert len(result) > 0

    def test_to_base64_is_ascii_encoded_png(self) -> None:
        page = RenderedPage(image=_fake_image(), page_index=0, width=100, height=80, dpi=200)
        b64 = page.to_base64()
        assert isinstance(b64, str)
        # Must be valid base64
        decoded = base64.b64decode(b64)
        assert len(decoded) > 0

    def test_to_data_url_format(self) -> None:
        page = RenderedPage(image=_fake_image(), page_index=0, width=100, height=80, dpi=200)
        url = page.to_data_url()
        assert url.startswith("data:image/png;base64,")


# ---------------------------------------------------------------------------
# _resolve_page_index helper
# ---------------------------------------------------------------------------


class TestResolvePageIndex:
    def test_defaults_to_zero(self) -> None:
        assert _resolve_page_index(None, None) == 0

    def test_page_index_passthrough(self) -> None:
        assert _resolve_page_index(2, None) == 2

    def test_page_number_converted_to_zero_based(self) -> None:
        assert _resolve_page_index(None, 1) == 0
        assert _resolve_page_index(None, 5) == 4

    def test_page_number_takes_priority_over_page_index(self) -> None:
        assert _resolve_page_index(99, page_number=3) == 2

    def test_page_number_less_than_one_raises(self) -> None:
        with pytest.raises(ValueError, match="one-based"):
            _resolve_page_index(None, 0)

    def test_negative_page_index_raises(self) -> None:
        with pytest.raises(ValueError, match="zero-based"):
            _resolve_page_index(-1, None)


# ---------------------------------------------------------------------------
# render_pdf_page — requires PyMuPDF + Pillow
# ---------------------------------------------------------------------------

try:
    import fitz  # noqa: F401
    from PIL import Image  # noqa: F401

    _PYMUPDF_AVAILABLE = True
except ImportError:
    _PYMUPDF_AVAILABLE = False

TEXT_PDF_BYTES = b"""%PDF-1.4
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


@pytest.mark.skipif(not _PYMUPDF_AVAILABLE, reason="PyMuPDF not installed")
def test_render_pdf_page_returns_pillow_image() -> None:
    from gwanbo_ocr.render import render_pdf_page

    img = render_pdf_page(TEXT_PDF_BYTES, page_index=0, dpi=72)
    assert hasattr(img, "size")
    width, height = img.size
    assert width > 0 and height > 0


@pytest.mark.skipif(not _PYMUPDF_AVAILABLE, reason="PyMuPDF not installed")
def test_render_pdf_page_result_has_metadata() -> None:
    from gwanbo_ocr.render import render_pdf_page_result

    result = render_pdf_page_result(TEXT_PDF_BYTES, page_index=0, dpi=72)
    assert result.page_index == 0
    assert result.page_number == 1
    assert result.dpi == 72
    assert result.width > 0
    assert result.height > 0


@pytest.mark.skipif(not _PYMUPDF_AVAILABLE, reason="PyMuPDF not installed")
def test_render_page_to_png_bytes_is_valid_png() -> None:
    from gwanbo_ocr.render import render_page_to_png_bytes

    png = render_page_to_png_bytes(TEXT_PDF_BYTES, page_index=0, dpi=72)
    assert isinstance(png, bytes)
    assert png[:8] == b"\x89PNG\r\n\x1a\n"


@pytest.mark.skipif(not _PYMUPDF_AVAILABLE, reason="PyMuPDF not installed")
def test_render_invalid_dpi_raises() -> None:
    from gwanbo_ocr.render import render_pdf_page_result

    with pytest.raises(ValueError, match="dpi"):
        render_pdf_page_result(TEXT_PDF_BYTES, dpi=0)


@pytest.mark.skipif(not _PYMUPDF_AVAILABLE, reason="PyMuPDF not installed")
def test_page_count_returns_correct_value() -> None:
    from gwanbo_ocr.render import page_count

    assert page_count(TEXT_PDF_BYTES) == 1
