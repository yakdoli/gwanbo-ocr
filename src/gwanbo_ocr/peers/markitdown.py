from __future__ import annotations

from pathlib import Path
from typing import Any

from ._helpers import SAMPLE_CHARS_DEFAULT, _normalize, _skipped, _with_timeout

_MarkItDown: Any
try:
    from markitdown import MarkItDown as _MarkItDown  # type: ignore[import-not-found]
except ImportError:
    _MarkItDown = None


def extract_markitdown(
    pdf_path: Path,
    *,
    sample_chars: int = SAMPLE_CHARS_DEFAULT,
    timeout_seconds: int = 30,
) -> dict[str, Any]:
    """Extract text using MarkItDown.convert()."""
    MarkItDown = _MarkItDown
    if MarkItDown is None:
        return _skipped("markitdown is not installed", method="MarkItDown.convert")

    def _run() -> dict[str, Any]:
        converter = MarkItDown(enable_plugins=False)
        converted = converter.convert(str(pdf_path))
        text = _normalize(str(getattr(converted, "text_content", "") or ""))
        return {
            "status": "ok",
            "method": "MarkItDown.convert",
            "text_extractable": bool(text),
            "text_chars": len(text),
            "sample_text": text[:sample_chars],
            "error": None,
        }

    return _with_timeout(_run, timeout_seconds, method="MarkItDown.convert")
