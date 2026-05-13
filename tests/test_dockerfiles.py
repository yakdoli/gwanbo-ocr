from __future__ import annotations

from pathlib import Path


def test_service_dockerfiles_copy_service_path_helper() -> None:
    for dockerfile in ("Dockerfile.markitdown.server", "Dockerfile.paddleocr.server"):
        text = Path(dockerfile).read_text(encoding="utf-8")
        assert "COPY scripts ./scripts" in text
