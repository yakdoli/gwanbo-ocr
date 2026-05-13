from __future__ import annotations

import re
from pathlib import Path


def test_service_dockerfiles_copy_service_path_helper() -> None:
    for dockerfile in ("Dockerfile.markitdown.server", "Dockerfile.paddleocr.server"):
        text = Path(dockerfile).read_text(encoding="utf-8")
        assert "COPY scripts ./scripts" in text


def test_convert_manifest_docs_prefer_public_input_option() -> None:
    paths = [
        Path("MARKITDOWN_QUICKSTART.md"),
        Path("MARKITDOWN_SUMMARY.md"),
        Path("docs/container-ocr-pipeline.md"),
        Path("docs/markitdown-architecture.md"),
    ]

    legacy_pattern = re.compile(r"gwanbo-ocr convert manifest[\s\\\n]+--manifest")
    for path in paths:
        text = path.read_text(encoding="utf-8")
        assert not legacy_pattern.search(text), path


def test_runtime_docs_match_cli_only_options() -> None:
    text = Path("docs/container-ocr-pipeline.md").read_text(encoding="utf-8")

    assert "gwanbo-ocr peer run \\\n  --manifest " in text
    assert "gwanbo-ocr strategy pipeline \\\n  --manifest " in text
    assert "gwanbo-ocr peer run \\\n  --input " not in text
    assert "gwanbo-ocr strategy pipeline \\\n  --input " not in text


def test_markitdown_api_docs_do_not_advertise_removed_fields() -> None:
    for path in (
        Path("MARKITDOWN_QUICKSTART.md"),
        Path("docs/markitdown-architecture.md"),
        Path("docs/markitdown-integration.md"),
    ):
        text = path.read_text(encoding="utf-8")
        assert "include_metadata" not in text, path
        assert "GET /` - API" not in text, path
        assert "curl http://localhost:8081/ | jq" not in text, path


def test_markitdown_docs_use_container_public_port() -> None:
    for path in (
        Path("MARKITDOWN_QUICKSTART.md"),
        Path("MARKITDOWN_SUMMARY.md"),
        Path("docs/markitdown-architecture.md"),
        Path("docs/markitdown-integration.md"),
    ):
        text = path.read_text(encoding="utf-8")
        assert "localhost:8080" not in text, path
        assert "python scripts/markitdown_server.py" not in text, path
        assert "\\  #" not in text, path
