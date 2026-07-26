from __future__ import annotations

import tomllib
from pathlib import Path


def test_pdf_extra_declares_direct_pillow_dependency() -> None:
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    payload = tomllib.loads(pyproject.read_text(encoding="utf-8"))

    requirements = payload["project"]["optional-dependencies"]["pdf"]
    assert any(requirement.lower().startswith("pillow") for requirement in requirements)
