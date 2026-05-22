from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
VERIFY_SEQUENCE = [
    "ruff check src tests scripts",
    "ruff format --check src tests scripts",
    ".venv/bin/pyrefly check --summary=none",
    ".venv/bin/pytest -ra --tb=short",
]


def _pyproject() -> dict[str, Any]:
    return tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))


def _assert_commands_in_order(path: Path, commands: list[str]) -> None:
    text = path.read_text(encoding="utf-8")
    cursor = -1
    for command in commands:
        position = text.find(command)
        assert position > cursor, f"{path} missing or out of order: {command}"
        cursor = position


def test_dev_extra_includes_pyrefly() -> None:
    payload = _pyproject()
    dev_deps = payload["project"]["optional-dependencies"]["dev"]

    assert any(dep.startswith("pyrefly>=1.0") for dep in dev_deps)


def test_pyrefly_config_scopes_project() -> None:
    payload = _pyproject()
    config = payload["tool"]["pyrefly"]

    assert config["python-version"] == "3.12"
    assert config["project-includes"] == ["src", "tests", "scripts"]
    assert config["search-path"] == [".", "src"]


def test_verification_workflow_documents_pyrefly() -> None:
    strict_paths = [
        ROOT / "AGENTS.md",
        ROOT / ".opencode/commands/verify.md",
        ROOT / ".opencode/agents/test.md",
    ]
    mention_paths = [
        ROOT / "CLAUDE.md",
        ROOT / ".opencode/agents/review.md",
    ]

    for path in strict_paths:
        _assert_commands_in_order(path, VERIFY_SEQUENCE)

    for path in mention_paths:
        text = path.read_text(encoding="utf-8")
        assert "pyrefly check --summary=none" in text, path


def test_lint_workflow_includes_service_scripts() -> None:
    paths = [
        ROOT / "AGENTS.md",
        ROOT / "CLAUDE.md",
        ROOT / ".opencode/commands/verify.md",
        ROOT / ".opencode/agents/test.md",
        ROOT / ".opencode/agents/review.md",
        ROOT / ".opencode/skills/ruff-check/SKILL.md",
    ]

    for path in paths:
        text = path.read_text(encoding="utf-8")
        assert "ruff check src tests scripts" in text, path
