"""Path allowlist helpers for local OCR service wrappers."""

from __future__ import annotations

import os
from pathlib import Path

DEFAULT_INPUT_ROOTS = (
    Path("/workspace/runs"),
    Path("/workspace/inputs"),
    Path("/root/peti/artifacts"),
    Path("/tmp"),
)
DEFAULT_OUTPUT_ROOTS = (
    Path("/workspace/runs"),
    Path("/workspace/outputs"),
    Path("/tmp"),
)


def resolve_allowed_path(path_text: str | Path) -> Path:
    """Resolve a service path and reject anything outside configured roots."""
    return resolve_allowed_input_path(path_text)


def resolve_allowed_input_path(path_text: str | Path) -> Path:
    """Resolve a service input path and reject anything outside configured input roots."""
    return _resolve_allowed_path(
        path_text,
        _allowed_roots(
            "GWANBO_SERVICE_INPUT_ROOTS",
            DEFAULT_INPUT_ROOTS,
            legacy_env_name="GWANBO_SERVICE_ALLOWED_ROOTS",
        ),
    )


def resolve_allowed_output_path(path_text: str | Path) -> Path:
    """Resolve a service output path and reject anything outside configured output roots."""
    return _resolve_allowed_path(
        path_text,
        _allowed_roots("GWANBO_SERVICE_OUTPUT_ROOTS", DEFAULT_OUTPUT_ROOTS),
    )


def _resolve_allowed_path(path_text: str | Path, roots: tuple[Path, ...]) -> Path:
    path = Path(path_text).expanduser().resolve(strict=False)
    if any(_is_relative_to(path, root) for root in roots):
        return path
    roots_text = ", ".join(str(root) for root in roots)
    raise ValueError(f"path is not under allowed roots: {path} (allowed: {roots_text})")


def _allowed_roots(
    env_name: str,
    default_roots: tuple[Path, ...],
    *,
    legacy_env_name: str | None = None,
) -> tuple[Path, ...]:
    configured = os.getenv(env_name)
    if configured is None and legacy_env_name is not None:
        configured = os.getenv(legacy_env_name)
    if configured:
        roots = [Path(item) for item in configured.split(os.pathsep) if item.strip()]
    else:
        roots = list(default_roots)
    return tuple(root.expanduser().resolve(strict=False) for root in roots)


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True
