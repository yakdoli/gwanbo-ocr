"""Path allowlist helpers for local OCR service wrappers."""

from __future__ import annotations

import os
from pathlib import Path

DEFAULT_ALLOWED_ROOTS = (
    Path("/workspace/runs"),
    Path("/workspace/inputs"),
    Path("/workspace/outputs"),
    Path("/root/peti"),
    Path("/tmp"),
)


def resolve_allowed_path(path_text: str | Path) -> Path:
    """Resolve a service path and reject anything outside configured roots."""
    path = Path(path_text).expanduser().resolve(strict=False)
    roots = _allowed_roots()
    if any(_is_relative_to(path, root) for root in roots):
        return path
    roots_text = ", ".join(str(root) for root in roots)
    raise ValueError(f"path is not under allowed roots: {path} (allowed: {roots_text})")


def _allowed_roots() -> tuple[Path, ...]:
    configured = os.getenv("GWANBO_SERVICE_ALLOWED_ROOTS")
    if configured:
        roots = [Path(item) for item in configured.split(os.pathsep) if item.strip()]
    else:
        roots = list(DEFAULT_ALLOWED_ROOTS)
    return tuple(root.expanduser().resolve(strict=False) for root in roots)


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True
