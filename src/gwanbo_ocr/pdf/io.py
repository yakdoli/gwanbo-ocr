"""Small IO helpers shared by PDF analysis modules."""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any


def ensure_directory(path: Path | str) -> Path:
    """Create and return a directory path."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def iter_pdf_paths(pdf_dir: Path | str) -> Iterator[Path]:
    """Yield PDF files below *pdf_dir* in stable order."""
    root = Path(pdf_dir)
    if not root.exists():
        return iter(())
    return (path for path in sorted(root.rglob("*.pdf")) if path.is_file())


def limited(paths: Iterable[Path], limit: int | None) -> Iterator[Path]:
    """Yield at most *limit* paths, or all paths when limit is ``None``."""
    for index, path in enumerate(paths):
        if limit is not None and index >= limit:
            break
        yield path


def read_json(path: Path | str) -> Any:
    """Read JSON, returning ``None`` for missing or malformed files."""
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def write_json_atomic(path: Path | str, payload: Any, *, indent: int = 2) -> None:
    """Write JSON with a same-directory temporary file and atomic replace."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp_path = target.with_name(f"{target.name}.{os.getpid()}.tmp")
    temp_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=indent, default=str),
        encoding="utf-8",
    )
    temp_path.replace(target)


def read_jsonl(path: Path | str) -> Iterator[Any]:
    """Yield decoded JSONL records, skipping blank lines."""
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl_atomic(path: Path | str, rows: Iterable[Any]) -> None:
    """Write JSONL rows using an atomic replace."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp_path = target.with_name(f"{target.name}.{os.getpid()}.tmp")
    with temp_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, default=str))
            handle.write("\n")
    temp_path.replace(target)


def file_digest(path: Path | str, algorithm: str = "sha256", *, chunk_size: int = 1024 * 1024) -> str:
    """Return a hex digest for *path* using hashlib *algorithm*."""
    digest = hashlib.new(algorithm)
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_sha256(path: Path | str) -> str:
    return file_digest(path, "sha256")


def file_md5(path: Path | str) -> str:
    return file_digest(path, "md5")


def normalize_text(text: str) -> str:
    """Collapse whitespace without changing text content otherwise."""
    return re.sub(r"\s+", " ", text).strip()


def resolve_path(path_text: str | Path, base_dir: Path | str | None = None) -> Path:
    """Resolve a possibly relative path against *base_dir* and cwd."""
    path = Path(path_text)
    if path.is_absolute():
        return path.resolve()

    candidates: list[Path] = []
    if base_dir is not None:
        candidates.append((Path(base_dir) / path).resolve())
    candidates.append((Path.cwd() / path).resolve())

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def relative_to_or_str(path: Path | str, root: Path | str) -> str:
    """Return path relative to root when possible, otherwise as a string."""
    candidate = Path(path)
    try:
        return candidate.relative_to(Path(root)).as_posix()
    except ValueError:
        return str(candidate)


def pdf_key_from_path(pdf_path: Path | str, pdf_dir: Path | str) -> str:
    """Return a stable extension-free key for a PDF below *pdf_dir*."""
    return Path(pdf_path).relative_to(Path(pdf_dir)).with_suffix("").as_posix()
