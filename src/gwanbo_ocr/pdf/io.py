"""Small IO helpers shared by PDF analysis modules."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO

from .limits import DEFAULT_MAX_JSONL_LINE_BYTES


@dataclass(frozen=True, slots=True)
class UnsafePathError(ValueError):
    path: str
    reason: str

    def __str__(self) -> str:
        return f"unsafe path {self.path!r}: {self.reason}"


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
    return (
        path for path in sorted(root.rglob("*.pdf")) if path.is_file() and not path.is_symlink()
    )


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
    with _atomic_text_writer(path) as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=indent, default=str)


def read_jsonl(path: Path | str) -> Iterator[Any]:
    """Yield decoded JSONL records, skipping blank lines."""
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def iter_jsonl_records(
    path: Path | str, *, max_line_bytes: int = DEFAULT_MAX_JSONL_LINE_BYTES
) -> Iterator[tuple[int, Any]]:
    """Yield ``(line_number, payload)`` pairs from a JSONL file.

    Malformed lines are returned as error dictionaries instead of aborting the
    whole batch. Batch pipeline stages use this when they need to record row
    errors and keep processing the rest of a large manifest.
    """
    with Path(path).open("rb") as handle:
        line_number = 0
        while raw_line := handle.readline(max_line_bytes + 1):
            line_number += 1
            if len(raw_line) > max_line_bytes:
                while not raw_line.endswith(b"\n"):
                    raw_line = handle.readline(max_line_bytes + 1)
                    if not raw_line:
                        break
                yield (
                    line_number,
                    {
                        "schema_version": "jsonl-error/v1",
                        "status": "error",
                        "error": "jsonl_line_too_large",
                        "line_number": line_number,
                    },
                )
                continue
            stripped = raw_line.strip()
            if not stripped:
                continue
            try:
                yield line_number, json.loads(stripped)
            except (json.JSONDecodeError, UnicodeDecodeError) as error:
                message = error.msg if isinstance(error, json.JSONDecodeError) else "invalid_utf8"
                yield (
                    line_number,
                    {
                        "schema_version": "jsonl-error/v1",
                        "status": "error",
                        "error": f"malformed_jsonl:{message}",
                        "line_number": line_number,
                    },
                )


def write_jsonl_atomic(path: Path | str, rows: Iterable[Any]) -> None:
    """Write JSONL rows using an atomic replace."""
    with _atomic_text_writer(path) as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, default=str))
            handle.write("\n")


def write_json_lines_atomic(path: Path | str, rows: Iterable[str]) -> None:
    with _atomic_text_writer(path) as handle:
        for row in rows:
            handle.write(row)
            handle.write("\n")


@contextmanager
def _atomic_text_writer(path: Path | str) -> Iterator[TextIO]:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    replaced = False
    try:
        file_descriptor, temp_name = tempfile.mkstemp(
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
            text=True,
        )
        temp_path = Path(temp_name)
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as handle:
            yield handle
            handle.flush()
            os.fsync(handle.fileno())
        temp_path.replace(target)
        replaced = True
    finally:
        if temp_path is not None and not replaced:
            temp_path.unlink(missing_ok=True)


def file_digest(
    path: Path | str, algorithm: str = "sha256", *, chunk_size: int = 1024 * 1024
) -> str:
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


def resolve_pdf_path(
    row: Mapping[str, Any],
    *,
    base_dir: Path | str | None = None,
    peti_root: Path | str | None = None,
    trusted_root: Path | str | None = None,
) -> Path:
    """Resolve the canonical PDF path for a manifest-like row.

    The project manifest stores both a repository-relative ``pdf_path`` and an
    absolute ``pdf_abs_path``. Runtime stages should prefer ``pdf_abs_path`` so
    they do not accidentally resolve `/root/peti` artifacts relative to the
    current working directory.
    """
    for key in ("pdf_abs_path", "pdf_path", "path"):
        value = str(row.get(key) or "").strip()
        if not value:
            continue
        path = Path(value)
        if path.is_absolute():
            return _trusted_path(path, trusted_root) if trusted_root is not None else path
        if key == "pdf_path" and peti_root is not None and value.startswith("artifacts/"):
            candidate = (Path(peti_root) / path).resolve(strict=False)
        else:
            candidate = resolve_path(path, base_dir=base_dir)
        return _trusted_path(candidate, trusted_root) if trusted_root is not None else candidate
    return Path("")


def pdf_key_from_row(row: Mapping[str, Any], *, fallback: str = "unknown") -> str:
    """Return a stable key for a manifest/profile row."""
    for key in ("pdf_key", "metadata_key", "sample_id", "id"):
        value = str(row.get(key) or "").strip()
        if value:
            return _safe_relative_path(value).as_posix()
    path_text = str(row.get("pdf_path") or row.get("pdf_abs_path") or "").strip()
    if path_text:
        return _safe_relative_path(Path(path_text).with_suffix("").as_posix()).as_posix()
    return fallback


def safe_output_path(root: Path | str, relative_path: Path | str) -> Path:
    root_path = Path(root).resolve(strict=False)
    candidate = (root_path / _safe_relative_path(relative_path)).resolve(strict=False)
    if not candidate.is_relative_to(root_path):
        raise UnsafePathError(str(relative_path), "escapes output root")
    return candidate


def _safe_relative_path(path: Path | str) -> Path:
    candidate = Path(str(path).replace("\\", "/"))
    if candidate.is_absolute() or any(part == ".." for part in candidate.parts):
        raise UnsafePathError(str(path), "must be a relative path without parent traversal")
    cleaned = candidate.as_posix().strip("/")
    if not cleaned:
        raise UnsafePathError(str(path), "path is empty")
    return Path(cleaned)


def _trusted_path(path: Path, trusted_root: Path | str | None) -> Path:
    if trusted_root is None:
        return path
    if path.is_symlink():
        raise UnsafePathError(str(path), "symlinks are not allowed")
    root = Path(trusted_root).resolve(strict=False)
    resolved = path.resolve(strict=False)
    if not resolved.is_relative_to(root):
        raise UnsafePathError(str(path), f"outside trusted root {root}")
    return resolved


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
