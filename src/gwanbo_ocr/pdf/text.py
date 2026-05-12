"""Native PDF text extraction metadata helpers."""

from __future__ import annotations

import contextlib
import re
import signal
import threading
import zlib
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from typing import Any

from .io import (
    file_sha256,
    iter_pdf_paths,
    limited,
    normalize_text,
    pdf_key_from_path,
    write_json_atomic,
)

PdfReader: Any
try:  # Prefer pypdf, then the older PyPDF2 package.
    from pypdf import PdfReader as _PypdfReader  # type: ignore[import-not-found]

    PdfReader = _PypdfReader
except ImportError:  # pragma: no cover - depends on optional environment.
    try:
        from PyPDF2 import PdfReader as _PyPdf2Reader  # type: ignore[import-not-found]

        PdfReader = _PyPdf2Reader
    except ImportError:  # pragma: no cover - default in the kata environment.
        PdfReader = None


def analyze_pdf_text(
    pdf_path: Path | str,
    *,
    include_sample: bool = False,
    sample_chars: int = 1000,
    include_sha256: bool = False,
    max_pages: int | None = None,
    timeout_seconds: int = 30,
) -> dict[str, Any]:
    """Inspect whether a PDF has an extractable text layer.

    The primary path uses pypdf/PyPDF2 when available. A conservative built-in
    parser handles simple content streams so callers still get useful metadata
    in minimal environments.
    """
    path = Path(pdf_path)
    result: dict[str, Any] = {
        "path": str(path),
        "filename": path.name,
        "status": "ok",
        "text_extractable": False,
        "text_pages": 0,
        "total_chars": 0,
        "extraction_method": "pypdf/PyPDF2.extract_text"
        if PdfReader is not None
        else "builtin.pdf_text_stream_parser",
        "generated_at": datetime.now().isoformat(),
    }

    try:
        result["size_bytes"] = path.stat().st_size
    except OSError as exc:
        result.update({"status": "error", "error": str(exc)})
        return result

    if include_sha256:
        result["sha256"] = file_sha256(path)

    with _alarm_timeout(timeout_seconds, "PDF text analysis timed out"):
        try:
            if PdfReader is not None:
                _analyze_with_reader(path, result, include_sample, sample_chars, max_pages)
            else:
                _analyze_with_builtin_parser(path, result, include_sample, sample_chars, max_pages)
        except Exception as exc:  # noqa: BLE001 - PDF parsers use broad exception families.
            result.update({"status": "error", "error": str(exc), "text_extractable": False})

    return result


def extract_pdf_text(
    pdf_path: Path | str, *, max_pages: int | None = None, timeout_seconds: int = 30
) -> str:
    """Return normalized text extracted from a PDF, or an empty string."""
    metadata = analyze_pdf_text(
        pdf_path,
        include_sample=True,
        sample_chars=10_000_000,
        max_pages=max_pages,
        timeout_seconds=timeout_seconds,
    )
    return str(metadata.get("sample_text") or "")


def generate_text_sidecars(
    pdf_dir: Path | str,
    output_dir: Path | str,
    *,
    limit: int | None = None,
    include_sample: bool = False,
    sample_chars: int = 1000,
    include_sha256: bool = False,
    max_pages: int | None = None,
    timeout_seconds: int = 30,
) -> dict[str, Any]:
    """Analyze PDFs and write sidecar/index JSON without mutating item JSON."""
    root = Path(pdf_dir)
    output = Path(output_dir)
    item_output = output / "items"
    item_output.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {
        "pdf_dir": str(root),
        "output_dir": str(output),
        "started_at": datetime.now().isoformat(),
        "total_pdfs": 0,
        "processed": 0,
        "text_extractable": 0,
        "image_or_unextractable": 0,
        "errors": 0,
    }
    index: dict[str, dict[str, Any]] = {}

    for pdf_path in limited(iter_pdf_paths(root), limit):
        summary["total_pdfs"] += 1
        key = pdf_key_from_path(pdf_path, root)
        metadata = analyze_pdf_text(
            pdf_path,
            include_sample=include_sample,
            sample_chars=sample_chars,
            include_sha256=include_sha256,
            max_pages=max_pages,
            timeout_seconds=timeout_seconds,
        )
        metadata["pdf_key"] = key
        metadata["pdf_path"] = str(pdf_path)

        if metadata.get("status") == "error":
            summary["errors"] += 1
        elif metadata.get("text_extractable"):
            summary["text_extractable"] += 1
        else:
            summary["image_or_unextractable"] += 1

        summary["processed"] += 1
        index[key] = compact_text_metadata(metadata)
        write_json_atomic(item_output / f"{key}.json", metadata)

    summary["completed_at"] = datetime.now().isoformat()
    write_json_atomic(output / "metadata.json", index)
    write_json_atomic(output / "summary.json", summary)
    return summary


def compact_text_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """Drop large sample text for indexes or item-safe metadata."""
    return {key: value for key, value in metadata.items() if key != "sample_text"}


def _analyze_with_reader(
    path: Path,
    result: dict[str, Any],
    include_sample: bool,
    sample_chars: int,
    max_pages: int | None,
) -> None:
    reader = PdfReader(str(path))  # type: ignore[misc,operator]
    page_count = len(reader.pages)
    result["pages"] = page_count
    result["pdf_metadata"] = {
        str(key): str(value) for key, value in (reader.metadata or {}).items()
    }
    pages_to_scan = page_count if max_pages is None else min(page_count, max_pages)
    page_errors: list[dict[str, Any]] = []
    sample_parts: list[str] = []

    for index in range(pages_to_scan):
        try:
            text = normalize_text(reader.pages[index].extract_text() or "")
        except Exception as exc:  # noqa: BLE001
            page_errors.append({"page_index": index, "error": str(exc)})
            continue
        if text:
            result["text_pages"] += 1
            result["total_chars"] += len(text)
            if include_sample and len(" ".join(sample_parts)) < sample_chars:
                sample_parts.append(text)

    _finish_text_result(
        result, pages_to_scan, page_errors, sample_parts, include_sample, sample_chars
    )


def _analyze_with_builtin_parser(
    path: Path,
    result: dict[str, Any],
    include_sample: bool,
    sample_chars: int,
    max_pages: int | None,
) -> None:
    data = path.read_bytes()
    page_count = _count_pages(data)
    result["pages"] = page_count
    pages_to_scan = page_count if max_pages is None else min(page_count, max_pages)
    if page_count == 0 and max_pages != 0:
        pages_to_scan = 1

    text = normalize_text(_extract_text_from_pdf_bytes(data))
    if text and pages_to_scan:
        result["text_pages"] = 1
        result["total_chars"] = len(text)

    _finish_text_result(
        result, pages_to_scan, [], [text] if text else [], include_sample, sample_chars
    )


def _finish_text_result(
    result: dict[str, Any],
    pages_to_scan: int,
    page_errors: list[dict[str, Any]],
    sample_parts: list[str],
    include_sample: bool,
    sample_chars: int,
) -> None:
    result["scanned_pages"] = pages_to_scan
    if page_errors:
        result["page_errors"] = page_errors
        result["page_error_count"] = len(page_errors)
    result["text_extractable"] = result["text_pages"] > 0 and result["total_chars"] > 0
    if result["text_pages"]:
        result["avg_chars_per_text_page"] = round(result["total_chars"] / result["text_pages"], 2)
    if include_sample:
        result["sample_text"] = " ".join(part for part in sample_parts if part)[:sample_chars]


def _count_pages(data: bytes) -> int:
    text = data.decode("latin-1", errors="ignore")
    return len(re.findall(r"/Type\s*/Page\b", text))


def _extract_text_from_pdf_bytes(data: bytes) -> str:
    chunks: list[str] = []
    for stream in _iter_streams(data):
        chunks.extend(_extract_text_from_stream(stream))
    return normalize_text(" ".join(chunks))


def _iter_streams(data: bytes) -> Iterator[str]:
    for match in re.finditer(rb"stream\r?\n?(.*?)\r?\n?endstream", data, flags=re.DOTALL):
        payload = match.group(1).strip(b"\r\n")
        decoded = _maybe_flate_decode(payload)
        yield decoded.decode("latin-1", errors="ignore")


def _maybe_flate_decode(payload: bytes) -> bytes:
    try:
        return zlib.decompress(payload)
    except zlib.error:
        return payload


def _extract_text_from_stream(stream: str) -> list[str]:
    text: list[str] = []
    for literal in _iter_pdf_string_literals(stream):
        value = _decode_pdf_literal(literal)
        if value:
            text.append(value)
    return text


def _iter_pdf_string_literals(stream: str) -> Iterator[str]:
    index = 0
    while index < len(stream):
        if stream[index] != "(":
            index += 1
            continue
        index += 1
        depth = 1
        escaped = False
        chars: list[str] = []
        while index < len(stream) and depth:
            char = stream[index]
            if escaped:
                chars.append("\\" + char)
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == "(":
                depth += 1
                chars.append(char)
            elif char == ")":
                depth -= 1
                if depth:
                    chars.append(char)
            else:
                chars.append(char)
            index += 1
        if chars:
            yield "".join(chars)


def _decode_pdf_literal(value: str) -> str:
    result: list[str] = []
    index = 0
    escapes = {
        "n": "\n",
        "r": "\r",
        "t": "\t",
        "b": "\b",
        "f": "\f",
        "(": "(",
        ")": ")",
        "\\": "\\",
    }
    while index < len(value):
        char = value[index]
        if char != "\\":
            result.append(char)
            index += 1
            continue

        index += 1
        if index >= len(value):
            break
        escaped = value[index]
        if escaped in escapes:
            result.append(escapes[escaped])
            index += 1
            continue
        if escaped in "\n\r":
            index += 1
            if escaped == "\r" and index < len(value) and value[index] == "\n":
                index += 1
            continue
        if escaped in "01234567":
            octal = escaped
            index += 1
            while index < len(value) and len(octal) < 3 and value[index] in "01234567":
                octal += value[index]
                index += 1
            result.append(chr(int(octal, 8)))
            continue
        result.append(escaped)
        index += 1
    return "".join(result)


@contextlib.contextmanager
def _alarm_timeout(seconds: int, message: str) -> Iterator[None]:
    if (
        seconds <= 0
        or not hasattr(signal, "SIGALRM")
        or threading.current_thread() is not threading.main_thread()
    ):
        yield
        return

    previous_handler = signal.getsignal(signal.SIGALRM)

    def raise_timeout(_signum: int, _frame: Any) -> None:
        raise TimeoutError(message)

    signal.signal(signal.SIGALRM, raise_timeout)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous_handler)
