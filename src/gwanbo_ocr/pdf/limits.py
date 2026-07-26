"""Shared resource limits and representative page sampling."""

from __future__ import annotations

import signal
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import FrameType
from typing import Final

DEFAULT_MAX_FILE_BYTES: Final = 256 * 1024 * 1024
DEFAULT_MAX_JSONL_LINE_BYTES: Final = 4 * 1024 * 1024
MAX_OCR_TEXT_CHARS: Final = 2 * 1024 * 1024
MAX_VLM_SPANS: Final = 1_000
MAX_VLM_SUGGESTIONS: Final = 1_000


@dataclass(frozen=True, slots=True)
class PdfResourceLimitError(ValueError):
    path: str
    size_bytes: int
    max_file_bytes: int

    def __str__(self) -> str:
        return f"PDF exceeds size limit: {self.size_bytes} > {self.max_file_bytes} bytes"


class PdfTimeoutError(TimeoutError):
    """Raised when bounded PDF analysis exceeds its deadline."""


def sample_page_indexes(total_pages: int, max_pages: int | None) -> tuple[int, ...]:
    if max_pages is None or max_pages >= total_pages:
        return tuple(range(total_pages))
    if max_pages <= 0 or total_pages == 0:
        return ()
    if max_pages == 1:
        return (0,)
    candidates = (0, total_pages // 2, total_pages - 1)
    selected = tuple(dict.fromkeys(candidates))
    if max_pages <= len(selected):
        return selected[:max_pages]
    remaining = (index for index in range(total_pages) if index not in selected)
    return (*selected, *tuple(remaining)[: max_pages - len(selected)])


def ensure_pdf_size(path: Path, max_file_bytes: int) -> None:
    size_bytes = path.stat().st_size
    if size_bytes > max_file_bytes:
        raise PdfResourceLimitError(str(path), size_bytes, max_file_bytes)


@contextmanager
def pdf_deadline(timeout_seconds: float) -> Iterator[None]:
    if (
        timeout_seconds <= 0
        or not hasattr(signal, "SIGALRM")
        or threading.current_thread() is not threading.main_thread()
    ):
        yield
        return

    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_timer = signal.getitimer(signal.ITIMER_REAL)

    def raise_timeout(_signum: int, _frame: FrameType | None) -> None:
        raise PdfTimeoutError("PDF analysis timed out")

    signal.signal(signal.SIGALRM, raise_timeout)
    signal.setitimer(signal.ITIMER_REAL, timeout_seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, *previous_timer)
        signal.signal(signal.SIGALRM, previous_handler)
