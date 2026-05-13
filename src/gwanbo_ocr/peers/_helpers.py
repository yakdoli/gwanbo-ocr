from __future__ import annotations

import re
import signal
from datetime import datetime
from pathlib import Path
from typing import Any

SAMPLE_CHARS_DEFAULT = 1200


def _skipped(reason: str, *, method: str) -> dict[str, Any]:
    return {
        "status": "skipped",
        "method": method,
        "skip_reason": reason,
        "text_extractable": False,
        "text_chars": 0,
        "sample_text": "",
        "error": None,
    }


def _error(message: str, *, method: str) -> dict[str, Any]:
    return {
        "status": "error",
        "method": method,
        "text_extractable": False,
        "text_chars": 0,
        "sample_text": "",
        "error": message,
    }


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _norm_for_sim(text: str) -> str:
    return re.sub(r"\W+", "", text).lower()[:2000]


def _with_timeout(fn: Any, timeout_seconds: int, *, method: str) -> dict[str, Any]:
    previous = None
    if timeout_seconds > 0:
        previous = signal.signal(signal.SIGALRM, _alarm_handler)
        signal.alarm(timeout_seconds)
    try:
        result = fn()
        return (
            result
            if isinstance(result, dict)
            else _error("method returned non-dict", method=method)
        )
    except Exception as exc:  # noqa: BLE001
        return _error(str(exc), method=method)
    finally:
        if timeout_seconds > 0:
            signal.alarm(0)
            if previous is not None:
                signal.signal(signal.SIGALRM, previous)


def _alarm_handler(_signum: int, _frame: Any) -> None:
    raise TimeoutError("peer review extraction timed out")


def _render_pages(
    pdf_path: Path,
    *,
    image_dir: Path,
    max_pages: int | None,
    dpi: int,
) -> dict[str, Any]:
    try:
        from gwanbo_ocr.render import page_count, render_page_to_png_bytes
    except ImportError:
        return {"status": "error", "error": "PyMuPDF (pymupdf) is not installed"}

    try:
        image_dir.mkdir(parents=True, exist_ok=True)
        total = page_count(pdf_path)
        pages_to_render = total if max_pages is None else min(total, max_pages)
        pages: list[dict[str, Any]] = []
        for idx in range(pages_to_render):
            png = render_page_to_png_bytes(pdf_path, page_index=idx, dpi=dpi)
            img_path = image_dir / f"page_{idx + 1:03d}.png"
            img_path.write_bytes(png)
            pages.append({"page_index": idx, "path": str(img_path)})
        return {"status": "ok", "page_count": total, "pages": pages}
    except Exception as exc:  # noqa: BLE001
        return {"status": "error", "error": str(exc)}


def _make_vlm_runner(base_url: str, model: str | None, api_key: str) -> Any:
    from gwanbo_ocr.runners.vllm import VllmChatRunner

    return VllmChatRunner(model=model or "default", base_url=base_url, api_key=api_key)


def _scope(max_pages: int | None) -> str:
    return "all_pages" if max_pages is None else f"first_{max_pages}_pages"


def _iso_now() -> str:
    return datetime.now().isoformat()
