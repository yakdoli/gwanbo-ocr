"""PDF page rendering helpers backed by PyMuPDF.

The functions in this module keep PyMuPDF as an optional runtime dependency:
importing the module succeeds without it, while rendering raises a clear error.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import os
from collections.abc import Iterable, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from gwanbo_ocr.pdf.io import pdf_key_from_row, resolve_pdf_path

ColorSpace = Literal["rgb", "gray", "cmyk"]
PdfInput = str | Path | bytes | bytearray | memoryview


@dataclass(frozen=True)
class RenderedPage:
    """A rendered PDF page plus lightweight metadata."""

    image: Any
    page_index: int
    width: int
    height: int
    dpi: int

    @property
    def page_number(self) -> int:
        """One-based page number for display and prompts."""

        return self.page_index + 1

    def to_png_bytes(self) -> bytes:
        """Return the page image encoded as PNG bytes."""

        buffer = io.BytesIO()
        self.image.save(buffer, format="PNG")
        return buffer.getvalue()

    def to_base64(self) -> str:
        """Return the page image as base64-encoded PNG text."""

        return base64.b64encode(self.to_png_bytes()).decode("ascii")

    def to_data_url(self) -> str:
        """Return the page image as an OpenAI-compatible PNG data URL."""

        return f"data:image/png;base64,{self.to_base64()}"


def _import_fitz() -> Any:
    try:
        import fitz  # type: ignore[import-not-found,import-untyped]
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError(
            "PyMuPDF is required for PDF rendering. Install it with `pip install pymupdf`."
        ) from exc
    return fitz


def _import_image() -> Any:
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError(
            "Pillow is required to materialize rendered pages as images. "
            "Install it with `pip install pillow`."
        ) from exc
    return Image


def _open_document(pdf: PdfInput) -> Any:
    fitz = _import_fitz()
    if isinstance(pdf, (bytes, bytearray, memoryview)):
        return fitz.open(stream=bytes(pdf), filetype="pdf")
    return fitz.open(str(Path(pdf).expanduser()))


def _fitz_colorspace(fitz: Any, colorspace: ColorSpace) -> Any:
    if colorspace == "rgb":
        return fitz.csRGB
    if colorspace == "gray":
        return fitz.csGRAY
    if colorspace == "cmyk":
        return fitz.csCMYK
    raise ValueError(f"Unsupported colorspace: {colorspace!r}")


def _resolve_page_index(
    page_index: int | None,
    page_number: int | None,
) -> int:
    if page_number is not None:
        if page_number < 1:
            raise ValueError("page_number is one-based and must be >= 1")
        return page_number - 1
    if page_index is None:
        return 0
    if page_index < 0:
        raise ValueError("page_index is zero-based and must be >= 0")
    return page_index


def _rect_from_clip(fitz: Any, clip: Sequence[float] | None) -> Any | None:
    if clip is None:
        return None
    if len(clip) != 4:
        raise ValueError("clip must contain four coordinates: x0, y0, x1, y1")
    return fitz.Rect(*clip)


def _pixmap_to_image(pixmap: Any) -> Any:
    Image = _import_image()
    image = Image.open(io.BytesIO(pixmap.tobytes("png")))
    image.load()
    return image


def render_pdf_page(
    pdf: PdfInput,
    page_index: int | None = 0,
    *,
    page_number: int | None = None,
    dpi: int = 200,
    colorspace: ColorSpace = "rgb",
    alpha: bool = False,
    clip: Sequence[float] | None = None,
    annotations: bool = True,
) -> Any:
    """Render one PDF page to a Pillow image.

    Args:
        pdf: A PDF path or PDF bytes.
        page_index: Zero-based page index. Ignored when ``page_number`` is set.
        page_number: One-based page number for call sites that prefer display
            numbering.
        dpi: Render resolution. PyMuPDF uses 72 points per inch internally.
        colorspace: Output color space requested from PyMuPDF.
        alpha: Whether to keep an alpha channel.
        clip: Optional page-space rectangle ``(x0, y0, x1, y1)``.
        annotations: Whether annotation appearances are included.
    """

    rendered = render_pdf_page_result(
        pdf,
        page_index,
        page_number=page_number,
        dpi=dpi,
        colorspace=colorspace,
        alpha=alpha,
        clip=clip,
        annotations=annotations,
    )
    return rendered.image


def render_pdf_page_result(
    pdf: PdfInput,
    page_index: int | None = 0,
    *,
    page_number: int | None = None,
    dpi: int = 200,
    colorspace: ColorSpace = "rgb",
    alpha: bool = False,
    clip: Sequence[float] | None = None,
    annotations: bool = True,
) -> RenderedPage:
    """Render one PDF page and return image metadata with it."""

    if dpi <= 0:
        raise ValueError("dpi must be positive")

    fitz = _import_fitz()
    resolved_index = _resolve_page_index(page_index, page_number)
    document = _open_document(pdf)
    try:
        if resolved_index >= len(document):
            raise IndexError(
                f"page_index {resolved_index} is outside PDF with {len(document)} pages"
            )
        page = document.load_page(resolved_index)
        scale = dpi / 72.0
        matrix = fitz.Matrix(scale, scale)
        pixmap = page.get_pixmap(
            matrix=matrix,
            colorspace=_fitz_colorspace(fitz, colorspace),
            alpha=alpha,
            clip=_rect_from_clip(fitz, clip),
            annots=annotations,
        )
        image = _pixmap_to_image(pixmap)
        return RenderedPage(
            image=image,
            page_index=resolved_index,
            width=image.width,
            height=image.height,
            dpi=dpi,
        )
    finally:
        document.close()


def render_pdf_pages(
    pdf: PdfInput,
    page_indices: Iterable[int] | None = None,
    *,
    dpi: int = 200,
    colorspace: ColorSpace = "rgb",
    alpha: bool = False,
    annotations: bool = True,
) -> list[RenderedPage]:
    """Render multiple PDF pages to ``RenderedPage`` objects."""

    if dpi <= 0:
        raise ValueError("dpi must be positive")

    fitz = _import_fitz()
    document = _open_document(pdf)
    try:
        indices = list(range(len(document))) if page_indices is None else list(page_indices)
        results: list[RenderedPage] = []
        scale = dpi / 72.0
        matrix = fitz.Matrix(scale, scale)
        fitz_colorspace = _fitz_colorspace(fitz, colorspace)
        for index in indices:
            if index < 0:
                raise ValueError("page indices must be zero-based and >= 0")
            if index >= len(document):
                raise IndexError(f"page_index {index} is outside PDF with {len(document)} pages")
            pixmap = document.load_page(index).get_pixmap(
                matrix=matrix,
                colorspace=fitz_colorspace,
                alpha=alpha,
                annots=annotations,
            )
            image = _pixmap_to_image(pixmap)
            results.append(
                RenderedPage(
                    image=image,
                    page_index=index,
                    width=image.width,
                    height=image.height,
                    dpi=dpi,
                )
            )
        return results
    finally:
        document.close()


def render_page_to_png_bytes(
    pdf: PdfInput,
    page_index: int | None = 0,
    *,
    page_number: int | None = None,
    dpi: int = 200,
) -> bytes:
    """Render one PDF page and return PNG bytes."""

    return render_pdf_page_result(
        pdf,
        page_index,
        page_number=page_number,
        dpi=dpi,
    ).to_png_bytes()


def render_page_to_data_url(
    pdf: PdfInput,
    page_index: int | None = 0,
    *,
    page_number: int | None = None,
    dpi: int = 200,
) -> str:
    """Render one PDF page and return a PNG data URL."""

    return render_pdf_page_result(
        pdf,
        page_index,
        page_number=page_number,
        dpi=dpi,
    ).to_data_url()


def page_count(pdf: PdfInput) -> int:
    """Return the number of pages in a PDF."""

    document = _open_document(pdf)
    try:
        return len(document)
    finally:
        document.close()


# Friendly aliases for likely call sites.
render_page = render_pdf_page
render_pdf_page_image = render_pdf_page
render_page_result = render_pdf_page_result
render_pages = render_pdf_pages
render_page_png = render_page_to_png_bytes
render_to_data_url = render_page_to_data_url
count_pages = page_count


def render_manifest(
    *,
    manifest_path: str | Path,
    output_dir: str | Path,
    dpi: int = 200,
    max_long_edge: int = 2400,
    max_pages: int | None = 1,
    workers: int = 4,
    limit: int | None = None,
) -> dict[str, Any]:
    """Render PDF pages referenced by a JSONL manifest into PNG files."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    input_rows = [
        (index, row)
        for index, row in enumerate(_read_jsonl(manifest_path), start=1)
        if isinstance(row, dict) and (limit is None or index <= limit)
    ]
    rows: list[dict[str, Any]] = []
    counts = {"total_rows": 0, "rendered_pages": 0, "errors": 0}

    tasks = [
        {
            "index": index,
            "row": row,
            "output_dir": str(output),
            "dpi": dpi,
            "max_long_edge": max_long_edge,
            "max_pages": max_pages,
        }
        for index, row in input_rows
    ]
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            results = list(executor.map(_render_manifest_row, tasks))
    else:
        results = [_render_manifest_row(task) for task in tasks]

    for result in results:
        counts["total_rows"] += 1
        counts["rendered_pages"] += int(result.get("rendered_pages") or 0)
        counts["errors"] += int(result.get("errors") or 0)
        rows.extend(result.get("rows") or [])

    manifest_out = output / "manifest.jsonl"
    _write_jsonl_atomic(manifest_out, rows)
    summary = {
        "status": "ok",
        "input": str(manifest_path),
        "output_dir": str(output),
        "manifest": str(manifest_out),
        "workers": max(1, workers),
        "counts": counts,
    }
    _write_json_atomic(output / "summary.json", summary)
    return summary


def _render_manifest_row(task: dict[str, Any]) -> dict[str, Any]:
    row = task["row"]
    index = int(task["index"])
    output = Path(str(task["output_dir"]))
    dpi = int(task["dpi"])
    max_long_edge = int(task["max_long_edge"])
    max_pages = task.get("max_pages")
    pdf_path = resolve_pdf_path(row, peti_root="/root/peti")
    pdf_key = pdf_key_from_row(row, fallback=pdf_path.stem or f"row-{index}")
    rows: list[dict[str, Any]] = []
    rendered_pages = 0
    errors = 0

    try:
        page_numbers = _selected_page_numbers(row, max_pages)
    except Exception as exc:  # noqa: BLE001
        return {
            "rendered_pages": 0,
            "errors": 1,
            "rows": [
                {
                    "pdf_key": pdf_key,
                    "id": row.get("id"),
                    "pdf_path": str(pdf_path),
                    "page_number": None,
                    "image_path": "",
                    "dpi": dpi,
                    "max_long_edge": max_long_edge,
                    "status": "error",
                    "error": f"invalid_page_selection:{exc}",
                }
            ],
        }

    for page_number in page_numbers:
        image_path = output / "pages" / pdf_key / f"page_{page_number:04d}.png"
        record = {
            "pdf_key": pdf_key,
            "id": row.get("id"),
            "pdf_path": str(pdf_path),
            "page_number": page_number,
            "image_path": str(image_path),
            "dpi": dpi,
            "max_long_edge": max_long_edge,
        }
        try:
            rendered = render_pdf_page_result(pdf_path, page_number=page_number, dpi=dpi)
            image = _resize_to_max_edge(rendered.image, max_long_edge)
            image_path.parent.mkdir(parents=True, exist_ok=True)
            image.save(image_path, format="PNG")
            payload = image_path.read_bytes()
            record.update(
                {
                    "status": "ok",
                    "width": image.width,
                    "height": image.height,
                    "image_sha256": hashlib.sha256(payload).hexdigest(),
                    "size_bytes": len(payload),
                }
            )
            rendered_pages += 1
        except Exception as exc:  # noqa: BLE001
            record.update({"status": "error", "error": str(exc)})
            errors += 1
        rows.append(record)
    return {"rendered_pages": rendered_pages, "errors": errors, "rows": rows}


def _selected_page_numbers(row: dict[str, Any], max_pages: int | None) -> list[int]:
    selected = row.get("selected_pages")
    if isinstance(selected, list) and selected:
        return [int(value) for value in selected if int(value) >= 1]
    if max_pages is None:
        pages = int(row.get("pages") or 1)
        return list(range(1, max(pages, 1) + 1))
    return list(range(1, max(max_pages, 1) + 1))


def _resize_to_max_edge(image: Any, max_long_edge: int) -> Any:
    if max_long_edge <= 0:
        return image
    longest = max(image.width, image.height)
    if longest <= max_long_edge:
        return image
    scale = max_long_edge / longest
    size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
    return image.resize(size)


def render_manifest_adaptive(
    *,
    manifest_path: str | Path,
    output_dir: str | Path,
    resolution_plan: dict[str, Any] | None = None,
    default_dpi: int = 200,
    default_max_long_edge: int = 1540,
    max_pages: int | None = 1,
    workers: int = 4,
    limit: int | None = None,
) -> dict[str, Any]:
    """Render PDF pages with per-page resolution settings from a plain dict plan."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    input_rows = [
        (index, row)
        for index, row in enumerate(_read_jsonl(manifest_path), start=1)
        if isinstance(row, dict) and (limit is None or index <= limit)
    ]
    rows: list[dict[str, Any]] = []
    counts = {"total_rows": 0, "rendered_pages": 0, "errors": 0}

    tasks = []
    for index, row in input_rows:
        try:
            page_numbers = _selected_page_numbers(row, max_pages)
            resolutions = {
                page_number: _resolve_page_resolution(
                    page_number - 1,
                    resolution_plan,
                    default_dpi,
                    default_max_long_edge,
                )
                for page_number in page_numbers
            }
        except Exception:
            resolutions = {}
        tasks.append(
            {
                "index": index,
                "row": row,
                "output_dir": str(output),
                "default_dpi": default_dpi,
                "default_max_long_edge": default_max_long_edge,
                "max_pages": max_pages,
                "resolutions": resolutions,
            }
        )
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            results = list(executor.map(_render_manifest_row_adaptive, tasks))
    else:
        results = [_render_manifest_row_adaptive(task) for task in tasks]

    for result in results:
        counts["total_rows"] += 1
        counts["rendered_pages"] += int(result.get("rendered_pages") or 0)
        counts["errors"] += int(result.get("errors") or 0)
        rows.extend(result.get("rows") or [])

    manifest_out = output / "manifest.jsonl"
    _write_jsonl_atomic(manifest_out, rows)
    summary = {
        "status": "ok",
        "input": str(manifest_path),
        "output_dir": str(output),
        "manifest": str(manifest_out),
        "workers": max(1, workers),
        "default_dpi": default_dpi,
        "default_max_long_edge": default_max_long_edge,
        "counts": counts,
    }
    _write_json_atomic(output / "summary.json", summary)
    return summary


def _resolve_page_resolution(
    page_index: int,
    resolution_plan: dict[str, Any] | None,
    default_dpi: int,
    default_max_long_edge: int,
) -> tuple[int, int]:
    """Return ``(dpi, max_long_edge)`` for a zero-based page index."""
    if not resolution_plan:
        return int(default_dpi), int(default_max_long_edge)

    for page in resolution_plan.get("pages") or []:
        if not isinstance(page, dict):
            continue
        if int(page.get("page_index", -1)) == page_index:
            return (
                int(page.get("dpi", default_dpi)),
                int(page.get("max_long_edge", default_max_long_edge)),
            )

    default_tier = resolution_plan.get("default_tier")
    if isinstance(default_tier, dict):
        return (
            int(default_tier.get("dpi", default_dpi)),
            int(default_tier.get("max_long_edge", default_max_long_edge)),
        )
    return int(default_dpi), int(default_max_long_edge)


def _render_manifest_row_adaptive(task: dict[str, Any]) -> dict[str, Any]:
    row = task["row"]
    index = int(task["index"])
    output = Path(str(task["output_dir"]))
    default_dpi = int(task["default_dpi"])
    default_max_long_edge = int(task["default_max_long_edge"])
    max_pages = task.get("max_pages")
    resolutions = task.get("resolutions") or {}
    pdf_path = resolve_pdf_path(row, peti_root="/root/peti")
    pdf_key = pdf_key_from_row(row, fallback=pdf_path.stem or f"row-{index}")
    rows: list[dict[str, Any]] = []
    rendered_pages = 0
    errors = 0

    try:
        page_numbers = _selected_page_numbers(row, max_pages)
    except Exception as exc:  # noqa: BLE001
        return {
            "rendered_pages": 0,
            "errors": 1,
            "rows": [
                {
                    "pdf_key": pdf_key,
                    "id": row.get("id"),
                    "pdf_path": str(pdf_path),
                    "page_number": None,
                    "image_path": "",
                    "dpi": default_dpi,
                    "max_long_edge": default_max_long_edge,
                    "status": "error",
                    "error": f"invalid_page_selection:{exc}",
                }
            ],
        }

    for page_number in page_numbers:
        dpi, max_long_edge = resolutions.get(
            page_number,
            (default_dpi, default_max_long_edge),
        )
        dpi = int(dpi)
        max_long_edge = int(max_long_edge)
        image_path = output / "pages" / pdf_key / f"page_{page_number:04d}.png"
        record = {
            "pdf_key": pdf_key,
            "id": row.get("id"),
            "pdf_path": str(pdf_path),
            "page_number": page_number,
            "image_path": str(image_path),
            "dpi": dpi,
            "max_long_edge": max_long_edge,
        }
        try:
            rendered = render_pdf_page_result(pdf_path, page_number=page_number, dpi=dpi)
            image = _resize_to_max_edge(rendered.image, max_long_edge)
            image_path.parent.mkdir(parents=True, exist_ok=True)
            image.save(image_path, format="PNG")
            payload = image_path.read_bytes()
            record.update(
                {
                    "status": "ok",
                    "width": image.width,
                    "height": image.height,
                    "image_sha256": hashlib.sha256(payload).hexdigest(),
                    "size_bytes": len(payload),
                }
            )
            rendered_pages += 1
        except Exception as exc:  # noqa: BLE001
            record.update({"status": "error", "error": str(exc)})
            errors += 1
        rows.append(record)
    return {"rendered_pages": rendered_pages, "errors": errors, "rows": rows}


def _read_jsonl(path: str | Path) -> Iterable[Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def _write_jsonl_atomic(path: Path, rows: Iterable[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    with temp_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, default=str))
            handle.write("\n")
    temp_path.replace(path)


def _write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temp_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    temp_path.replace(path)
