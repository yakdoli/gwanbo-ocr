"""Optional PaddleOCR runner adapter."""

from __future__ import annotations

import base64
import tempfile
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from gwanbo_ocr.runners.base import ImageInput, TranscriptionResult


class PaddleOcrRunner:
    """Wrap PaddleOCR behind the shared runner protocol.

    The PaddleOCR package is imported lazily so this adapter can live in the
    codebase without making PaddleOCR a required dependency.
    """

    def __init__(
        self,
        ocr: Any | None = None,
        *,
        lang: str = "ch",
        use_angle_cls: bool = True,
        **ocr_kwargs: Any,
    ) -> None:
        self._ocr = ocr
        self.lang = lang
        self.use_angle_cls = use_angle_cls
        self.ocr_kwargs = dict(ocr_kwargs)

    @property
    def ocr(self) -> Any:
        if self._ocr is None:
            try:
                from paddleocr import PaddleOCR  # type: ignore[import-not-found,import-untyped]
            except ImportError as exc:  # pragma: no cover - optional dependency
                raise RuntimeError(
                    "PaddleOCR is not installed. Install it to use PaddleOcrRunner."
                ) from exc
            self._ocr = PaddleOCR(
                use_angle_cls=self.use_angle_cls,
                lang=self.lang,
                **self.ocr_kwargs,
            )
        return self._ocr

    def transcribe(
        self,
        image: ImageInput,
        *,
        page_number: int | None = None,
        **kwargs: Any,
    ) -> TranscriptionResult:
        path, cleanup_path = _coerce_image_to_path(image)
        try:
            raw = self.ocr.ocr(path, cls=kwargs.pop("cls", self.use_angle_cls), **kwargs)
        finally:
            if cleanup_path is not None:
                cleanup_path.unlink(missing_ok=True)

        payload = _paddle_payload(raw)
        return TranscriptionResult.from_payload(
            payload,
            raw_text=payload.get("text", ""),
            page_number=page_number,
            raw_response=raw,
            backend="paddleocr",
        )

    def run(self, image: ImageInput, **kwargs: Any) -> TranscriptionResult:
        """Alias for ``transcribe`` used by simple pipeline code."""

        return self.transcribe(image, **kwargs)

    def transcribe_json(self, image: ImageInput, **kwargs: Any) -> dict[str, Any]:
        """Transcribe and return only the normalized JSON payload."""

        return dict(self.transcribe(image, **kwargs).data)


class PaddleOcrVlRunner:
    """Wrap PaddleOCR-VL document parsing behind the shared runner protocol."""

    def __init__(
        self,
        pipeline: Any | None = None,
        *,
        pipeline_version: str = "v1.5",
        vl_rec_backend: str | None = None,
        vl_rec_server_url: str | None = None,
        vl_rec_api_model_name: str | None = None,
        vl_rec_api_key: str | None = None,
        **pipeline_kwargs: Any,
    ) -> None:
        self._pipeline = pipeline
        self.pipeline_version = pipeline_version
        self.pipeline_kwargs = {
            **_drop_none(
                {
                    "vl_rec_backend": vl_rec_backend,
                    "vl_rec_server_url": vl_rec_server_url,
                    "vl_rec_api_model_name": vl_rec_api_model_name,
                    "vl_rec_api_key": vl_rec_api_key,
                }
            ),
            **pipeline_kwargs,
        }

    @property
    def pipeline(self) -> Any:
        if self._pipeline is None:
            try:
                from paddleocr import PaddleOCRVL  # type: ignore[import-not-found,import-untyped]
            except ImportError as exc:  # pragma: no cover - optional dependency
                raise RuntimeError(
                    "PaddleOCR-VL is not installed. Install paddleocr[doc-parser] "
                    "or this project's paddleocr extra to use PaddleOcrVlRunner."
                ) from exc
            self._pipeline = PaddleOCRVL(
                pipeline_version=self.pipeline_version,
                **self.pipeline_kwargs,
            )
        return self._pipeline

    def transcribe(
        self,
        image: ImageInput,
        *,
        page_number: int | None = None,
        **kwargs: Any,
    ) -> TranscriptionResult:
        path, cleanup_path = _coerce_image_to_path(image)
        try:
            raw = self.pipeline.predict(path, **kwargs)
        finally:
            if cleanup_path is not None:
                cleanup_path.unlink(missing_ok=True)

        payload = _paddle_vl_payload(raw)
        return TranscriptionResult.from_payload(
            payload,
            raw_text=payload.get("text", ""),
            page_number=page_number,
            raw_response=raw,
            backend="paddleocr_vl",
        )

    def run(self, image: ImageInput, **kwargs: Any) -> TranscriptionResult:
        """Alias for ``transcribe`` used by simple pipeline code."""

        return self.transcribe(image, **kwargs)

    def transcribe_json(self, image: ImageInput, **kwargs: Any) -> dict[str, Any]:
        """Transcribe and return only the normalized JSON payload."""

        return dict(self.transcribe(image, **kwargs).data)


def _drop_none(values: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in values.items() if value is not None}


def _coerce_image_to_path(image: ImageInput) -> tuple[str, Path | None]:
    if isinstance(image, Path):
        return str(image.expanduser()), None

    if isinstance(image, str):
        if image.startswith("data:"):
            return _write_temp_image(_decode_data_url(image))
        return str(Path(image).expanduser()), None

    if isinstance(image, (bytes, bytearray, memoryview)):
        return _write_temp_image(bytes(image))

    if hasattr(image, "save"):
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as handle:
            temp_path = Path(handle.name)
            image.save(handle, format="PNG")
        return str(temp_path), temp_path

    raise TypeError("image must be a data URL, bytes, pathlib.Path, path string, or Pillow image")


def _decode_data_url(data_url: str) -> bytes:
    try:
        _, encoded = data_url.split(",", 1)
    except ValueError as exc:
        raise ValueError("Invalid data URL") from exc
    return base64.b64decode(encoded)


def _write_temp_image(data: bytes) -> tuple[str, Path]:
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as handle:
        handle.write(data)
        temp_path = Path(handle.name)
    return str(temp_path), temp_path


def _paddle_payload(raw: Any) -> dict[str, Any]:
    blocks = list(_iter_paddle_blocks(raw))
    text = "\n".join(block["text"] for block in blocks if block.get("text"))
    return {
        "text": text,
        "blocks": blocks,
        "tables": [],
        "warnings": [],
    }


def _paddle_vl_payload(raw: Any) -> dict[str, Any]:
    pages: list[dict[str, Any]] = []
    all_blocks: list[dict[str, Any]] = []
    markdown_pages: list[str] = []

    for index, item in enumerate(_as_items(raw)):
        page_json = _extract_result_json(item)
        markdown_text = _extract_markdown_text(item)
        source = page_json if page_json is not None else item
        blocks = list(_iter_paddle_vl_blocks(source))
        if not blocks:
            blocks = list(_iter_paddle_blocks(source))

        page_text = markdown_text or "\n".join(
            block["text"] for block in blocks if block.get("text")
        )
        if markdown_text:
            markdown_pages.append(markdown_text)

        page_payload: dict[str, Any] = {
            "page_index": _page_index(page_json, fallback=index),
            "text": page_text,
            "text_chars": len(page_text),
        }
        if markdown_text:
            page_payload["markdown"] = markdown_text
        if blocks:
            page_payload["blocks"] = blocks
        pages.append(page_payload)
        all_blocks.extend(blocks)

    text = "\n\n".join(page["text"] for page in pages if page.get("text"))
    payload: dict[str, Any] = {
        "text": text,
        "blocks": all_blocks,
        "tables": [block for block in all_blocks if block.get("label") == "table"],
        "pages": pages,
        "warnings": [],
    }
    if markdown_pages:
        payload["markdown"] = "\n\n".join(markdown_pages)
    return payload


def _iter_paddle_blocks(raw: Any) -> Iterable[dict[str, Any]]:
    if raw is None:
        return

    if hasattr(raw, "json"):
        json_value = raw.json
        if callable(json_value):
            json_value = json_value()
        yield from _iter_paddle_blocks(json_value)
        return

    if isinstance(raw, Mapping):
        yield from _iter_mapping_blocks(raw)
        return

    if isinstance(raw, (list, tuple)):
        if _looks_like_classic_line(raw):
            block = _classic_line_to_block(raw)
            if block is not None:
                yield block
            return
        for item in raw:
            yield from _iter_paddle_blocks(item)


def _iter_mapping_blocks(raw: Mapping[str, Any]) -> Iterable[dict[str, Any]]:
    texts = raw.get("rec_texts") or raw.get("texts") or raw.get("text")
    scores = raw.get("rec_scores") or raw.get("scores") or raw.get("confidence")
    boxes = raw.get("rec_polys") or raw.get("dt_polys") or raw.get("boxes") or raw.get("bbox")

    if isinstance(texts, str):
        yield {
            "text": texts,
            "bbox": _bbox_from_points(boxes),
            "confidence": _score_at(scores, 0),
        }
        return

    if isinstance(texts, list):
        for index, text in enumerate(texts):
            if not isinstance(text, str):
                continue
            yield {
                "text": text,
                "bbox": _bbox_from_points(_item_at(boxes, index)),
                "confidence": _score_at(scores, index),
            }


def _as_items(raw: Any) -> Iterable[Any]:
    if raw is None:
        return
    if isinstance(raw, (str, bytes, bytearray, memoryview, Mapping)):
        yield raw
        return
    if isinstance(raw, Iterable):
        yield from raw
        return
    yield raw


def _extract_result_json(value: Any) -> Any:
    json_value = getattr(value, "json", None)
    if json_value is not None:
        if callable(json_value):
            return json_value()
        return json_value
    if isinstance(value, Mapping):
        return value
    return None


def _extract_markdown_text(value: Any) -> str:
    markdown_value = None
    if isinstance(value, Mapping):
        markdown_value = value.get("markdown")
    if markdown_value is None:
        markdown_value = getattr(value, "markdown", None)
        if callable(markdown_value):
            markdown_value = markdown_value()
    return _text_from_markdown_value(markdown_value)


def _text_from_markdown_value(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        for key in ("markdown_texts", "markdown", "text", "content", "res"):
            text = _text_from_markdown_value(value.get(key))
            if text:
                return text
    if isinstance(value, (list, tuple)):
        parts = [_text_from_markdown_value(item) for item in value]
        return "\n\n".join(part for part in parts if part)
    return ""


def _iter_paddle_vl_blocks(raw: Any) -> Iterable[dict[str, Any]]:
    if raw is None:
        return

    if hasattr(raw, "json"):
        yield from _iter_paddle_vl_blocks(_extract_result_json(raw))
        return

    if isinstance(raw, Mapping):
        value = raw.get("res") if isinstance(raw.get("res"), Mapping) else raw

        parsing_blocks = value.get("parsing_res_list") if isinstance(value, Mapping) else None
        if isinstance(parsing_blocks, list):
            for block in parsing_blocks:
                normalized = _paddle_vl_block(block)
                if normalized is not None:
                    yield normalized

        if isinstance(value, Mapping):
            for key in ("overall_ocr_res", "spotting_res"):
                yield from _iter_paddle_blocks(value.get(key))
        return

    if isinstance(raw, (list, tuple)):
        for item in raw:
            yield from _iter_paddle_vl_blocks(item)


def _paddle_vl_block(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, Mapping):
        return None
    text = _first_string(value, ("block_content", "content", "text", "rec_text"))
    if not text:
        return None

    block: dict[str, Any] = {
        "text": text,
        "bbox": _bbox_from_points(
            value.get("block_bbox")
            or value.get("bbox")
            or value.get("block_polygon_points")
            or value.get("polygon_points")
        ),
    }
    label = _first_string(value, ("block_label", "label"))
    if label:
        block["label"] = label
    for source_key, target_key in (
        ("block_order", "order"),
        ("block_id", "block_id"),
        ("group_id", "group_id"),
    ):
        if source_key in value:
            block[target_key] = value[source_key]
    return block


def _first_string(value: Mapping[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        candidate = value.get(key)
        if isinstance(candidate, str):
            return candidate
    return ""


def _page_index(value: Any, *, fallback: int) -> int:
    if isinstance(value, Mapping):
        inner = value.get("res") if isinstance(value.get("res"), Mapping) else value
        page_index = inner.get("page_index") if isinstance(inner, Mapping) else None
        if isinstance(page_index, int):
            return page_index
    return fallback


def _looks_like_classic_line(value: Any) -> bool:
    return (
        isinstance(value, (list, tuple))
        and len(value) >= 2
        and isinstance(value[1], (list, tuple))
        and len(value[1]) >= 1
        and isinstance(value[1][0], str)
    )


def _classic_line_to_block(line: Any) -> dict[str, Any] | None:
    if not _looks_like_classic_line(line):
        return None
    text = line[1][0]
    confidence = line[1][1] if len(line[1]) > 1 else None
    return {
        "text": text,
        "bbox": _bbox_from_points(line[0]),
        "confidence": confidence,
    }


def _bbox_from_points(points: Any) -> list[float] | None:
    if points is None:
        return None

    if (
        isinstance(points, (list, tuple))
        and len(points) == 4
        and all(isinstance(value, (int, float)) for value in points)
    ):
        return [float(value) for value in points]

    if not isinstance(points, (list, tuple)):
        return None

    xs: list[float] = []
    ys: list[float] = []
    for point in points:
        if (
            isinstance(point, (list, tuple))
            and len(point) >= 2
            and isinstance(point[0], (int, float))
            and isinstance(point[1], (int, float))
        ):
            xs.append(float(point[0]))
            ys.append(float(point[1]))
    if not xs or not ys:
        return None
    return [min(xs), min(ys), max(xs), max(ys)]


def _item_at(values: Any, index: int) -> Any:
    if isinstance(values, (list, tuple)) and index < len(values):
        return values[index]
    return values


def _score_at(values: Any, index: int) -> float | None:
    value = _item_at(values, index)
    if isinstance(value, (int, float)):
        return float(value)
    return None


PaddleOCRRunner = PaddleOcrRunner
PaddleRunner = PaddleOcrRunner
PaddleOcrVLRunner = PaddleOcrVlRunner
PaddleOCRVLRunner = PaddleOcrVlRunner
