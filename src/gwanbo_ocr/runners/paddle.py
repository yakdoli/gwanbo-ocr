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
                from paddleocr import PaddleOCR
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


def _iter_paddle_blocks(raw: Any) -> Iterable[dict[str, Any]]:
    if raw is None:
        return

    if isinstance(raw, Mapping):
        yield from _iter_mapping_blocks(raw)
        return

    if hasattr(raw, "json"):
        json_value = raw.json
        if callable(json_value):
            json_value = json_value()
        yield from _iter_paddle_blocks(json_value)
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
