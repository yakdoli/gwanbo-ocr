from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gwanbo_ocr.runners.paddle import PaddleOcrVlRunner


class FakePaddleVlResult:
    @property
    def markdown(self) -> dict[str, Any]:
        return {"markdown_texts": "# 제목\n\n본문"}

    @property
    def json(self) -> dict[str, Any]:
        return {
            "res": {
                "page_index": 7,
                "parsing_res_list": [
                    {
                        "block_label": "text",
                        "block_content": "본문",
                        "block_bbox": [1, 2, 30, 40],
                        "block_order": 1,
                    },
                    {
                        "block_label": "table",
                        "block_content": "<table>...</table>",
                        "block_bbox": [5, 6, 70, 80],
                        "block_order": 2,
                    },
                ],
            }
        }


class FakePaddleVlPipeline:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def predict(self, input: str, **kwargs: Any) -> list[FakePaddleVlResult]:
        self.calls.append((input, kwargs))
        return [FakePaddleVlResult()]


def test_paddle_ocr_vl_runner_normalizes_markdown_and_blocks(tmp_path: Path) -> None:
    image = tmp_path / "page.png"
    image.write_bytes(b"fake png")
    pipeline = FakePaddleVlPipeline()

    result = PaddleOcrVlRunner(pipeline=pipeline).transcribe(image, page_number=3, temperature=0)

    assert pipeline.calls == [(str(image), {"temperature": 0})]
    assert result.backend == "paddleocr_vl"
    assert result.page_number == 3
    assert result.text == "# 제목\n\n본문"
    assert result["pages"][0]["page_index"] == 7
    assert result["blocks"][0]["text"] == "본문"
    assert result["blocks"][1]["label"] == "table"
    assert result["tables"][0]["text"] == "<table>...</table>"
