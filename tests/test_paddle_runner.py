from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gwanbo_ocr.runners.paddle import PaddleOcrRunner, PaddleOcrVlRunner


class FakePaddleOcrNoCls:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def ocr(self, path: str, **kwargs: Any) -> dict[str, Any]:
        self.calls.append((path, dict(kwargs)))
        if "cls" in kwargs:
            raise TypeError("PaddleOCR.predict() got an unexpected keyword argument 'cls'")
        return {
            "rec_texts": ["정부공고", "본문"],
            "rec_scores": [0.99, 0.97],
            "rec_polys": [
                [[0, 0], [10, 0], [10, 10], [0, 10]],
                [[0, 20], [10, 20], [10, 30], [0, 30]],
            ],
        }


class FakePaddleOcrResult(dict[str, Any]):
    def json(self) -> dict[str, Any]:
        return {"rec_texts": []}


class FakePaddleOcrMappingResult:
    def ocr(self, path: str, **_kwargs: Any) -> list[FakePaddleOcrResult]:
        return [
            FakePaddleOcrResult(
                {
                    "rec_texts": ["매핑우선"],
                    "rec_scores": [0.98],
                    "rec_polys": [[[1, 2], [3, 2], [3, 4], [1, 4]]],
                }
            )
        ]


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


def test_paddle_ocr_runner_retries_without_cls_for_new_api(tmp_path: Path) -> None:
    image = tmp_path / "page.png"
    image.write_bytes(b"fake png")
    ocr = FakePaddleOcrNoCls()

    result = PaddleOcrRunner(ocr=ocr, lang="korean").transcribe(image, page_number=2)

    assert ocr.calls == [(str(image), {"cls": True}), (str(image), {})]
    assert result.backend == "paddleocr"
    assert result.page_number == 2
    assert result.text == "정부공고\n본문"
    assert result["blocks"][0]["bbox"] == [0.0, 0.0, 10.0, 10.0]


def test_paddle_ocr_runner_prefers_mapping_result_over_json_method(tmp_path: Path) -> None:
    image = tmp_path / "page.png"
    image.write_bytes(b"fake png")

    result = PaddleOcrRunner(ocr=FakePaddleOcrMappingResult()).transcribe(image)

    assert result.text == "매핑우선"
    assert result["blocks"][0]["bbox"] == [1.0, 2.0, 3.0, 4.0]


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
