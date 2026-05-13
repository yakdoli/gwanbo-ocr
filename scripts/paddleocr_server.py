#!/usr/bin/env python3
"""FastAPI service for containerized PaddleOCR and PaddleOCR-VL."""

from __future__ import annotations

import os
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from gwanbo_ocr.runners.paddle import PaddleOcrRunner, PaddleOcrVlRunner
from scripts.service_paths import resolve_allowed_input_path

app = FastAPI(
    title="Gwanbo PaddleOCR API",
    description="HTTP service wrapper around PaddleOCR and PaddleOCR-VL.",
    version="0.1.0",
)

_classic_runner: PaddleOcrRunner | None = None
_vl_runner: PaddleOcrVlRunner | None = None
_vl_runner_key: tuple[str | None, ...] | None = None


class ClassicOcrRequest(BaseModel):
    image_path: str
    lang: str = "korean"
    page_number: int | None = None


class VlOcrRequest(BaseModel):
    image_path: str
    page_number: int | None = None
    pipeline_version: str = "v1.5"
    vl_rec_backend: str | None = None
    vl_rec_server_url: str | None = None
    vl_rec_api_model_name: str | None = None
    vl_rec_api_key: str | None = None


@app.get("/health")
async def health_check() -> dict[str, Any]:
    return {
        "status": "ok",
        "service": "paddleocr-api",
        "version": "0.1.0",
    }


@app.post("/ocr/classic")
async def ocr_classic(request: ClassicOcrRequest) -> dict[str, Any]:
    try:
        image_path = resolve_allowed_input_path(request.image_path)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not image_path.exists():
        raise HTTPException(status_code=400, detail=f"image not found: {image_path}")
    try:
        result = _classic(request.lang).transcribe(image_path, page_number=request.page_number)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {"status": "ok", **result.to_dict()}


@app.post("/ocr/vl")
async def ocr_vl(request: VlOcrRequest) -> dict[str, Any]:
    try:
        image_path = resolve_allowed_input_path(request.image_path)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not image_path.exists():
        raise HTTPException(status_code=400, detail=f"image not found: {image_path}")
    try:
        result = _vl(request).transcribe(image_path, page_number=request.page_number)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {"status": "ok", **result.to_dict()}


def _classic(lang: str) -> PaddleOcrRunner:
    global _classic_runner
    if _classic_runner is None or _classic_runner.lang != lang:
        _classic_runner = PaddleOcrRunner(lang=lang)
    return _classic_runner


def _vl(request: VlOcrRequest) -> PaddleOcrVlRunner:
    global _vl_runner, _vl_runner_key
    key = _vl_key(request)
    if _vl_runner is None or _vl_runner_key != key:
        _vl_runner = PaddleOcrVlRunner(
            pipeline_version=request.pipeline_version,
            vl_rec_backend=request.vl_rec_backend or "vllm-server",
            vl_rec_server_url=request.vl_rec_server_url or os.getenv("PADDLEOCR_VL_REC_SERVER_URL"),
            vl_rec_api_model_name=request.vl_rec_api_model_name or os.getenv("PADDLEOCR_VL_MODEL"),
            vl_rec_api_key=request.vl_rec_api_key or os.getenv("PADDLEOCR_VL_API_KEY"),
        )
        _vl_runner_key = key
    return _vl_runner


def _vl_key(request: VlOcrRequest) -> tuple[str | None, ...]:
    return (
        request.pipeline_version,
        request.vl_rec_backend or "vllm-server",
        request.vl_rec_server_url or os.getenv("PADDLEOCR_VL_REC_SERVER_URL"),
        request.vl_rec_api_model_name or os.getenv("PADDLEOCR_VL_MODEL"),
        request.vl_rec_api_key or os.getenv("PADDLEOCR_VL_API_KEY"),
    )
