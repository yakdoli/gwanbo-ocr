#!/usr/bin/env python3
"""FastAPI service for MarkItDown plain and OCR+LLM conversion."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from gwanbo_ocr.conversion import ConversionError, convert_document
from scripts.service_paths import resolve_allowed_path

app = FastAPI(
    title="Gwanbo MarkItDown OCR API",
    description="Document-to-Markdown conversion with optional markitdown-ocr and LLM support.",
    version="0.2.0",
)


class ConversionRequest(BaseModel):
    file_path: str | None = None
    output_path: str | None = None
    mode: str = "plain"
    llm_base_url: str | None = None
    llm_model: str | None = None
    llm_api_key: str | None = None
    llm_prompt: str | None = None


class BatchConversionRequest(BaseModel):
    file_paths: list[str]
    output_dir: str = "/workspace/runs/markitdown"
    mode: str = "plain"
    llm_base_url: str | None = None
    llm_model: str | None = None
    llm_api_key: str | None = None
    llm_prompt: str | None = None


@app.get("/health")
async def health_check() -> dict[str, Any]:
    return {
        "status": "ok",
        "service": "markitdown-ocr-api",
        "version": "0.2.0",
    }


@app.post("/convert/path")
async def convert_path(request: ConversionRequest) -> dict[str, Any]:
    if not request.file_path:
        raise HTTPException(status_code=400, detail="file_path is required")
    try:
        input_path = resolve_allowed_path(request.file_path)
        output_path = resolve_allowed_path(request.output_path) if request.output_path else None
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return _convert_path(
        input_path,
        output=output_path or _default_output_path(),
        mode=request.mode,
        llm_base_url=request.llm_base_url,
        llm_model=request.llm_model,
        llm_api_key=request.llm_api_key,
        llm_prompt=request.llm_prompt,
    )


@app.post("/convert/file")
async def convert_file(
    file: UploadFile = File(...),
    mode: str = "plain",
    llm_base_url: str | None = None,
    llm_model: str | None = None,
    llm_api_key: str | None = None,
    llm_prompt: str | None = None,
) -> dict[str, Any]:
    suffix = Path(file.filename or "upload.bin").suffix or ".bin"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as handle:
        temp_path = Path(handle.name)
        handle.write(await file.read())
    try:
        return _convert_path(
            temp_path,
            output=_default_output_path(),
            mode=mode,
            llm_base_url=llm_base_url,
            llm_model=llm_model,
            llm_api_key=llm_api_key,
            llm_prompt=llm_prompt,
        )
    finally:
        temp_path.unlink(missing_ok=True)


@app.post("/convert/batch")
async def convert_batch(request: BatchConversionRequest) -> dict[str, Any]:
    output_dir = resolve_allowed_path(request.output_dir)
    output_names = _unique_output_names([Path(path) for path in request.file_paths])
    results: dict[str, Any] = {}
    for index, file_path in enumerate(request.file_paths):
        try:
            results[file_path] = _convert_path(
                resolve_allowed_path(file_path),
                output=output_dir / output_names[index],
                mode=request.mode,
                llm_base_url=request.llm_base_url,
                llm_model=request.llm_model,
                llm_api_key=request.llm_api_key,
                llm_prompt=request.llm_prompt,
            )
        except HTTPException as exc:
            results[file_path] = {"status": "error", "error": exc.detail}
    return {
        "status": "ok",
        "total": len(request.file_paths),
        "results": results,
    }


def _convert_path(
    input_path: Path,
    *,
    output: Path,
    mode: str,
    llm_base_url: str | None,
    llm_model: str | None,
    llm_api_key: str | None,
    llm_prompt: str | None,
) -> dict[str, Any]:
    input_path = resolve_allowed_path(input_path)
    output = resolve_allowed_path(output)
    try:
        summary = convert_document(
            input_path=input_path,
            output=output,
            mode=mode,
            llm_base_url=llm_base_url or os.getenv("MARKITDOWN_LLM_BASE_URL"),
            llm_model=llm_model or os.getenv("MARKITDOWN_LLM_MODEL"),
            llm_api_key=llm_api_key or os.getenv("OPENAI_API_KEY", "dummy"),
            llm_prompt=llm_prompt,
        )
    except ConversionError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    markdown_content = Path(summary["output_path"]).read_text(encoding="utf-8")
    return {
        "status": "ok",
        "markdown_content": markdown_content,
        "file_path": summary["output_path"],
        "metadata": summary,
    }


def _default_output_path() -> Path:
    return resolve_allowed_path(os.getenv("MARKITDOWN_OUTPUT_DIR", "/tmp/gwanbo-ocr-markitdown"))


def _unique_output_names(paths: list[Path]) -> list[str]:
    counts: dict[str, int] = {}
    names: list[str] = []
    for path in paths:
        base = path.stem or "document"
        count = counts.get(base, 0) + 1
        counts[base] = count
        names.append(f"{base}.md" if count == 1 else f"{base}_{count}.md")
    return names
