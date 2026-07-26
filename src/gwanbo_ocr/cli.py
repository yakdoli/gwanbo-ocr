"""Command-line entry point for the focused PDF workflow."""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Annotated

import typer

app = typer.Typer(no_args_is_help=True)
pdf_app = typer.Typer(no_args_is_help=True)
app.add_typer(pdf_app, name="pdf")


@pdf_app.command("classify")
def classify_pdf_command(
    input_path: Annotated[Path, typer.Option("--input", exists=True, dir_okay=False)],
    output_dir: Annotated[Path, typer.Option("--output", file_okay=False)],
    max_pages: Annotated[int, typer.Option("--max-pages", min=1)] = 3,
    peti_root: Annotated[Path | None, typer.Option("--peti-root", file_okay=False)] = None,
    timeout_seconds: Annotated[float, typer.Option("--timeout-seconds", min=0.1)] = 30,
    max_file_bytes: Annotated[int, typer.Option("--max-file-bytes", min=1)] = 256 * 1024 * 1024,
) -> None:
    """Classify manifest PDFs by observable page content."""
    from .pdf.classification import classify_manifest

    summary = classify_manifest(
        manifest_path=input_path,
        output_dir=output_dir,
        max_pages=max_pages,
        peti_root=peti_root,
        timeout_seconds=timeout_seconds,
        max_file_bytes=max_file_bytes,
    )
    typer.echo(json.dumps(summary, ensure_ascii=False, default=str))


@pdf_app.command("correct")
def correct_pdf_command(
    input_path: Annotated[Path, typer.Option("--input", exists=True, dir_okay=False)],
    output_path: Annotated[Path, typer.Option("--output", dir_okay=False)],
    vlm_base_url: Annotated[str | None, typer.Option("--vlm-base-url")] = None,
    vlm_model: Annotated[str | None, typer.Option("--vlm-model")] = None,
    vlm_crop_root: Annotated[
        Path | None, typer.Option("--vlm-crop-root", exists=True, file_okay=False)
    ] = None,
    vlm_timeout: Annotated[float, typer.Option("--vlm-timeout", min=0.1)] = 30,
) -> None:
    """Apply safe normalization and optional gated VLM span review."""
    from .pdf.correct import correct_manifest
    from .pdf.openai_review import OpenAICompatibleReviewBackend

    if (vlm_base_url is None) != (vlm_model is None):
        raise typer.BadParameter("--vlm-base-url and --vlm-model must be supplied together")
    if vlm_base_url is not None and vlm_crop_root is None:
        raise typer.BadParameter("--vlm-crop-root is required when VLM review is enabled")
    review_backend = (
        OpenAICompatibleReviewBackend(
            base_url=vlm_base_url,
            model=vlm_model,
            api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"),
            timeout=vlm_timeout,
            crop_root=vlm_crop_root,
        )
        if vlm_base_url is not None and vlm_model is not None
        else None
    )
    summary = correct_manifest(input_path, output_path, review_backend)
    typer.echo(json.dumps(asdict(summary), ensure_ascii=False))
