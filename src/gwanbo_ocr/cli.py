"""Command line interface for gwanbo-ocr."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import typer

app = typer.Typer(help="PDF OCR and metadata extraction pipeline for Gwanbo artifacts.")
manifest_app = typer.Typer(help="Build immutable PDF manifests from source artifacts.")
pdf_app = typer.Typer(help="Classify PDFs, extract text layouts, and render pages.")
bench_app = typer.Typer(help="Run and score OCR/VLM benchmarks.")

app.add_typer(manifest_app, name="manifest")
app.add_typer(pdf_app, name="pdf")
app.add_typer(bench_app, name="bench")


@manifest_app.command("build")
def manifest_build(
    peti_root: Path = typer.Option(Path("/root/peti"), help="Read-only /root/peti project root."),
    output: Path = typer.Option(..., help="Output JSONL manifest path."),
    sources: str = typer.Option("pety,searchThema", help="Comma-separated sources to include."),
    limit: int | None = typer.Option(None, help="Maximum manifest rows to write."),
    include_issue_pdfs: bool = typer.Option(True, help="Include searchThema issue_pdfs fallback artifacts."),
) -> None:
    """Build a compact JSONL manifest from /root/peti artifacts."""
    from gwanbo_ocr.manifest import build_peti_manifest

    summary = build_peti_manifest(
        peti_root=peti_root,
        output_path=output,
        sources=[source.strip() for source in sources.split(",") if source.strip()],
        limit=limit,
        include_issue_pdfs=include_issue_pdfs,
    )
    _echo_summary(summary)


@pdf_app.command("classify")
def pdf_classify(
    input: Path = typer.Option(..., "--input", help="Input PDF manifest JSONL."),
    output: Path = typer.Option(..., help="Output classification directory."),
    max_pages: int = typer.Option(3, help="Pages to inspect per PDF; 0 means all pages."),
    workers: int = typer.Option(1, help="Worker count."),
    method: str = typer.Option("extract-text", help="Classification method."),
    only_missing: bool = typer.Option(False, help="Skip rows with existing sidecars."),
    force: bool = typer.Option(False, help="Regenerate existing sidecars."),
    limit: int | None = typer.Option(None, help="Maximum rows to process."),
) -> None:
    """Classify PDFs as native-text, image/unextractable, or invalid."""
    from gwanbo_ocr.pdf.classification import classify_manifest

    summary = classify_manifest(
        manifest_path=input,
        output_dir=output,
        max_pages=None if max_pages == 0 else max_pages,
        workers=workers,
        method=method,
        only_missing=only_missing,
        force=force,
        limit=limit,
    )
    _echo_summary(summary)


@pdf_app.command("layout")
def pdf_layout(
    classification: Path = typer.Option(..., help="Classification manifest JSONL."),
    output: Path = typer.Option(..., help="Output layout directory."),
    max_pages: int = typer.Option(3, help="Pages to inspect per PDF; 0 means all pages."),
    table_strategy: str = typer.Option("auto", help="pdfplumber table extraction strategy."),
    workers: int = typer.Option(1, help="Worker count."),
    only_missing: bool = typer.Option(False, help="Skip rows with existing sidecars."),
    force: bool = typer.Option(False, help="Regenerate existing sidecars."),
    limit: int | None = typer.Option(None, help="Maximum rows to process."),
) -> None:
    """Generate layout and table sidecars for text-extractable PDFs."""
    from gwanbo_ocr.pdf.layout import generate_layout_manifest

    summary = generate_layout_manifest(
        classification_manifest=classification,
        output_dir=output,
        max_pages=None if max_pages == 0 else max_pages,
        table_strategy=table_strategy,
        workers=workers,
        only_missing=only_missing,
        force=force,
        limit=limit,
    )
    _echo_summary(summary)


@pdf_app.command("validate")
def pdf_validate(
    input: Path = typer.Option(..., "--input", help="Input PDF manifest JSONL."),
    output: Path = typer.Option(..., help="Output validation directory."),
    limit: int | None = typer.Option(None, help="Maximum rows to process."),
) -> None:
    """Validate manifest paths, PDF headers, EOF markers, and hashes."""
    from gwanbo_ocr.pdf.integrity import validate_manifest

    summary = validate_manifest(manifest_path=input, output_dir=output, limit=limit)
    _echo_summary(summary)


@pdf_app.command("render")
def pdf_render(
    input: Path = typer.Option(..., "--input", help="Classification or sample manifest JSONL."),
    output: Path = typer.Option(..., help="Output image directory."),
    dpi: int = typer.Option(200, help="Render DPI."),
    max_long_edge: int = typer.Option(2400, help="Resize rendered images to this max long edge."),
    max_pages: int = typer.Option(1, help="Pages to render per PDF; 0 means selected/all pages."),
    workers: int = typer.Option(4, help="Worker count."),
    limit: int | None = typer.Option(None, help="Maximum rows to process."),
) -> None:
    """Render PDF pages to PNG images for OCR/VLM runners."""
    from gwanbo_ocr.render import render_manifest

    summary = render_manifest(
        manifest_path=input,
        output_dir=output,
        dpi=dpi,
        max_long_edge=max_long_edge,
        max_pages=None if max_pages == 0 else max_pages,
        workers=workers,
        limit=limit,
    )
    _echo_summary(summary)


@bench_app.command("run")
def bench_run(
    suite: str = typer.Option("smoke", help="Suite name or sample JSONL path."),
    runner: str = typer.Option("qwen36_baseline", help="Runner/model config name."),
    run_dir: Path = typer.Option(Path("runs/current"), help="Run output directory."),
    base_url: str = typer.Option("http://127.0.0.1:8000/v1", help="OpenAI-compatible base URL."),
    api_key: str = typer.Option("dummy", help="OpenAI-compatible API key."),
    concurrency: int = typer.Option(4, help="Concurrent inference requests."),
    limit: int | None = typer.Option(None, help="Maximum page tasks to process."),
) -> None:
    """Run OCR/VLM benchmark tasks for one runner."""
    from gwanbo_ocr.bench import run_benchmark

    summary = run_benchmark(
        suite=suite,
        runner_name=runner,
        run_dir=run_dir,
        base_url=base_url,
        api_key=api_key,
        concurrency=concurrency,
        limit=limit,
    )
    _echo_summary(summary)


@bench_app.command("score")
def bench_score(
    run: Path = typer.Option(..., help="Run directory containing benchmark results."),
    output: Path = typer.Option(..., help="Report output directory."),
) -> None:
    """Score benchmark outputs and write aggregate reports."""
    from gwanbo_ocr.bench import score_benchmark

    summary = score_benchmark(run_dir=run, output_dir=output)
    _echo_summary(summary)


def _echo_summary(summary: Any) -> None:
    if hasattr(summary, "to_display"):
        typer.echo(summary.to_display())
        return
    if isinstance(summary, dict):
        import json

        typer.echo(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
        return
    typer.echo(str(summary))


if __name__ == "__main__":
    app()
