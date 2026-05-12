"""Command line interface for gwanbo-ocr."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import typer

app = typer.Typer(help="PDF OCR and metadata extraction pipeline for Gwanbo artifacts.")
manifest_app = typer.Typer(help="Build immutable PDF manifests from source artifacts.")
pdf_app = typer.Typer(help="Classify PDFs, extract text layouts, and render pages.")
bench_app = typer.Typer(help="Run and score OCR/VLM benchmarks.")
peer_app = typer.Typer(
    help="Multi-method peer review: compare native text, pdfplumber, MarkItDown, PaddleOCR, VLM."
)
strategy_app = typer.Typer(help="Cluster PDF layouts and evaluate parsing strategies.")

app.add_typer(manifest_app, name="manifest")
app.add_typer(pdf_app, name="pdf")
app.add_typer(bench_app, name="bench")
app.add_typer(peer_app, name="peer")
app.add_typer(strategy_app, name="strategy")


@manifest_app.command("build")
def manifest_build(
    peti_root: Path = typer.Option(Path("/root/peti"), help="Read-only /root/peti project root."),
    output: Path = typer.Option(..., help="Output JSONL manifest path."),
    sources: str = typer.Option("pety,searchThema", help="Comma-separated sources to include."),
    limit: int | None = typer.Option(None, help="Maximum manifest rows to write."),
    include_issue_pdfs: bool = typer.Option(
        True, help="Include searchThema issue_pdfs fallback artifacts."
    ),
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


@pdf_app.command("profile")
def pdf_profile(
    input: Path = typer.Option(..., "--input", help="Input PDF manifest JSONL."),
    output: Path = typer.Option(..., help="Output profile directory."),
    max_pages: int = typer.Option(3, help="Pages to inspect per PDF; 0 means all pages."),
    workers: int = typer.Option(1, help="Worker count."),
    sample_per_bucket: int = typer.Option(
        20,
        "--sample-per-bucket",
        help="Maximum rows to profile per theme/year/category bucket; 0 means all rows.",
    ),
    table_strategy: str = typer.Option("auto", help="pdfplumber table extraction strategy."),
    limit: int | None = typer.Option(None, help="Maximum input rows to inspect before sampling."),
    only_missing: bool = typer.Option(False, help="Skip rows with existing sidecars."),
    force: bool = typer.Option(False, help="Regenerate existing sidecars."),
) -> None:
    """Build lightweight PDF feature profiles for layout clustering."""
    from gwanbo_ocr.pdf.profile import profile_manifest

    summary = profile_manifest(
        manifest_path=input,
        output_dir=output,
        max_pages=None if max_pages == 0 else max_pages,
        workers=workers,
        sample_per_bucket=None if sample_per_bucket == 0 else sample_per_bucket,
        table_strategy=table_strategy,
        limit=limit,
        only_missing=only_missing,
        force=force,
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


@strategy_app.command("cluster")
def strategy_cluster(
    profiles: Path = typer.Option(..., "--profiles", help="Input pdf profile manifest JSONL."),
    output: Path = typer.Option(..., help="Output cluster directory."),
    sample_keys: int = typer.Option(10, help="Representative PDF keys to keep per cluster."),
) -> None:
    """Cluster PDF profiles and assign deterministic parsing strategies."""
    from gwanbo_ocr.strategy import cluster_profiles

    summary = cluster_profiles(
        profiles_path=profiles,
        output_dir=output,
        sample_keys=sample_keys,
    )
    _echo_summary(summary)


@strategy_app.command("evaluate")
def strategy_evaluate(
    clusters: Path = typer.Option(..., "--clusters", help="Input layout cluster manifest JSONL."),
    output: Path = typer.Option(..., help="Output strategy evaluation directory."),
    limit: int | None = typer.Option(None, help="Maximum clusters to evaluate."),
) -> None:
    """Evaluate assigned parsing strategies using available cluster metrics."""
    from gwanbo_ocr.strategy import evaluate_clusters

    summary = evaluate_clusters(
        clusters_path=clusters,
        output_dir=output,
        limit=limit,
    )
    _echo_summary(summary)


@peer_app.command("run")
def peer_run(
    manifest: Path = typer.Option(
        ..., "--manifest", help="Input JSONL manifest (from manifest build)."
    ),
    output: Path = typer.Option(..., help="Output directory for peer-review sidecars and index."),
    vlm_base_url: str | None = typer.Option(
        None, "--vlm-base-url", help="OpenAI-compatible VLM base URL (enables VLM peer)."
    ),
    vlm_model: str | None = typer.Option(
        None, "--vlm-model", help="Model name/alias for the VLM runner."
    ),
    vlm_api_key: str = typer.Option("dummy", "--vlm-api-key", help="API key for the VLM endpoint."),
    run_paddle: bool = typer.Option(False, "--paddle/--no-paddle", help="Enable PaddleOCR peer."),
    skip_markitdown: bool = typer.Option(False, "--skip-markitdown", help="Skip MarkItDown peer."),
    skip_pdfplumber: bool = typer.Option(False, "--skip-pdfplumber", help="Skip pdfplumber peer."),
    skip_native: bool = typer.Option(
        False, "--skip-native", help="Skip native text extraction peer."
    ),
    max_pages: int = typer.Option(1, help="Pages to process per PDF; 0 means all."),
    dpi: int = typer.Option(200, help="Render DPI for image-based OCR peers."),
    workers: int = typer.Option(1, help="Parallel worker count."),
    limit: int | None = typer.Option(None, help="Maximum rows to process."),
    force: bool = typer.Option(False, help="Regenerate existing sidecars."),
    timeout: int = typer.Option(60, "--timeout", help="Per-method timeout in seconds."),
    progress_every: int = typer.Option(0, help="Log progress every N items."),
) -> None:
    """Run multi-method peer review on a PDF manifest."""
    from gwanbo_ocr.peer_review import run_peer_review_manifest

    summary = run_peer_review_manifest(
        manifest_path=manifest,
        output_dir=output,
        vlm_base_url=vlm_base_url,
        vlm_model=vlm_model,
        vlm_api_key=vlm_api_key,
        run_paddle=run_paddle,
        run_markitdown=not skip_markitdown,
        run_pdfplumber=not skip_pdfplumber,
        run_native_text=not skip_native,
        max_pages=None if max_pages == 0 else max_pages,
        dpi=dpi,
        workers=workers,
        limit=limit,
        force=force,
        timeout_seconds=timeout,
        progress_every=progress_every,
    )
    _echo_summary(summary)


@peer_app.command("score")
def peer_score(
    review_dir: Path = typer.Option(
        ..., "--review-dir", help="Peer review output directory (from peer run)."
    ),
    output: Path = typer.Option(..., help="Report output directory."),
) -> None:
    """Aggregate peer review scores across all sidecars and write a report."""
    from gwanbo_ocr.pdf.io import read_json, write_json_atomic

    index_path = review_dir / "metadata.json"
    index = read_json(index_path) or {}
    if not index:
        typer.echo(f"No metadata.json found in {review_dir}", err=True)
        raise typer.Exit(1)

    output.mkdir(parents=True, exist_ok=True)
    method_counts: dict[str, int] = {}
    decision_counts: dict[str, int] = {}
    f1_by_method: dict[str, list[float]] = {}

    for entry in index.values():
        best = str(entry.get("best_text_method") or "none")
        method_counts[best] = method_counts.get(best, 0) + 1
        needs_ocr = bool((entry.get("decision") or {}).get("needs_ocr"))
        key = "needs_ocr" if needs_ocr else "text_layer"
        decision_counts[key] = decision_counts.get(key, 0) + 1
        for name, score in (entry.get("peer_summaries") or {}).items():
            if isinstance(score, dict) and score.get("critical_token_f1") is not None:
                f1_by_method.setdefault(name, []).append(float(score["critical_token_f1"]))

    avg_f1 = {name: round(sum(vals) / len(vals), 4) for name, vals in f1_by_method.items() if vals}
    report = {
        "total": len(index),
        "by_best_method": method_counts,
        "by_decision": decision_counts,
        "avg_critical_token_f1_by_method": avg_f1,
        "review_dir": str(review_dir),
    }
    write_json_atomic(output / "peer_score_report.json", report)
    _echo_summary(report)


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
