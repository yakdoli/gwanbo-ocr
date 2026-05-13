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
    help=(
        "Multi-method peer review: compare native text, pdfplumber, MarkItDown, "
        "PaddleOCR, PaddleOCR-VL, VLM."
    )
)
strategy_app = typer.Typer(help="Cluster PDF layouts and evaluate parsing strategies.")

app.add_typer(manifest_app, name="manifest")
app.add_typer(pdf_app, name="pdf")
app.add_typer(bench_app, name="bench")
app.add_typer(peer_app, name="peer")
app.add_typer(strategy_app, name="strategy")

convert_app = typer.Typer(help="Convert PDFs and documents to Markdown using MarkItDown.")
app.add_typer(convert_app, name="convert")


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


@convert_app.command("file")
def convert_file(
    input_path: Path = typer.Argument(..., help="Input PDF or document file path."),
    output: Path = typer.Option(..., help="Output Markdown file or directory."),
    mode: str = typer.Option("plain", "--mode", help="Conversion mode: plain or ocr-llm."),
    service_url: str | None = typer.Option(
        None, "--service-url", help="Optional MarkItDown HTTP service URL."
    ),
    llm_base_url: str | None = typer.Option(
        None, "--llm-base-url", help="OpenAI-compatible base URL for OCR+LLM mode."
    ),
    llm_model: str | None = typer.Option(
        None, "--llm-model", help="Vision-capable model for OCR+LLM mode."
    ),
    llm_api_key: str = typer.Option("dummy", "--llm-api-key", help="LLM API key."),
    llm_prompt: str | None = typer.Option(
        None, "--llm-prompt", help="Optional MarkItDown OCR prompt override."
    ),
    timeout: float = typer.Option(120.0, "--timeout", help="Conversion timeout in seconds."),
) -> None:
    """Convert a single document to Markdown."""
    from gwanbo_ocr.conversion import ConversionError, convert_document

    try:
        summary = convert_document(
            input_path=input_path,
            output=output,
            mode=mode,
            service_url=service_url,
            llm_base_url=llm_base_url,
            llm_model=llm_model,
            llm_api_key=llm_api_key,
            llm_prompt=llm_prompt,
            timeout_seconds=timeout,
        )
    except ConversionError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1) from exc
    _echo_summary(summary)


@convert_app.command("manifest")
def convert_manifest(
    input: Path = typer.Option(..., "--input", "--manifest", help="Input PDF manifest JSONL."),
    output: Path = typer.Option(..., help="Output directory for Markdown files."),
    mode: str = typer.Option("plain", "--mode", help="Conversion mode: plain or ocr-llm."),
    key: str = typer.Option("sample_id", help="Manifest key field for output filename."),
    pdf_path_field: str = typer.Option("pdf_path", help="Manifest field containing PDF path."),
    workers: int = typer.Option(1, help="Parallel worker count."),
    limit: int | None = typer.Option(None, help="Maximum rows to process."),
    skip_errors: bool = typer.Option(True, help="Skip files with conversion errors."),
    force: bool = typer.Option(False, help="Overwrite existing output files."),
    service_url: str | None = typer.Option(
        None, "--service-url", help="Optional MarkItDown HTTP service URL."
    ),
    llm_base_url: str | None = typer.Option(
        None, "--llm-base-url", help="OpenAI-compatible base URL for OCR+LLM mode."
    ),
    llm_model: str | None = typer.Option(
        None, "--llm-model", help="Vision-capable model for OCR+LLM mode."
    ),
    llm_api_key: str = typer.Option("dummy", "--llm-api-key", help="LLM API key."),
    llm_prompt: str | None = typer.Option(
        None, "--llm-prompt", help="Optional MarkItDown OCR prompt override."
    ),
    timeout: float = typer.Option(120.0, "--timeout", help="Per-file timeout in seconds."),
) -> None:
    """Batch convert PDFs from a manifest to Markdown files."""
    from gwanbo_ocr.conversion import convert_manifest as convert_manifest_rows

    summary = convert_manifest_rows(
        manifest_path=input,
        output_dir=output,
        mode=mode,
        key=key,
        pdf_path_field=pdf_path_field,
        workers=workers,
        limit=limit,
        skip_errors=skip_errors,
        force=force,
        service_url=service_url,
        llm_base_url=llm_base_url,
        llm_model=llm_model,
        llm_api_key=llm_api_key,
        llm_prompt=llm_prompt,
        timeout_seconds=timeout,
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
    enforce_strategy_routing: bool = typer.Option(
        True,
        "--enforce-strategy-routing/--no-enforce-strategy-routing",
        help="Route tasks by strategy and apply fallback policies.",
    ),
    preflight_vllm: bool = typer.Option(
        True,
        "--preflight-vllm/--no-preflight-vllm",
        help="Check VLM endpoint reachability before benchmark execution.",
    ),
    preflight_timeout_s: float = typer.Option(
        5.0,
        "--preflight-timeout-s",
        help="Timeout in seconds for VLM endpoint preflight check.",
    ),
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
        enforce_strategy_routing=enforce_strategy_routing,
        preflight_vllm=preflight_vllm,
        preflight_timeout_s=preflight_timeout_s,
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
    bench_scores: Path | None = typer.Option(
        None,
        "--bench-scores",
        help="Optional bench score JSONL path (for cer/wer/table_f1/critical_token_f1).",
    ),
    peer_score_report: Path | None = typer.Option(
        None,
        "--peer-score-report",
        help="Optional peer score report JSON (for critical_token_f1 fallback).",
    ),
    limit: int | None = typer.Option(None, help="Maximum clusters to evaluate."),
) -> None:
    """Evaluate assigned parsing strategies using available cluster metrics."""
    from gwanbo_ocr.strategy import evaluate_clusters

    summary = evaluate_clusters(
        clusters_path=clusters,
        output_dir=output,
        bench_scores_path=bench_scores,
        peer_score_report_path=peer_score_report,
        limit=limit,
    )
    _echo_summary(summary)


@strategy_app.command("pipeline")
def strategy_pipeline(
    manifest: Path = typer.Option(..., "--manifest", help="Input PDF manifest JSONL."),
    output: Path = typer.Option(..., help="Pipeline output root directory."),
    runner: str = typer.Option("qwen36_baseline", help="Bench runner/model alias."),
    base_url: str = typer.Option("http://127.0.0.1:8000/v1", help="OpenAI-compatible base URL."),
    api_key: str = typer.Option("dummy", help="OpenAI-compatible API key."),
    sample_per_bucket: int = typer.Option(
        20,
        "--sample-per-bucket",
        help="Max profile rows per theme/year/category bucket; 0 means all.",
    ),
    profile_max_pages: int = typer.Option(3, help="Pages to inspect for profiling; 0 means all."),
    render_max_pages: int = typer.Option(1, help="Pages to render per PDF; 0 means all selected."),
    cluster_sample_keys: int = typer.Option(
        20,
        "--cluster-sample-keys",
        help="Representative pdf keys to keep per cluster (for strategy suite mapping).",
    ),
    workers: int = typer.Option(1, help="Worker count for profile/peer stages."),
    render_workers: int = typer.Option(4, help="Worker count for render stage."),
    concurrency: int = typer.Option(4, help="Concurrent requests for bench run."),
    enforce_strategy_routing: bool = typer.Option(
        True,
        "--enforce-strategy-routing/--no-enforce-strategy-routing",
        help="Route tasks by strategy and apply fallback policies during bench run.",
    ),
    preflight_vllm: bool = typer.Option(
        True,
        "--preflight-vllm/--no-preflight-vllm",
        help="Check VLM endpoint reachability before benchmark execution.",
    ),
    preflight_timeout_s: float = typer.Option(
        5.0,
        "--preflight-timeout-s",
        help="Timeout in seconds for VLM endpoint preflight check.",
    ),
    run_peer: bool = typer.Option(True, "--peer/--no-peer", help="Run peer review stages."),
    run_paddle: bool = typer.Option(False, "--paddle/--no-paddle", help="Enable PaddleOCR peer."),
    run_paddle_vl: bool = typer.Option(
        False, "--paddle-vl/--no-paddle-vl", help="Enable PaddleOCR-VL peer."
    ),
    run_markitdown_ocr_llm: bool = typer.Option(
        False,
        "--markitdown-ocr-llm/--no-markitdown-ocr-llm",
        help="Enable MarkItDown OCR plugin with an OpenAI-compatible VLM.",
    ),
    markitdown_service_url: str | None = typer.Option(
        None,
        "--markitdown-service-url",
        help="Optional MarkItDown OCR API URL, e.g. http://127.0.0.1:8081.",
    ),
    markitdown_llm_base_url: str | None = typer.Option(
        None,
        "--markitdown-llm-base-url",
        help="OpenAI-compatible base URL used by MarkItDown OCR+LLM.",
    ),
    markitdown_llm_model: str | None = typer.Option(
        None,
        "--markitdown-llm-model",
        help="Vision-capable model name used by MarkItDown OCR+LLM.",
    ),
    markitdown_llm_api_key: str = typer.Option(
        "dummy", "--markitdown-llm-api-key", help="API key for MarkItDown OCR+LLM."
    ),
    markitdown_llm_prompt: str | None = typer.Option(
        None, "--markitdown-llm-prompt", help="Optional MarkItDown OCR prompt override."
    ),
    paddle_service_url: str | None = typer.Option(
        None,
        "--paddle-service-url",
        help="Optional PaddleOCR API URL for classic OCR, e.g. http://127.0.0.1:8082.",
    ),
    paddle_vl_service_url: str | None = typer.Option(
        None,
        "--paddle-vl-service-url",
        help="Optional PaddleOCR API URL for PaddleOCR-VL client calls.",
    ),
    paddle_vl_backend: str | None = typer.Option(
        None,
        "--paddle-vl-backend",
        help="PaddleOCR-VL recognition backend, e.g. vllm-server.",
    ),
    paddle_vl_server_url: str | None = typer.Option(
        None,
        "--paddle-vl-server-url",
        help="PaddleOCR-VL recognition server URL, e.g. http://127.0.0.1:8000/v1.",
    ),
    paddle_vl_model: str | None = typer.Option(
        None,
        "--paddle-vl-model",
        help="PaddleOCR-VL API model name for server-backed recognition.",
    ),
    paddle_vl_api_key: str | None = typer.Option(
        None, "--paddle-vl-api-key", help="PaddleOCR-VL recognition server API key."
    ),
    limit: int | None = typer.Option(None, help="Optional row/task cap for quick runs."),
) -> None:
    """Run end-to-end strategy pipeline from profiling to strategy evaluation."""
    from gwanbo_ocr.strategy import run_pipeline

    pipeline_summary = run_pipeline(
        manifest=manifest,
        output=output,
        runner=runner,
        base_url=base_url,
        api_key=api_key,
        sample_per_bucket=sample_per_bucket,
        profile_max_pages=profile_max_pages,
        render_max_pages=render_max_pages,
        cluster_sample_keys=cluster_sample_keys,
        workers=workers,
        render_workers=render_workers,
        concurrency=concurrency,
        enforce_strategy_routing=enforce_strategy_routing,
        preflight_vllm=preflight_vllm,
        preflight_timeout_s=preflight_timeout_s,
        run_peer=run_peer,
        run_paddle=run_paddle,
        run_paddle_vl=run_paddle_vl,
        run_markitdown_ocr_llm=run_markitdown_ocr_llm,
        markitdown_service_url=markitdown_service_url,
        markitdown_llm_base_url=markitdown_llm_base_url,
        markitdown_llm_model=markitdown_llm_model,
        markitdown_llm_api_key=markitdown_llm_api_key,
        markitdown_llm_prompt=markitdown_llm_prompt,
        paddle_service_url=paddle_service_url,
        paddle_vl_service_url=paddle_vl_service_url,
        paddle_vl_backend=paddle_vl_backend,
        paddle_vl_server_url=paddle_vl_server_url,
        paddle_vl_model=paddle_vl_model,
        paddle_vl_api_key=paddle_vl_api_key,
        limit=limit,
    )
    _echo_summary(
        {
            "status": "ok",
            "output": str(output),
            "pipeline_summary": str(output / "pipeline_summary.json"),
            "evaluated_clusters": pipeline_summary.get("strategy_eval", {}).get(
                "evaluated_clusters", 0
            ),
            "bench_tasks": pipeline_summary.get("bench_run", {}).get("tasks", 0),
        }
    )


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
    run_paddle_vl: bool = typer.Option(
        False, "--paddle-vl/--no-paddle-vl", help="Enable PaddleOCR-VL peer."
    ),
    run_markitdown_ocr_llm: bool = typer.Option(
        False,
        "--markitdown-ocr-llm/--no-markitdown-ocr-llm",
        help="Enable MarkItDown OCR plugin with an OpenAI-compatible VLM.",
    ),
    markitdown_service_url: str | None = typer.Option(
        None,
        "--markitdown-service-url",
        help="Optional MarkItDown OCR API URL, e.g. http://127.0.0.1:8081.",
    ),
    markitdown_llm_base_url: str | None = typer.Option(
        None,
        "--markitdown-llm-base-url",
        help="OpenAI-compatible base URL used by MarkItDown OCR+LLM.",
    ),
    markitdown_llm_model: str | None = typer.Option(
        None,
        "--markitdown-llm-model",
        help="Vision-capable model name used by MarkItDown OCR+LLM.",
    ),
    markitdown_llm_api_key: str = typer.Option(
        "dummy", "--markitdown-llm-api-key", help="API key for MarkItDown OCR+LLM."
    ),
    markitdown_llm_prompt: str | None = typer.Option(
        None, "--markitdown-llm-prompt", help="Optional MarkItDown OCR prompt override."
    ),
    paddle_service_url: str | None = typer.Option(
        None,
        "--paddle-service-url",
        help="Optional PaddleOCR API URL for classic OCR, e.g. http://127.0.0.1:8082.",
    ),
    paddle_vl_service_url: str | None = typer.Option(
        None,
        "--paddle-vl-service-url",
        help="Optional PaddleOCR API URL for PaddleOCR-VL client calls.",
    ),
    paddle_vl_backend: str | None = typer.Option(
        None,
        "--paddle-vl-backend",
        help="PaddleOCR-VL recognition backend, e.g. vllm-server.",
    ),
    paddle_vl_server_url: str | None = typer.Option(
        None,
        "--paddle-vl-server-url",
        help="PaddleOCR-VL recognition server URL, e.g. http://127.0.0.1:8000/v1.",
    ),
    paddle_vl_model: str | None = typer.Option(
        None,
        "--paddle-vl-model",
        help="PaddleOCR-VL API model name for server-backed recognition.",
    ),
    paddle_vl_api_key: str | None = typer.Option(
        None, "--paddle-vl-api-key", help="PaddleOCR-VL recognition server API key."
    ),
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
    from gwanbo_ocr.peers import run_peer_review_manifest

    summary = run_peer_review_manifest(
        manifest_path=manifest,
        output_dir=output,
        vlm_base_url=vlm_base_url,
        vlm_model=vlm_model,
        vlm_api_key=vlm_api_key,
        run_paddle=run_paddle,
        run_paddle_vl=run_paddle_vl,
        run_markitdown_ocr_llm=run_markitdown_ocr_llm,
        markitdown_service_url=markitdown_service_url,
        markitdown_llm_base_url=markitdown_llm_base_url,
        markitdown_llm_model=markitdown_llm_model,
        markitdown_llm_api_key=markitdown_llm_api_key,
        markitdown_llm_prompt=markitdown_llm_prompt,
        paddle_service_url=paddle_service_url,
        paddle_vl_service_url=paddle_vl_service_url,
        paddle_vl_backend=paddle_vl_backend,
        paddle_vl_server_url=paddle_vl_server_url,
        paddle_vl_model=paddle_vl_model,
        paddle_vl_api_key=paddle_vl_api_key,
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
    from gwanbo_ocr.pdf.io import write_json_atomic
    from gwanbo_ocr.peers import aggregate_peer_scores

    try:
        report = aggregate_peer_scores(review_dir)
    except ValueError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc

    output.mkdir(parents=True, exist_ok=True)
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
