"""Multi-method PDF extraction peer review.

Five extraction peers are compared side-by-side for each PDF:

    native_text  — pypdf/PyPDF2 direct text extraction
    pdfplumber   — pdfplumber page.extract_text()
    markitdown   — MarkItDown.convert() markdown output
    paddle_ocr   — PyMuPDF render → PaddleOCR
    vlm_ocr      — PyMuPDF render → OpenAI-compatible VLM endpoint

Each peer result conforms to this schema:
    status          ok | error | skipped
    method          human-readable method identifier
    text_extractable bool
    text_chars      int
    sample_text     str (up to sample_chars)
    error           str | None
    ...method-specific extras
"""

from __future__ import annotations

import concurrent.futures
import difflib
import re
import signal
from datetime import datetime
from pathlib import Path
from typing import Any

# Optional heavy dependencies — imported lazily so the module loads even when
# the [pdf], [qwen], or [paddleocr] extras are not installed.
_MarkItDown: Any
try:
    from markitdown import MarkItDown as _MarkItDown  # type: ignore[import-not-found]
except ImportError:
    _MarkItDown = None

_pdfplumber: Any
try:
    import pdfplumber as _pdfplumber  # type: ignore[import-not-found]
except ImportError:
    _pdfplumber = None

SAMPLE_CHARS_DEFAULT = 1200
METHOD_PREFERENCE = {
    "vlm_ocr": 5,
    "paddle_ocr": 4,
    "pdfplumber": 3,
    "markitdown": 2,
    "native_text": 1,
}


# ---------------------------------------------------------------------------
# Extraction peers
# ---------------------------------------------------------------------------


def extract_native_text(
    pdf_path: Path,
    *,
    max_pages: int | None = 1,
    sample_chars: int = SAMPLE_CHARS_DEFAULT,
    timeout_seconds: int = 30,
) -> dict[str, Any]:
    """Extract text using pypdf / builtin parser."""
    from gwanbo_ocr.pdf.text import analyze_pdf_text

    metadata = analyze_pdf_text(
        pdf_path,
        include_sample=True,
        sample_chars=sample_chars,
        max_pages=max_pages,
        timeout_seconds=timeout_seconds,
    )
    sample = str(metadata.get("sample_text") or "")
    return {
        "status": str(metadata.get("status") or "unknown"),
        "method": "pypdf.extract_text",
        "text_extractable": bool(metadata.get("text_extractable")),
        "pages": metadata.get("pages"),
        "scanned_pages": metadata.get("scanned_pages"),
        "text_chars": int(metadata.get("total_chars") or len(sample)),
        "sample_text": sample[:sample_chars],
        "error": metadata.get("error"),
    }


def extract_pdfplumber(
    pdf_path: Path,
    *,
    max_pages: int | None = 1,
    sample_chars: int = SAMPLE_CHARS_DEFAULT,
    timeout_seconds: int = 30,
) -> dict[str, Any]:
    """Extract text using pdfplumber.page.extract_text()."""
    plumber = _pdfplumber
    if plumber is None:
        return _skipped("pdfplumber is not installed", method="pdfplumber.extract_text")

    def _run() -> dict[str, Any]:
        with plumber.open(str(pdf_path)) as doc:
            page_count = len(doc.pages)
            pages_to_scan = page_count if max_pages is None else min(page_count, max_pages)
            parts: list[str] = []
            total_chars = 0
            page_errors: list[dict[str, Any]] = []
            for idx in range(pages_to_scan):
                try:
                    text = doc.pages[idx].extract_text() or ""
                    text = _normalize(text)
                    total_chars += len(text)
                    if text and len(" ".join(parts)) < sample_chars:
                        parts.append(text)
                except Exception as exc:  # noqa: BLE001
                    page_errors.append({"page_index": idx, "error": str(exc)})
            result: dict[str, Any] = {
                "status": "ok",
                "method": "pdfplumber.extract_text",
                "text_extractable": total_chars > 0,
                "pages": page_count,
                "scanned_pages": pages_to_scan,
                "text_chars": total_chars,
                "sample_text": " ".join(parts)[:sample_chars],
                "error": None,
            }
            if page_errors:
                result["page_errors"] = page_errors
                result["page_error_count"] = len(page_errors)
            return result

    return _with_timeout(_run, timeout_seconds, method="pdfplumber.extract_text")


def extract_markitdown(
    pdf_path: Path,
    *,
    sample_chars: int = SAMPLE_CHARS_DEFAULT,
    timeout_seconds: int = 30,
) -> dict[str, Any]:
    """Extract text using MarkItDown.convert()."""
    MarkItDown = _MarkItDown
    if MarkItDown is None:
        return _skipped("markitdown is not installed", method="MarkItDown.convert")

    def _run() -> dict[str, Any]:
        converter = MarkItDown(enable_plugins=False)
        converted = converter.convert(str(pdf_path))
        text = _normalize(str(getattr(converted, "text_content", "") or ""))
        return {
            "status": "ok",
            "method": "MarkItDown.convert",
            "text_extractable": bool(text),
            "text_chars": len(text),
            "sample_text": text[:sample_chars],
            "error": None,
        }

    return _with_timeout(_run, timeout_seconds, method="MarkItDown.convert")


def extract_paddle_ocr(
    pdf_path: Path,
    *,
    image_dir: Path,
    max_pages: int | None = 1,
    sample_chars: int = SAMPLE_CHARS_DEFAULT,
    dpi: int = 200,
    lang: str = "korean",
    timeout_seconds: int = 60,
) -> dict[str, Any]:
    """Render pages with PyMuPDF and transcribe with PaddleOCR."""
    try:
        from gwanbo_ocr.runners.paddle import PaddleOcrRunner
    except ImportError:
        return _skipped("paddleocr is not installed", method="PaddleOCR")

    rendered = _render_pages(pdf_path, image_dir=image_dir, max_pages=max_pages, dpi=dpi)
    if rendered.get("status") != "ok":
        return _error(str(rendered.get("error", "render failed")), method="PaddleOCR")

    def _run() -> dict[str, Any]:
        runner = PaddleOcrRunner(lang=lang)
        parts: list[str] = []
        total_chars = 0
        page_results: list[dict[str, Any]] = []
        for page in rendered["pages"]:
            try:
                result = runner.transcribe(Path(page["path"]))
                text = _normalize(result.text)
                total_chars += len(text)
                if text and len(" ".join(parts)) < sample_chars:
                    parts.append(text)
                page_results.append(
                    {"page_index": page["page_index"], "status": "ok", "text_chars": len(text)}
                )
            except Exception as exc:  # noqa: BLE001
                page_results.append(
                    {"page_index": page["page_index"], "status": "error", "error": str(exc)}
                )
        return {
            "status": "ok",
            "method": "PaddleOCR",
            "text_extractable": total_chars > 0,
            "pages": rendered["page_count"],
            "scanned_pages": len(rendered["pages"]),
            "text_chars": total_chars,
            "sample_text": " ".join(parts)[:sample_chars],
            "dpi": dpi,
            "lang": lang,
            "page_results": page_results,
            "error": None,
        }

    return _with_timeout(_run, timeout_seconds, method="PaddleOCR")


def extract_vlm_ocr(
    pdf_path: Path,
    *,
    image_dir: Path,
    runner: Any,
    max_pages: int | None = 1,
    sample_chars: int = SAMPLE_CHARS_DEFAULT,
    dpi: int = 200,
    timeout_seconds: int = 120,
) -> dict[str, Any]:
    """Render pages with PyMuPDF and transcribe with an OpenAI-compatible VLM."""
    rendered = _render_pages(pdf_path, image_dir=image_dir, max_pages=max_pages, dpi=dpi)
    if rendered.get("status") != "ok":
        return _error(str(rendered.get("error", "render failed")), method="VLM-OCR")

    method_label = f"VLM-OCR({getattr(runner, 'model', 'unknown')})"

    def _run() -> dict[str, Any]:
        parts: list[str] = []
        total_chars = 0
        page_results: list[dict[str, Any]] = []
        for page in rendered["pages"]:
            try:
                result = runner.transcribe(Path(page["path"]), page_number=page["page_index"] + 1)
                text = _normalize(result.text)
                total_chars += len(text)
                if text and len(" ".join(parts)) < sample_chars:
                    parts.append(text)
                page_results.append(
                    {
                        "page_index": page["page_index"],
                        "status": "ok",
                        "text_chars": len(text),
                        "latency_ms": result.data.get("latency_ms"),
                    }
                )
            except Exception as exc:  # noqa: BLE001
                page_results.append(
                    {"page_index": page["page_index"], "status": "error", "error": str(exc)}
                )
        return {
            "status": "ok",
            "method": method_label,
            "text_extractable": total_chars > 0,
            "pages": rendered["page_count"],
            "scanned_pages": len(rendered["pages"]),
            "text_chars": total_chars,
            "sample_text": " ".join(parts)[:sample_chars],
            "dpi": dpi,
            "model": getattr(runner, "model", None),
            "page_results": page_results,
            "error": None,
        }

    return _with_timeout(_run, timeout_seconds, method=method_label)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def analyze_pdf_peer_review(
    pdf_path: Path,
    *,
    image_dir: Path,
    vlm_runner: Any | None = None,
    max_pages: int | None = 1,
    sample_chars: int = SAMPLE_CHARS_DEFAULT,
    dpi: int = 200,
    timeout_seconds: int = 60,
    run_paddle: bool = False,
    run_markitdown: bool = True,
    run_pdfplumber: bool = True,
    run_native_text: bool = True,
    ocr_lang: str = "korean",
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run all enabled peers and produce a complete peer-review report."""
    report: dict[str, Any] = {
        "path": str(pdf_path),
        "filename": pdf_path.name,
        "status": "ok",
        "analysis_scope": _scope(max_pages),
        "generated_at": _iso_now(),
        "peers": {},
        "review": {},
        "score": {},
        "decision": {},
    }

    peers = report["peers"]

    if run_native_text:
        peers["native_text"] = extract_native_text(
            pdf_path,
            max_pages=max_pages,
            sample_chars=sample_chars,
            timeout_seconds=timeout_seconds,
        )
    if run_pdfplumber:
        peers["pdfplumber"] = extract_pdfplumber(
            pdf_path,
            max_pages=max_pages,
            sample_chars=sample_chars,
            timeout_seconds=timeout_seconds,
        )
    if run_markitdown:
        peers["markitdown"] = extract_markitdown(
            pdf_path, sample_chars=sample_chars, timeout_seconds=timeout_seconds
        )
    if run_paddle:
        peers["paddle_ocr"] = extract_paddle_ocr(
            pdf_path,
            image_dir=image_dir / "paddle",
            max_pages=max_pages,
            sample_chars=sample_chars,
            dpi=dpi,
            lang=ocr_lang,
            timeout_seconds=timeout_seconds * 2,
        )
    if vlm_runner is not None:
        peers["vlm_ocr"] = extract_vlm_ocr(
            pdf_path,
            image_dir=image_dir / "vlm",
            runner=vlm_runner,
            max_pages=max_pages,
            sample_chars=sample_chars,
            dpi=dpi,
            timeout_seconds=timeout_seconds * 3,
        )

    report["review"] = review_extraction_peers(peers)
    if metadata:
        report["score"] = score_against_metadata(peers, metadata)
    report["decision"] = decide_extraction(peers, report["review"])

    if all(p.get("status") in {"error", "skipped"} for p in peers.values()):
        report["status"] = "error"
        report["error"] = "all extraction methods failed or were skipped"

    return report


def review_extraction_peers(peers: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Cross-compare all extraction peers."""
    summaries: dict[str, dict[str, Any]] = {}
    warnings: list[str] = []

    for name, peer in peers.items():
        status = str(peer.get("status") or "unknown")
        text_chars = int(peer.get("text_chars") or 0)
        summaries[name] = {
            "status": status,
            "text_chars": text_chars,
            "text_extractable": bool(peer.get("text_extractable")),
            "error": peer.get("error"),
        }
        if status == "error":
            warnings.append(f"{name} failed: {peer.get('error')}")
        elif status == "skipped":
            warnings.append(f"{name} skipped: {peer.get('skip_reason')}")

    best = _choose_best_method(peers)
    similarities = _pairwise_similarities(peers)

    if best:
        best_chars = int(peers[best].get("text_chars") or 0)
        for name, peer in peers.items():
            if name == best or peer.get("status") != "ok":
                continue
            chars = int(peer.get("text_chars") or 0)
            if best_chars and chars < best_chars * 0.3:
                warnings.append(f"{name} produced much less text than {best}")
    else:
        warnings.append("no successful extraction method produced text")

    return {
        "best_text_method": best,
        "peer_summaries": summaries,
        "pairwise_sample_similarity": similarities,
        "warnings": warnings,
    }


def score_against_metadata(
    peers: dict[str, dict[str, Any]],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """Score each peer's sample text against critical tokens from metadata."""
    from gwanbo_ocr.metrics import critical_token_f1, extract_critical_tokens

    token_fields = ("title", "category", "agency")
    reference_text = " ".join(str(metadata.get(f) or "") for f in token_fields if metadata.get(f))
    reference_tokens = extract_critical_tokens(reference_text)

    scores: dict[str, Any] = {"reference_tokens": sorted(reference_tokens)}
    for name, peer in peers.items():
        if peer.get("status") != "ok":
            scores[name] = {"status": peer.get("status"), "critical_token_f1": None}
            continue
        sample = str(peer.get("sample_text") or "")
        candidate_tokens = extract_critical_tokens(sample)
        f1 = critical_token_f1(reference_tokens, candidate_tokens)
        scores[name] = {
            "status": "ok",
            "critical_token_f1": round(f1.f1, 4),
            "precision": round(f1.precision, 4),
            "recall": round(f1.recall, 4),
        }

    ranked = [
        (name, scores[name]["critical_token_f1"])
        for name in scores
        if name != "reference_tokens" and scores[name].get("critical_token_f1") is not None
    ]
    ranked.sort(key=lambda item: item[1], reverse=True)
    scores["ranked_by_f1"] = [name for name, _ in ranked]
    return scores


def decide_extraction(
    peers: dict[str, dict[str, Any]],
    review: dict[str, Any],
) -> dict[str, Any]:
    """Recommend the best extraction strategy."""
    best_method = review.get("best_text_method")
    text_layer = ("native_text", "pdfplumber", "markitdown")
    ocr_methods = ("paddle_ocr", "vlm_ocr")

    text_layer_ok = any(
        peers.get(m, {}).get("status") == "ok" and int(peers.get(m, {}).get("text_chars") or 0) > 0
        for m in text_layer
    )
    ocr_ok = any(
        peers.get(m, {}).get("status") == "ok" and int(peers.get(m, {}).get("text_chars") or 0) > 0
        for m in ocr_methods
    )

    if text_layer_ok:
        return {
            "text_extractable": True,
            "preferred_text_source": best_method,
            "needs_ocr": False,
            "reason": "text layer produced usable text",
        }
    if ocr_ok:
        return {
            "text_extractable": True,
            "preferred_text_source": best_method,
            "needs_ocr": True,
            "reason": "image OCR produced text; no native text layer",
        }
    return {
        "text_extractable": False,
        "preferred_text_source": None,
        "needs_ocr": True,
        "reason": "no extraction method produced text",
    }


# ---------------------------------------------------------------------------
# Batch manifest processing
# ---------------------------------------------------------------------------


def run_peer_review_manifest(
    manifest_path: Path,
    output_dir: Path,
    *,
    vlm_base_url: str | None = None,
    vlm_model: str | None = None,
    vlm_api_key: str = "dummy",
    run_paddle: bool = False,
    run_markitdown: bool = True,
    run_pdfplumber: bool = True,
    run_native_text: bool = True,
    max_pages: int | None = 1,
    dpi: int = 200,
    workers: int = 1,
    limit: int | None = None,
    force: bool = False,
    timeout_seconds: int = 60,
    progress_every: int = 0,
) -> dict[str, Any]:
    """Process a JSONL manifest and write peer-review sidecars + index."""
    from gwanbo_ocr.pdf.io import read_jsonl, resolve_pdf_path, write_json_atomic

    vlm_runner = _make_vlm_runner(vlm_base_url, vlm_model, vlm_api_key) if vlm_base_url else None

    output_dir.mkdir(parents=True, exist_ok=True)
    items_dir = output_dir / "items"
    images_dir = output_dir / "images"

    summary: dict[str, Any] = {
        "manifest": str(manifest_path),
        "output_dir": str(output_dir),
        "started_at": _iso_now(),
        "total": 0,
        "processed": 0,
        "skipped_existing": 0,
        "errors": 0,
        "by_best_method": {},
        "by_decision": {},
        "settings": {
            "max_pages": max_pages,
            "dpi": dpi,
            "workers": workers,
            "run_native_text": run_native_text,
            "run_pdfplumber": run_pdfplumber,
            "run_markitdown": run_markitdown,
            "run_paddle": run_paddle,
            "vlm_model": vlm_model,
        },
    }

    work_items: list[dict[str, Any]] = []
    for row in read_jsonl(manifest_path):
        if limit is not None and summary["total"] >= limit:
            break
        summary["total"] += 1
        pdf_path_text = str(resolve_pdf_path(row, peti_root="/root/peti"))
        sample_id = str(row.get("sample_id") or row.get("id") or "unknown")
        sidecar_path = items_dir / f"{sample_id}.json"
        if sidecar_path.exists() and not force:
            summary["skipped_existing"] += 1
            continue
        work_items.append(
            {
                "row": row,
                "sample_id": sample_id,
                "pdf_path": pdf_path_text,
                "sidecar_path": str(sidecar_path),
                "image_dir": str(images_dir / sample_id),
                "vlm_runner": vlm_runner,
                "run_paddle": run_paddle,
                "run_markitdown": run_markitdown,
                "run_pdfplumber": run_pdfplumber,
                "run_native_text": run_native_text,
                "max_pages": max_pages,
                "dpi": dpi,
                "timeout_seconds": timeout_seconds,
            }
        )

    index: dict[str, Any] = {}
    for result in _bounded_process(work_items, workers):
        sample_id = result["sample_id"]
        report = result["report"]
        sidecar_path = Path(result["sidecar_path"])
        write_json_atomic(sidecar_path, report)
        summary["processed"] += 1
        if report.get("status") == "error":
            summary["errors"] += 1
        best = (report.get("review") or {}).get("best_text_method") or "none"
        summary["by_best_method"][best] = summary["by_best_method"].get(best, 0) + 1
        decision_key = (
            "needs_ocr" if (report.get("decision") or {}).get("needs_ocr") else "text_layer"
        )
        summary["by_decision"][decision_key] = summary["by_decision"].get(decision_key, 0) + 1
        index[sample_id] = _compact_index(report, sidecar_path)
        if progress_every and summary["processed"] % progress_every == 0:
            print(f"processed={summary['processed']} errors={summary['errors']}", flush=True)

    summary["completed_at"] = _iso_now()
    write_json_atomic(output_dir / "metadata.json", index)
    write_json_atomic(output_dir / "summary.json", summary)
    return summary


def aggregate_peer_scores(review_dir: Path) -> dict[str, Any]:
    """Aggregate method-level peer scores from peer-review sidecars."""
    from gwanbo_ocr.pdf.io import read_json

    index_path = review_dir / "metadata.json"
    index = read_json(index_path) or {}
    if not isinstance(index, dict) or not index:
        raise ValueError(f"No metadata.json found in {review_dir}")

    method_counts: dict[str, int] = {}
    decision_counts: dict[str, int] = {}
    f1_by_method: dict[str, list[float]] = {}

    for entry in index.values():
        if not isinstance(entry, dict):
            continue
        best = str(entry.get("best_text_method") or "none")
        method_counts[best] = method_counts.get(best, 0) + 1
        needs_ocr = bool((entry.get("decision") or {}).get("needs_ocr"))
        decision_key = "needs_ocr" if needs_ocr else "text_layer"
        decision_counts[decision_key] = decision_counts.get(decision_key, 0) + 1

        sidecar_path = Path(str(entry.get("sidecar_path") or ""))
        if not sidecar_path.is_file():
            continue
        sidecar = read_json(sidecar_path) or {}
        if not isinstance(sidecar, dict):
            continue
        score_payload = sidecar.get("score")
        score = score_payload if isinstance(score_payload, dict) else {}
        for method, metrics in score.items():
            if method in {"reference_tokens", "ranked_by_f1"}:
                continue
            if not isinstance(metrics, dict):
                continue
            value = metrics.get("critical_token_f1")
            try:
                if value is not None:
                    f1_by_method.setdefault(str(method), []).append(float(value))
            except (TypeError, ValueError):
                continue

    avg_f1 = {
        method: round(sum(values) / len(values), 4)
        for method, values in f1_by_method.items()
        if values
    }
    return {
        "total": len(index),
        "by_best_method": method_counts,
        "by_decision": decision_counts,
        "avg_critical_token_f1_by_method": avg_f1,
        "review_dir": str(review_dir),
    }


def _process_work_item(work_item: dict[str, Any]) -> dict[str, Any]:
    row = work_item["row"]
    pdf_path = Path(work_item["pdf_path"])
    metadata = {k: row.get(k) for k in ("title", "category", "agency", "id") if row.get(k)}

    report = analyze_pdf_peer_review(
        pdf_path,
        image_dir=Path(work_item["image_dir"]),
        vlm_runner=work_item.get("vlm_runner"),
        max_pages=work_item.get("max_pages"),
        dpi=int(work_item.get("dpi") or 200),
        timeout_seconds=int(work_item.get("timeout_seconds") or 60),
        run_paddle=bool(work_item.get("run_paddle")),
        run_markitdown=bool(work_item.get("run_markitdown", True)),
        run_pdfplumber=bool(work_item.get("run_pdfplumber", True)),
        run_native_text=bool(work_item.get("run_native_text", True)),
        metadata=metadata or None,
    )
    report.update(
        {
            "sample_id": work_item["sample_id"],
            "source": row.get("source") or row.get("theme"),
            "pdf_key": row.get("pdf_key") or row.get("id"),
        }
    )
    return {
        "sample_id": work_item["sample_id"],
        "sidecar_path": work_item["sidecar_path"],
        "report": report,
    }


def _bounded_process(
    work_items: list[dict[str, Any]],
    workers: int,
) -> Any:
    if workers <= 1:
        for item in work_items:
            yield _process_work_item(item)
        return

    max_pending = max(workers * 2, workers)
    iterator = iter(work_items)
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        pending: set[concurrent.futures.Future[dict[str, Any]]] = set()
        for _ in range(min(max_pending, len(work_items))):
            try:
                item = next(iterator)
            except StopIteration:
                break
            pending.add(executor.submit(_process_work_item, item))

        while pending:
            done, pending = concurrent.futures.wait(
                pending, return_when=concurrent.futures.FIRST_COMPLETED
            )
            for future in done:
                yield future.result()
                try:
                    item = next(iterator)
                except StopIteration:
                    continue
                pending.add(executor.submit(_process_work_item, item))


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _render_pages(
    pdf_path: Path,
    *,
    image_dir: Path,
    max_pages: int | None,
    dpi: int,
) -> dict[str, Any]:
    """Render PDF pages to PNG files and return page list."""
    try:
        from gwanbo_ocr.render import page_count, render_page_to_png_bytes
    except ImportError:
        return {"status": "error", "error": "PyMuPDF (pymupdf) is not installed"}

    try:
        image_dir.mkdir(parents=True, exist_ok=True)
        total = page_count(pdf_path)
        pages_to_render = total if max_pages is None else min(total, max_pages)
        pages: list[dict[str, Any]] = []
        for idx in range(pages_to_render):
            png = render_page_to_png_bytes(pdf_path, page_index=idx, dpi=dpi)
            img_path = image_dir / f"page_{idx + 1:03d}.png"
            img_path.write_bytes(png)
            pages.append({"page_index": idx, "path": str(img_path)})
        return {"status": "ok", "page_count": total, "pages": pages}
    except Exception as exc:  # noqa: BLE001
        return {"status": "error", "error": str(exc)}


def _make_vlm_runner(base_url: str, model: str | None, api_key: str) -> Any:
    from gwanbo_ocr.runners.vllm import VllmChatRunner

    return VllmChatRunner(model=model or "default", base_url=base_url, api_key=api_key)


def _choose_best_method(peers: dict[str, dict[str, Any]]) -> str | None:
    candidates = [
        (name, int(peer.get("text_chars") or 0), METHOD_PREFERENCE.get(name, 0))
        for name, peer in peers.items()
        if peer.get("status") == "ok" and int(peer.get("text_chars") or 0) > 0
    ]
    if not candidates:
        return None
    candidates.sort(key=lambda t: (t[1], t[2]), reverse=True)
    return candidates[0][0]


def _pairwise_similarities(peers: dict[str, dict[str, Any]]) -> dict[str, float]:
    samples = {
        name: _norm_for_sim(str(peer.get("sample_text") or ""))
        for name, peer in peers.items()
        if peer.get("status") == "ok" and peer.get("sample_text")
    }
    result: dict[str, float] = {}
    names = sorted(samples)
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            ratio = difflib.SequenceMatcher(None, samples[left], samples[right]).ratio()
            result[f"{left}:{right}"] = round(ratio, 4)
    return result


def _compact_index(report: dict[str, Any], sidecar_path: Path) -> dict[str, Any]:
    review = report.get("review") or {}
    decision = report.get("decision") or {}
    score = report.get("score") or {}
    return {
        "status": report.get("status"),
        "sample_id": report.get("sample_id"),
        "pdf_key": report.get("pdf_key"),
        "analysis_scope": report.get("analysis_scope"),
        "best_text_method": review.get("best_text_method"),
        "decision": decision,
        "peer_summaries": review.get("peer_summaries"),
        "ranked_by_f1": score.get("ranked_by_f1"),
        "sidecar_path": str(sidecar_path),
        "generated_at": report.get("generated_at"),
        "error": report.get("error"),
    }


def _with_timeout(fn: Any, timeout_seconds: int, *, method: str) -> dict[str, Any]:
    previous = None
    if timeout_seconds > 0:
        previous = signal.signal(signal.SIGALRM, _alarm_handler)
        signal.alarm(timeout_seconds)
    try:
        result = fn()
        return (
            result
            if isinstance(result, dict)
            else _error("method returned non-dict", method=method)
        )
    except Exception as exc:  # noqa: BLE001
        return _error(str(exc), method=method)
    finally:
        if timeout_seconds > 0:
            signal.alarm(0)
            if previous is not None:
                signal.signal(signal.SIGALRM, previous)


def _alarm_handler(_signum: int, _frame: Any) -> None:
    raise TimeoutError("peer review extraction timed out")


def _skipped(reason: str, *, method: str) -> dict[str, Any]:
    return {
        "status": "skipped",
        "method": method,
        "skip_reason": reason,
        "text_extractable": False,
        "text_chars": 0,
        "sample_text": "",
        "error": None,
    }


def _error(message: str, *, method: str) -> dict[str, Any]:
    return {
        "status": "error",
        "method": method,
        "text_extractable": False,
        "text_chars": 0,
        "sample_text": "",
        "error": message,
    }


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _norm_for_sim(text: str) -> str:
    return re.sub(r"\W+", "", text).lower()[:2000]


def _scope(max_pages: int | None) -> str:
    return "all_pages" if max_pages is None else f"first_{max_pages}_pages"


def _iso_now() -> str:
    return datetime.now().isoformat()
