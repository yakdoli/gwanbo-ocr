from __future__ import annotations

import concurrent.futures
import difflib
from pathlib import Path
from typing import Any

from ._helpers import SAMPLE_CHARS_DEFAULT, _iso_now, _make_vlm_runner, _norm_for_sim, _scope
from .markitdown import extract_markitdown, extract_markitdown_ocr_llm
from .native import extract_native_text
from .paddle import extract_paddle_ocr, extract_paddle_ocr_vl
from .pdfplumber import extract_pdfplumber
from .vlm import extract_vlm_ocr

PADDLE_OCR_VL_MODEL = "PaddlePaddle/PaddleOCR-VL-1.5"

METHOD_PREFERENCE = {
    "vlm_ocr": 5,
    "paddle_ocr_vl": 5,
    "markitdown_ocr_llm": 4,
    "paddle_ocr": 4,
    "pdfplumber": 3,
    "markitdown": 2,
    "native_text": 1,
}

__all__ = [
    "SAMPLE_CHARS_DEFAULT",
    "METHOD_PREFERENCE",
    "PADDLE_OCR_VL_MODEL",
    "extract_markitdown",
    "extract_markitdown_ocr_llm",
    "extract_native_text",
    "extract_paddle_ocr",
    "extract_paddle_ocr_vl",
    "extract_pdfplumber",
    "extract_vlm_ocr",
    "analyze_pdf_peer_review",
    "review_extraction_peers",
    "score_against_metadata",
    "decide_extraction",
    "run_peer_review_manifest",
    "aggregate_peer_scores",
]


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
    run_paddle_vl: bool = False,
    run_markitdown: bool = True,
    run_markitdown_ocr_llm: bool = False,
    run_pdfplumber: bool = True,
    run_native_text: bool = True,
    ocr_lang: str = "korean",
    paddle_service_url: str | None = None,
    paddle_vl_service_url: str | None = None,
    paddle_vl_backend: str | None = None,
    paddle_vl_server_url: str | None = None,
    paddle_vl_model: str | None = None,
    paddle_vl_api_key: str | None = None,
    markitdown_service_url: str | None = None,
    markitdown_llm_base_url: str | None = None,
    markitdown_llm_model: str | None = None,
    markitdown_llm_api_key: str = "dummy",
    markitdown_llm_prompt: str | None = None,
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
    if run_markitdown_ocr_llm:
        peers["markitdown_ocr_llm"] = extract_markitdown_ocr_llm(
            pdf_path,
            sample_chars=sample_chars,
            timeout_seconds=timeout_seconds * 2,
            service_url=markitdown_service_url,
            llm_base_url=markitdown_llm_base_url,
            llm_model=markitdown_llm_model,
            llm_api_key=markitdown_llm_api_key,
            llm_prompt=markitdown_llm_prompt,
        )
    if run_paddle:
        peers["paddle_ocr"] = extract_paddle_ocr(
            pdf_path,
            image_dir=image_dir / "paddle",
            max_pages=max_pages,
            sample_chars=sample_chars,
            dpi=dpi,
            lang=ocr_lang,
            service_url=paddle_service_url,
            timeout_seconds=timeout_seconds * 2,
        )
    if run_paddle_vl:
        effective_paddle_vl_model = paddle_vl_model or (
            PADDLE_OCR_VL_MODEL if paddle_vl_server_url else None
        )
        peers["paddle_ocr_vl"] = extract_paddle_ocr_vl(
            pdf_path,
            image_dir=image_dir / "paddle_vl",
            max_pages=max_pages,
            sample_chars=sample_chars,
            dpi=dpi,
            vl_rec_backend=paddle_vl_backend or ("vllm-server" if paddle_vl_server_url else None),
            vl_rec_server_url=paddle_vl_server_url,
            vl_rec_api_model_name=effective_paddle_vl_model,
            vl_rec_api_key=paddle_vl_api_key,
            service_url=paddle_vl_service_url,
            timeout_seconds=timeout_seconds * 3,
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
            if name == best or not _peer_has_usable_text(peer):
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
        if not _peer_has_usable_text(peer):
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
    ocr_methods = ("markitdown_ocr_llm", "paddle_ocr", "paddle_ocr_vl", "vlm_ocr")

    text_layer_ok = any(_peer_has_usable_text(peers.get(m, {})) for m in text_layer)
    ocr_ok = any(_peer_has_usable_text(peers.get(m, {})) for m in ocr_methods)

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


def run_peer_review_manifest(
    manifest_path: Path,
    output_dir: Path,
    *,
    vlm_base_url: str | None = None,
    vlm_model: str | None = None,
    vlm_api_key: str = "dummy",
    run_paddle: bool = False,
    run_paddle_vl: bool = False,
    run_markitdown: bool = True,
    run_markitdown_ocr_llm: bool = False,
    run_pdfplumber: bool = True,
    run_native_text: bool = True,
    paddle_service_url: str | None = None,
    paddle_vl_service_url: str | None = None,
    paddle_vl_backend: str | None = None,
    paddle_vl_server_url: str | None = None,
    paddle_vl_model: str | None = None,
    paddle_vl_api_key: str | None = None,
    markitdown_service_url: str | None = None,
    markitdown_llm_base_url: str | None = None,
    markitdown_llm_model: str | None = None,
    markitdown_llm_api_key: str = "dummy",
    markitdown_llm_prompt: str | None = None,
    sample_artifacts_dir: Path | None = None,
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
    effective_paddle_vl_model = paddle_vl_model or (
        PADDLE_OCR_VL_MODEL if paddle_vl_server_url else None
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    items_dir = output_dir / "items"
    images_dir = output_dir / "images"
    samples_dir = sample_artifacts_dir or (output_dir / "samples")

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
            "run_markitdown_ocr_llm": run_markitdown_ocr_llm,
            "run_paddle": run_paddle,
            "run_paddle_vl": run_paddle_vl,
            "paddle_service_url": paddle_service_url,
            "paddle_vl_service_url": paddle_vl_service_url,
            "paddle_vl_backend": paddle_vl_backend,
            "paddle_vl_server_url": paddle_vl_server_url,
            "paddle_vl_model": effective_paddle_vl_model,
            "markitdown_service_url": markitdown_service_url,
            "markitdown_llm_base_url": markitdown_llm_base_url,
            "markitdown_llm_model": markitdown_llm_model,
            "sample_artifacts_dir": str(samples_dir),
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
                "run_paddle_vl": run_paddle_vl,
                "run_markitdown": run_markitdown,
                "run_markitdown_ocr_llm": run_markitdown_ocr_llm,
                "run_pdfplumber": run_pdfplumber,
                "run_native_text": run_native_text,
                "paddle_service_url": paddle_service_url,
                "paddle_vl_service_url": paddle_vl_service_url,
                "paddle_vl_backend": paddle_vl_backend,
                "paddle_vl_server_url": paddle_vl_server_url,
                "paddle_vl_model": effective_paddle_vl_model,
                "paddle_vl_api_key": paddle_vl_api_key,
                "markitdown_service_url": markitdown_service_url,
                "markitdown_llm_base_url": markitdown_llm_base_url,
                "markitdown_llm_model": markitdown_llm_model,
                "markitdown_llm_api_key": markitdown_llm_api_key,
                "markitdown_llm_prompt": markitdown_llm_prompt,
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
        artifact_dir = _write_sample_artifact(
            samples_dir / _safe_artifact_name(sample_id),
            row=result["row"],
            report=report,
            image_dir=Path(result["image_dir"]),
        )
        report["sample_artifact_dir"] = str(artifact_dir)
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
        run_paddle_vl=bool(work_item.get("run_paddle_vl")),
        run_markitdown=bool(work_item.get("run_markitdown", True)),
        run_markitdown_ocr_llm=bool(work_item.get("run_markitdown_ocr_llm")),
        run_pdfplumber=bool(work_item.get("run_pdfplumber", True)),
        run_native_text=bool(work_item.get("run_native_text", True)),
        paddle_service_url=work_item.get("paddle_service_url"),
        paddle_vl_service_url=work_item.get("paddle_vl_service_url"),
        paddle_vl_backend=work_item.get("paddle_vl_backend"),
        paddle_vl_server_url=work_item.get("paddle_vl_server_url"),
        paddle_vl_model=work_item.get("paddle_vl_model"),
        paddle_vl_api_key=work_item.get("paddle_vl_api_key"),
        markitdown_service_url=work_item.get("markitdown_service_url"),
        markitdown_llm_base_url=work_item.get("markitdown_llm_base_url"),
        markitdown_llm_model=work_item.get("markitdown_llm_model"),
        markitdown_llm_api_key=str(work_item.get("markitdown_llm_api_key") or "dummy"),
        markitdown_llm_prompt=work_item.get("markitdown_llm_prompt"),
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
        "row": row,
        "image_dir": work_item["image_dir"],
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


def _choose_best_method(peers: dict[str, dict[str, Any]]) -> str | None:
    candidates = [
        (name, int(peer.get("text_chars") or 0), METHOD_PREFERENCE.get(name, 0))
        for name, peer in peers.items()
        if _peer_has_usable_text(peer)
    ]
    if not candidates:
        return None
    candidates.sort(key=lambda t: (t[1], t[2]), reverse=True)
    return candidates[0][0]


def _pairwise_similarities(peers: dict[str, dict[str, Any]]) -> dict[str, float]:
    samples = {
        name: _norm_for_sim(str(peer.get("sample_text") or ""))
        for name, peer in peers.items()
        if _peer_has_usable_text(peer) and peer.get("sample_text")
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
        "sample_artifact_dir": report.get("sample_artifact_dir"),
        "generated_at": report.get("generated_at"),
        "error": report.get("error"),
    }


def _write_sample_artifact(
    artifact_dir: Path,
    *,
    row: dict[str, Any],
    report: dict[str, Any],
    image_dir: Path,
) -> Path:
    from gwanbo_ocr.pdf.io import write_json_atomic

    artifact_dir.mkdir(parents=True, exist_ok=True)
    source = {
        "sample_id": report.get("sample_id"),
        "pdf_key": report.get("pdf_key"),
        "source": report.get("source"),
        "path": report.get("path"),
        "title": row.get("title"),
        "category": row.get("category"),
        "agency": row.get("agency"),
        "row": row,
        "rendered_images": [str(path) for path in sorted(image_dir.rglob("*.png"))]
        if image_dir.exists()
        else [],
    }
    peer_samples = {
        name: {
            "status": peer.get("status"),
            "method": peer.get("method"),
            "text_chars": peer.get("text_chars"),
            "sample_text": peer.get("sample_text"),
            "error": peer.get("error"),
            "skip_reason": peer.get("skip_reason"),
        }
        for name, peer in (report.get("peers") or {}).items()
        if isinstance(peer, dict)
    }
    diff_summary = {
        "best_text_method": (report.get("review") or {}).get("best_text_method"),
        "pairwise_sample_similarity": (report.get("review") or {}).get(
            "pairwise_sample_similarity"
        ),
        "warnings": (report.get("review") or {}).get("warnings"),
        "decision": report.get("decision"),
    }
    write_json_atomic(artifact_dir / "source.json", source)
    write_json_atomic(artifact_dir / "peer_samples.json", peer_samples)
    write_json_atomic(artifact_dir / "diff_summary.json", diff_summary)
    (artifact_dir / "peer_samples.md").write_text(
        _peer_samples_markdown(peer_samples),
        encoding="utf-8",
    )
    return artifact_dir


def _peer_samples_markdown(peer_samples: dict[str, dict[str, Any]]) -> str:
    parts = ["# Peer Samples", ""]
    for name, peer in sorted(peer_samples.items()):
        parts.extend(
            [
                f"## {name}",
                "",
                f"- status: {peer.get('status')}",
                f"- method: {peer.get('method')}",
                f"- text_chars: {peer.get('text_chars')}",
                "",
                "```text",
                str(peer.get("sample_text") or ""),
                "```",
                "",
            ]
        )
    return "\n".join(parts)


def _safe_artifact_name(value: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value).strip("._-")
    return safe or "unknown"


def _peer_has_usable_text(peer: dict[str, Any]) -> bool:
    return peer.get("status") in {"ok", "partial"} and int(peer.get("text_chars") or 0) > 0
