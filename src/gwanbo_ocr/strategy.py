"""Deterministic layout clustering and parsing strategy assignment."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from gwanbo_ocr.pdf.io import read_jsonl, write_json_atomic, write_jsonl_atomic

CLUSTER_SCHEMA_VERSION = "layout-cluster/v1"
EVAL_SCHEMA_VERSION = "strategy-eval/v1"

STRATEGIES = {
    "native_text_body",
    "native_pdfplumber_table",
    "ocr_paddle_simple",
    "ocr_vlm_structured",
    "peer_review_escalation",
    "skip_invalid",
}

PEER_METHOD_TO_STRATEGY = {
    "native_text": "native_text_body",
    "markitdown": "native_text_body",
    "pdfplumber": "native_pdfplumber_table",
    "paddle_ocr": "ocr_paddle_simple",
    "vlm_ocr": "ocr_vlm_structured",
}


def cluster_profiles(
    *,
    profiles_path: str | Path,
    output_dir: str | Path,
    sample_keys: int = 10,
) -> dict[str, Any]:
    """Cluster profile rows by deterministic, explainable feature buckets."""
    rows = [row for row in read_jsonl(profiles_path) if isinstance(row, dict)]
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[_feature_key(row)].append(row)

    clusters = [
        _cluster_from_group(key, group, sample_keys=sample_keys)
        for key, group in sorted(groups.items())
    ]
    clusters.sort(key=lambda row: str(row["cluster_id"]))

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    manifest = output / "cluster_manifest.jsonl"
    write_jsonl_atomic(manifest, clusters)
    summary = {
        "status": "ok",
        "schema_version": CLUSTER_SCHEMA_VERSION,
        "profiles": str(profiles_path),
        "output_dir": str(output),
        "manifest": str(manifest),
        "profiles_count": len(rows),
        "clusters": len(clusters),
        "by_strategy": dict(sorted(Counter(row["assigned_strategy"] for row in clusters).items())),
    }
    write_json_atomic(output / "summary.json", summary)
    return summary


def evaluate_clusters(
    *,
    clusters_path: str | Path,
    output_dir: str | Path,
    bench_scores_path: str | Path | None = None,
    peer_score_report_path: str | Path | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    """Create v1 strategy-eval rows using available cluster proxy metrics."""
    clusters = [row for row in read_jsonl(clusters_path) if isinstance(row, dict)]
    if limit is not None:
        clusters = clusters[:limit]

    bench_metrics = _load_bench_metrics_by_strategy(bench_scores_path)
    peer_metrics = _load_peer_metrics_by_strategy(peer_score_report_path)
    evaluations = [
        _evaluate_cluster(
            row,
            bench_metrics_by_strategy=bench_metrics,
            peer_metrics_by_strategy=peer_metrics,
        )
        for row in clusters
    ]
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    scores = output / "strategy_eval.jsonl"
    write_jsonl_atomic(scores, evaluations)
    summary = {
        "status": "ok",
        "schema_version": EVAL_SCHEMA_VERSION,
        "clusters": str(clusters_path),
        "output_dir": str(output),
        "scores": str(scores),
        "bench_scores": str(bench_scores_path) if bench_scores_path else None,
        "peer_score_report": str(peer_score_report_path) if peer_score_report_path else None,
        "evaluated_clusters": len(evaluations),
        "by_recommendation": dict(
            sorted(Counter(row["recommendation"] for row in evaluations).items())
        ),
    }
    write_json_atomic(output / "summary.json", summary)
    return summary


def assign_strategy(feature_signature: Mapping[str, Any]) -> tuple[str, float, list[str]]:
    """Assign a parsing strategy from a cluster feature signature."""
    text_mode = str(feature_signature.get("text_mode") or "unknown")
    layout_class = str(feature_signature.get("layout_class") or "unknown_text")
    page_bucket = str(feature_signature.get("page_count_bucket") or "unknown")
    table_bucket = str(feature_signature.get("table_density_bucket") or "none")
    form_bucket = str(feature_signature.get("form_score_bucket") or "none")

    if text_mode in {"missing", "invalid"}:
        return "skip_invalid", 0.95, [f"text_mode:{text_mode}"]
    if text_mode == "text" and (
        layout_class in {"table_heavy", "table_with_body"} or table_bucket in {"medium", "high"}
    ):
        return "native_pdfplumber_table", 0.86, ["text layer exists", "table-like layout"]
    if text_mode == "text":
        return "native_text_body", 0.82, ["text layer exists", f"layout:{layout_class}"]
    if text_mode == "image" and page_bucket in {"1", "2-3"} and form_bucket in {"none", "low"}:
        return "ocr_paddle_simple", 0.72, ["image PDF", "short/simple document"]
    if text_mode == "image":
        return "ocr_vlm_structured", 0.76, ["image PDF", "structured or longer document"]
    return "peer_review_escalation", 0.55, [f"ambiguous text_mode:{text_mode}"]


def feature_signature(row: Mapping[str, Any]) -> dict[str, str]:
    """Return the stable clustering signature for a profile row."""
    return {
        "theme": _string(row.get("theme")) or "unknown",
        "year": _string(row.get("year")) or "unknown",
        "category": _string(row.get("category")) or "uncategorized",
        "text_mode": _text_mode(row),
        "layout_class": _string(row.get("layout_class")) or "unknown_text",
        "page_count_bucket": page_count_bucket(row.get("pages")),
        "table_density_bucket": table_density_bucket(
            row.get("table_text_ratio"),
            row.get("table_count"),
        ),
        "form_score_bucket": form_score_bucket(row.get("form_score")),
    }


def page_count_bucket(value: Any) -> str:
    pages = _optional_int(value)
    if pages is None or pages <= 0:
        return "unknown"
    if pages == 1:
        return "1"
    if pages <= 3:
        return "2-3"
    if pages <= 10:
        return "4-10"
    if pages <= 50:
        return "11-50"
    return "51+"


def table_density_bucket(ratio_value: Any, count_value: Any = None) -> str:
    ratio = _optional_float(ratio_value) or 0.0
    count = _optional_int(count_value) or 0
    if count <= 0 and ratio <= 0:
        return "none"
    if ratio < 0.2:
        return "low"
    if ratio < 0.5:
        return "medium"
    return "high"


def form_score_bucket(value: Any) -> str:
    score = _optional_float(value) or 0.0
    if score <= 0:
        return "none"
    if score < 0.2:
        return "low"
    if score < 0.45:
        return "medium"
    return "high"


def _cluster_from_group(
    key: tuple[str, ...], group: list[dict[str, Any]], *, sample_keys: int
) -> dict[str, Any]:
    signature = dict(zip(_signature_fields(), key, strict=True))
    strategy, confidence, reasons = assign_strategy(signature)
    sorted_group = sorted(group, key=lambda row: str(row.get("pdf_key") or ""))
    profile_summary = _profile_summary(group)
    return {
        "schema_version": CLUSTER_SCHEMA_VERSION,
        "cluster_id": _cluster_id(signature),
        "year": signature["year"],
        "theme": signature["theme"],
        "dominant_category": _dominant_value(group, "category"),
        "feature_signature": signature,
        "count": len(group),
        "sample_pdf_keys": [
            str(row.get("pdf_key") or "")
            for row in sorted_group[:sample_keys]
            if row.get("pdf_key")
        ],
        "assigned_strategy": strategy,
        "confidence": confidence,
        "reasons": reasons,
        "profile_summary": profile_summary,
    }


def _evaluate_cluster(
    cluster: Mapping[str, Any],
    *,
    bench_metrics_by_strategy: Mapping[str, Mapping[str, float]],
    peer_metrics_by_strategy: Mapping[str, Mapping[str, float]],
) -> dict[str, Any]:
    count = _optional_int(cluster.get("count")) or 0
    summary_payload = cluster.get("profile_summary")
    profile_summary = summary_payload if isinstance(summary_payload, Mapping) else {}
    errors = _optional_int(profile_summary.get("error_count")) or 0
    strategy = str(cluster.get("assigned_strategy") or "peer_review_escalation")
    error_rate = errors / count if count else 0.0
    completion_rate = 0.0 if strategy == "skip_invalid" else max(0.0, 1.0 - error_rate)
    bench = bench_metrics_by_strategy.get(strategy, {})
    peer = peer_metrics_by_strategy.get(strategy, {})
    critical_token_f1 = _first_float(bench.get("critical_token_f1"), peer.get("critical_token_f1"))
    cer = _optional_float(bench.get("cer"))
    wer = _optional_float(bench.get("wer"))
    table_f1 = _optional_float(bench.get("table_f1"))
    confidence = float(cluster.get("confidence") or 0.0)
    recommendation = _recommendation(
        strategy,
        error_rate,
        confidence,
        critical_token_f1=critical_token_f1,
        cer=cer,
    )
    return {
        "schema_version": EVAL_SCHEMA_VERSION,
        "cluster_id": cluster.get("cluster_id"),
        "strategy": strategy,
        "sample_count": count,
        "completion_rate": round(completion_rate, 4),
        "error_rate": round(error_rate, 4),
        "critical_token_f1": _round_optional(critical_token_f1),
        "cer": _round_optional(cer),
        "wer": _round_optional(wer),
        "table_f1": _round_optional(table_f1),
        "peer_agreement": round(confidence, 4),
        "recommendation": recommendation,
    }


def _recommendation(
    strategy: str,
    error_rate: float,
    confidence: float,
    *,
    critical_token_f1: float | None,
    cer: float | None,
) -> str:
    if strategy == "skip_invalid":
        return "skip_or_repair_sources"
    if critical_token_f1 is not None and critical_token_f1 < 0.9:
        return "run_peer_review_before_scaling"
    if cer is not None and cer > 0.12:
        return "run_peer_review_before_scaling"
    if error_rate > 0.1 or confidence < 0.65:
        return "run_peer_review_before_scaling"
    return "ready_for_representative_sample"


def build_strategy_benchmark_suite(
    *,
    render_manifest_path: str | Path,
    clusters_path: str | Path,
    output_path: str | Path,
    include_strategies: set[str] | None = None,
) -> dict[str, Any]:
    """Attach cluster-assigned strategy metadata to rendered page tasks."""
    rows = [row for row in read_jsonl(render_manifest_path) if isinstance(row, dict)]
    clusters = [row for row in read_jsonl(clusters_path) if isinstance(row, dict)]
    strategy_by_pdf_key: dict[str, dict[str, Any]] = {}
    for cluster in clusters:
        strategy = str(cluster.get("assigned_strategy") or "peer_review_escalation")
        payload = {
            "strategy": strategy,
            "cluster_id": cluster.get("cluster_id"),
            "strategy_confidence": cluster.get("confidence"),
        }
        for pdf_key in cluster.get("sample_pdf_keys") or []:
            key = str(pdf_key or "").strip()
            if key:
                strategy_by_pdf_key[key] = payload

    suite_rows: list[dict[str, Any]] = []
    for row in rows:
        pdf_key = str(row.get("pdf_key") or "").strip()
        strategy_payload = strategy_by_pdf_key.get(pdf_key)
        strategy = str(
            (strategy_payload or {}).get("strategy")
            or row.get("strategy")
            or "peer_review_escalation"
        )
        if include_strategies and strategy not in include_strategies:
            continue
        suite_row = dict(row)
        suite_row["strategy"] = strategy
        if strategy_payload:
            suite_row["cluster_id"] = strategy_payload.get("cluster_id")
            suite_row["strategy_confidence"] = strategy_payload.get("strategy_confidence")
        suite_rows.append(suite_row)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl_atomic(output, suite_rows)
    by_strategy = Counter(str(row.get("strategy") or "") for row in suite_rows)
    return {
        "status": "ok",
        "render_manifest": str(render_manifest_path),
        "clusters": str(clusters_path),
        "output": str(output),
        "tasks": len(suite_rows),
        "by_strategy": dict(sorted((k, v) for k, v in by_strategy.items() if k)),
        "unmapped_rows": sum(1 for row in suite_rows if row.get("cluster_id") is None),
    }


def _load_bench_metrics_by_strategy(path: str | Path | None) -> dict[str, dict[str, float]]:
    if path is None:
        return {}
    try:
        rows = [row for row in read_jsonl(path) if isinstance(row, dict)]
    except OSError:
        return {}
    aggregates: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        strategy = str(row.get("strategy") or "").strip()
        if not strategy:
            continue
        metrics_payload = row.get("metrics")
        if not isinstance(metrics_payload, Mapping):
            continue
        for key in ("critical_token_f1", "cer", "wer", "table_f1"):
            value = _optional_float(metrics_payload.get(key))
            if value is not None:
                aggregates[strategy][key].append(value)
    return {
        strategy: {
            key: (sum(values) / len(values))
            for key, values in metrics.items()
            if values
        }
        for strategy, metrics in aggregates.items()
    }


def _load_peer_metrics_by_strategy(path: str | Path | None) -> dict[str, dict[str, float]]:
    if path is None:
        return {}
    payload: dict[str, Any] = {}
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    by_method = payload.get("avg_critical_token_f1_by_method")
    if not isinstance(by_method, Mapping):
        return {}
    per_strategy: dict[str, list[float]] = defaultdict(list)
    for method, value in by_method.items():
        strategy = PEER_METHOD_TO_STRATEGY.get(str(method))
        score = _optional_float(value)
        if strategy and score is not None:
            per_strategy[strategy].append(score)
    return {
        strategy: {"critical_token_f1": sum(scores) / len(scores)}
        for strategy, scores in per_strategy.items()
        if scores
    }


def _first_float(*values: Any) -> float | None:
    for value in values:
        parsed = _optional_float(value)
        if parsed is not None:
            return parsed
    return None


def _round_optional(value: float | None, digits: int = 4) -> float | None:
    return round(value, digits) if value is not None else None


def _feature_key(row: Mapping[str, Any]) -> tuple[str, ...]:
    signature = feature_signature(row)
    return tuple(signature[field] for field in _signature_fields())


def _signature_fields() -> tuple[str, ...]:
    return (
        "theme",
        "year",
        "category",
        "text_mode",
        "layout_class",
        "page_count_bucket",
        "table_density_bucket",
        "form_score_bucket",
    )


def _cluster_id(signature: Mapping[str, Any]) -> str:
    payload = json.dumps(signature, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _profile_summary(group: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    rows = list(group)
    total = len(rows)
    error_count = sum(1 for row in rows if row.get("error"))
    text_count = sum(1 for row in rows if bool(row.get("text_extractable")))
    table_counts = [_optional_int(row.get("table_count")) or 0 for row in rows]
    return {
        "total": total,
        "error_count": error_count,
        "text_extractable_count": text_count,
        "avg_table_count": round(sum(table_counts) / total, 4) if total else 0.0,
        "by_layout_class": dict(
            sorted(
                Counter(_string(row.get("layout_class")) or "unknown_text" for row in rows).items()
            )
        ),
    }


def _dominant_value(rows: Iterable[Mapping[str, Any]], field: str) -> str:
    counts = Counter(_string(row.get(field)) or "uncategorized" for row in rows)
    return counts.most_common(1)[0][0] if counts else "uncategorized"


def _text_mode(row: Mapping[str, Any]) -> str:
    explicit = _string(row.get("text_mode"))
    if explicit:
        return explicit
    integrity = _string(row.get("integrity_status"))
    if integrity in {"missing", "fail"}:
        return "invalid" if integrity == "fail" else "missing"
    if row.get("text_extractable") is True:
        return "text"
    if row.get("text_extractable") is False:
        return "image"
    return "unknown"


def _string(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _optional_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
