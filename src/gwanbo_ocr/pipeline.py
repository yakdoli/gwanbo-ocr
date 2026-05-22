"""Adaptive PDF OCR pipeline orchestration."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from gwanbo_ocr.bench.run import resolve_runner_config, resolve_runner_model
from gwanbo_ocr.pdf.font_analysis import (
    FontStats,
    ResolutionTier,
    classify_font_tier,
    extract_page_font_stats,
)
from gwanbo_ocr.pdf.io import (
    pdf_key_from_row,
    read_jsonl,
    resolve_pdf_path,
    write_json_atomic,
    write_jsonl_atomic,
)
from gwanbo_ocr.pdf.resolution_strategy import (
    RESOLUTION_TIERS,
    ResolutionPlan,
    build_resolution_plan,
    determine_resolution_tier_ocr_probe,
)
from gwanbo_ocr.prompts import TRANSCRIPTION_JSON_SCHEMA, TRANSCRIPTION_SYSTEM_PROMPT
from gwanbo_ocr.render import render_manifest_adaptive, render_pdf_page_result
from gwanbo_ocr.runners.vllm import VllmChatRunner

try:
    import pdfplumber  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - optional PDF dependency
    pdfplumber = None  # type: ignore[assignment]


RUNNERS_BY_TIER: dict[ResolutionTier, str] = {
    ResolutionTier.HIGH: "chandra_ocr_2_lemonade_vllm",
    ResolutionTier.STANDARD: "lightonocr2_1b_lemonade_vllm",
    ResolutionTier.LOW: "bizonai_ocr_lemonade_vllm",
}


@dataclass
class PageResult:
    page_index: int
    page_number: int
    tier: ResolutionTier
    resolution_method: str
    max_long_edge: int
    dpi: int
    model_used: str
    image_path: str
    text: str
    status: str
    error: str | None = None
    duration_s: float | None = None
    token_usage: dict | None = None


@dataclass
class PipelineResult:
    pdf_path: str
    pdf_key: str
    status: str
    pages: list[PageResult]
    resolution_plan: dict
    combined_text: str
    started_at: str
    completed_at: str
    total_duration_s: float
    summary: dict


class AdaptivePipeline:
    """Run adaptive per-page rendering and OCR for PDFs or PDF manifests."""

    def __init__(
        self,
        *,
        base_url: str = "http://127.0.0.1:13305/api/v1",
        api_key: str = "dummy",
        default_runner: str = "lightonocr2_1b_lemonade_vllm",
        probe_runner: str = "lightonocr2_1b_lemonade_vllm",
        output_dir: str | Path = "runs/adaptive",
        workers: int = 1,
    ) -> None:
        self.base_url = base_url
        self.api_key = api_key
        self.default_runner = default_runner
        self.probe_runner = probe_runner
        self.output_dir = Path(output_dir)
        self.workers = max(1, workers)
        self.runners_by_tier = dict(RUNNERS_BY_TIER)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def process_pdf(self, pdf_path: str | Path, **kwargs: Any) -> PipelineResult:
        """Process one PDF through classification, adaptive rendering, and OCR."""
        path = Path(pdf_path).expanduser()
        pdf_key = str(kwargs.get("pdf_key") or path.with_suffix("").name)
        row = dict(kwargs.get("manifest_row") or {})
        row.setdefault("pdf_key", pdf_key)
        row.setdefault("pdf_path", str(path))
        started_at = _now_iso()
        started = time.perf_counter()
        pdf_output_dir = self.output_dir / _safe_key(pdf_key)

        pages: list[PageResult] = []
        plan_dict: dict[str, Any] = {}
        combined_text = ""
        status = "error"

        try:
            font_stats, ocr_probes, analysis_errors, page_count = self._analyze_resolution_inputs(
                path, pdf_key, pdf_output_dir
            )
            plan = build_resolution_plan(
                path, font_stats_per_page=font_stats, ocr_probe_results=ocr_probes
            )
            plan_dict = _resolution_plan_to_dict(plan)
            rendered_by_page = self._render_pdf_adaptive(
                row, pdf_key, path, page_count, plan_dict, pdf_output_dir
            )
            pages = self._ocr_rendered_pages(plan, rendered_by_page, analysis_errors)
            combined_text = "\n\n".join(page.text for page in pages if page.text)
            status = _pipeline_status(pages, expected_pages=page_count)
        except Exception as exc:  # noqa: BLE001
            pages = [
                PageResult(
                    page_index=0,
                    page_number=1,
                    tier=ResolutionTier.STANDARD,
                    resolution_method="font_analysis",
                    max_long_edge=int(RESOLUTION_TIERS[ResolutionTier.STANDARD]["max_long_edge"]),
                    dpi=int(RESOLUTION_TIERS[ResolutionTier.STANDARD]["dpi"]),
                    model_used=resolve_runner_model(self.default_runner),
                    image_path="",
                    text="",
                    status="error",
                    error=str(exc),
                    duration_s=round(time.perf_counter() - started, 3),
                )
            ]
            status = "error"

        completed_at = _now_iso()
        result = PipelineResult(
            pdf_path=str(path),
            pdf_key=pdf_key,
            status=status,
            pages=pages,
            resolution_plan=plan_dict,
            combined_text=combined_text,
            started_at=started_at,
            completed_at=completed_at,
            total_duration_s=round(time.perf_counter() - started, 3),
            summary=_summarize_pages(pages),
        )
        write_json_atomic(pdf_output_dir / "result.json", pipeline_result_to_dict(result))
        return result

    def process_manifest(self, manifest_path: str | Path, **kwargs: Any) -> list[PipelineResult]:
        """Process every PDF row in a JSONL manifest sequentially."""
        manifest = Path(manifest_path)
        results: list[PipelineResult] = []
        limit = kwargs.get("limit")
        for index, row in enumerate(read_jsonl(manifest), start=1):
            if limit is not None and index > int(limit):
                break
            if not isinstance(row, dict):
                continue
            pdf_path = resolve_pdf_path(row, base_dir=manifest.parent, peti_root="/root/peti")
            pdf_key = pdf_key_from_row(row, fallback=pdf_path.stem or f"row-{index}")
            results.append(
                self.process_pdf(
                    pdf_path,
                    pdf_key=pdf_key,
                    manifest_row=row,
                )
            )
        return results

    def _analyze_resolution_inputs(
        self,
        pdf_path: Path,
        pdf_key: str,
        output_dir: Path,
    ) -> tuple[dict[int, FontStats], dict[int, dict[str, Any]], dict[int, str], int]:
        if pdfplumber is None:
            raise RuntimeError(
                "pdfplumber is required for adaptive PDF classification. "
                "Install it with `pip install pdfplumber`."
            )

        font_stats: dict[int, FontStats] = {}
        ocr_probes: dict[int, dict[str, Any]] = {}
        analysis_errors: dict[int, str] = {}
        with pdfplumber.open(str(pdf_path)) as pdf:
            page_count = len(pdf.pages)
            for page_index, page in enumerate(pdf.pages):
                page_chars = getattr(page, "chars", []) or []
                if page_chars:
                    stats = extract_page_font_stats(page)
                    font_stats[page_index] = stats
                    classify_font_tier(stats)
                    continue
                try:
                    ocr_probes[page_index] = self._ocr_probe_page(
                        pdf_path, pdf_key, page_index, output_dir
                    )
                except Exception as exc:  # noqa: BLE001
                    analysis_errors[page_index] = f"ocr_probe_failed:{exc}"
                    ocr_probes[page_index] = {
                        "ocr_text": "",
                        "image_dimensions": (0, 0),
                        "probe_runner": self.probe_runner,
                        "error": str(exc),
                    }
        return font_stats, ocr_probes, analysis_errors, page_count

    def _ocr_probe_page(
        self,
        pdf_path: Path,
        pdf_key: str,
        page_index: int,
        output_dir: Path,
    ) -> dict[str, Any]:
        probe_config = RESOLUTION_TIERS[ResolutionTier.LOW]
        rendered = render_pdf_page_result(
            pdf_path,
            page_index=page_index,
            dpi=int(probe_config["dpi"]),
        )
        image = _resize_to_max_edge(rendered.image, int(probe_config["max_long_edge"]))
        image_path = output_dir / "probe" / _safe_key(pdf_key) / f"page_{page_index + 1:04d}.png"
        image_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(image_path, format="PNG")
        result = self._transcribe_image(
            image_path,
            page_number=page_index + 1,
            runner_name=self.probe_runner,
        )
        tier = determine_resolution_tier_ocr_probe(
            result.text, (int(image.width), int(image.height))
        )
        return {
            "ocr_text": result.text,
            "image_dimensions": (int(image.width), int(image.height)),
            "probe_image_path": str(image_path),
            "probe_runner": self.probe_runner,
            "tier": tier.value,
        }

    def _render_pdf_adaptive(
        self,
        row: dict[str, Any],
        pdf_key: str,
        pdf_path: Path,
        page_count: int,
        resolution_plan: dict[str, Any],
        output_dir: Path,
    ) -> dict[int, dict[str, Any]]:
        manifest_path = output_dir / "input_manifest.jsonl"
        manifest_row = dict(row)
        manifest_row.update(
            {
                "pdf_key": pdf_key,
                "pdf_path": str(pdf_path),
                "pdf_abs_path": str(pdf_path),
                "pages": page_count,
                "selected_pages": list(range(1, page_count + 1)),
            }
        )
        write_jsonl_atomic(manifest_path, [manifest_row])
        render_summary = render_manifest_adaptive(
            manifest_path=manifest_path,
            output_dir=output_dir / "rendered",
            resolution_plan=resolution_plan,
            default_dpi=int(RESOLUTION_TIERS[ResolutionTier.STANDARD]["dpi"]),
            default_max_long_edge=int(RESOLUTION_TIERS[ResolutionTier.STANDARD]["max_long_edge"]),
            max_pages=None,
            workers=1,
        )
        manifest_out = Path(str(render_summary["manifest"]))
        rendered_rows = [row for row in read_jsonl(manifest_out) if isinstance(row, dict)]
        return {int(row.get("page_number") or 0): row for row in rendered_rows}

    def _ocr_rendered_pages(
        self,
        resolution_plan: ResolutionPlan,
        rendered_by_page: dict[int, dict[str, Any]],
        analysis_errors: dict[int, str],
    ) -> list[PageResult]:
        page_results: list[PageResult] = []
        for page_resolution in resolution_plan.pages:
            page_number = page_resolution.page_index + 1
            rendered_row = rendered_by_page.get(page_number, {})
            image_path = str(rendered_row.get("image_path") or "")
            runner_name = self.runners_by_tier.get(
                page_resolution.tier,
                page_resolution.recommended_model or self.default_runner,
            )
            started = time.perf_counter()
            try:
                if page_resolution.page_index in analysis_errors:
                    raise RuntimeError(analysis_errors[page_resolution.page_index])
                if rendered_row.get("status") != "ok":
                    raise RuntimeError(str(rendered_row.get("error") or "adaptive render failed"))
                result = self._transcribe_image(
                    image_path,
                    page_number=page_number,
                    runner_name=runner_name,
                )
                metadata = _extract_response_metadata(result.raw_response)
                page_results.append(
                    PageResult(
                        page_index=page_resolution.page_index,
                        page_number=page_number,
                        tier=page_resolution.tier,
                        resolution_method=page_resolution.method,
                        max_long_edge=page_resolution.max_long_edge,
                        dpi=page_resolution.dpi,
                        model_used=resolve_runner_model(runner_name),
                        image_path=image_path,
                        text=result.text,
                        status="ok",
                        duration_s=round(time.perf_counter() - started, 3),
                        token_usage=metadata.get("usage"),
                    )
                )
            except Exception as exc:  # noqa: BLE001
                page_results.append(
                    PageResult(
                        page_index=page_resolution.page_index,
                        page_number=page_number,
                        tier=page_resolution.tier,
                        resolution_method=page_resolution.method,
                        max_long_edge=page_resolution.max_long_edge,
                        dpi=page_resolution.dpi,
                        model_used=resolve_runner_model(runner_name),
                        image_path=image_path,
                        text="",
                        status="error",
                        error=str(exc),
                        duration_s=round(time.perf_counter() - started, 3),
                    )
                )
        return page_results

    def _transcribe_image(
        self,
        image_path: str | Path,
        *,
        page_number: int,
        runner_name: str,
    ) -> Any:
        model_id = resolve_runner_model(runner_name)
        runner_config = resolve_runner_config(runner_name)
        json_mode = bool(runner_config.get("json_mode", True))
        system_prompt = runner_config.get("system_prompt", "default")
        if system_prompt == "default":
            system_prompt = TRANSCRIPTION_SYSTEM_PROMPT
        elif system_prompt is not None:
            system_prompt = str(system_prompt)
        user_prompt = runner_config.get("user_prompt")
        runner = VllmChatRunner(
            model=model_id,
            base_url=self.base_url,
            api_key=self.api_key,
            timeout=float(runner_config.get("timeout_seconds") or 120),
            temperature=float(runner_config.get("temperature") or 0.0),
            top_p=runner_config.get("top_p"),
            max_tokens=int(runner_config.get("max_tokens") or 4096),
            response_format={"type": "json_object"} if json_mode else None,
            strict_json=False,
        )
        return runner.transcribe(
            image_path,
            page_number=page_number,
            language_hint="ko,en",
            schema=TRANSCRIPTION_JSON_SCHEMA if json_mode else None,
            user_prompt=str(user_prompt) if user_prompt is not None else None,
            system_prompt=system_prompt,
        )


def run_adaptive_pipeline(
    manifest_path: str | Path,
    output_dir: str | Path,
    base_url: str = "http://127.0.0.1:13305/api/v1",
    **kwargs: Any,
) -> dict[str, Any]:
    pipeline = AdaptivePipeline(base_url=base_url, output_dir=output_dir, **kwargs)
    results = pipeline.process_manifest(manifest_path)
    summary = {
        "status": _batch_status(results),
        "manifest_path": str(manifest_path),
        "output_dir": str(output_dir),
        "total_pdfs": len(results),
        "ok_pdfs": sum(1 for result in results if result.status == "ok"),
        "partial_pdfs": sum(1 for result in results if result.status == "partial"),
        "error_pdfs": sum(1 for result in results if result.status == "error"),
        "total_pages": sum(len(result.pages) for result in results),
        "results": [pipeline_result_to_dict(result) for result in results],
    }
    write_json_atomic(Path(output_dir) / "summary.json", summary)
    return summary


def pipeline_result_to_dict(result: PipelineResult) -> dict[str, Any]:
    payload = asdict(result)
    for page in payload["pages"]:
        tier = page.get("tier")
        if isinstance(tier, ResolutionTier):
            page["tier"] = tier.value
    return payload


def _resolution_plan_to_dict(plan: ResolutionPlan) -> dict[str, Any]:
    return {
        "pdf_path": plan.pdf_path,
        "pages": [
            {
                "page_index": page.page_index,
                "tier": page.tier.value,
                "method": page.method,
                "max_long_edge": page.max_long_edge,
                "dpi": page.dpi,
                "recommended_model": page.recommended_model,
                "confidence": page.confidence,
                "reason": page.reason,
            }
            for page in plan.pages
        ],
        "default_tier": plan.default_tier.value,
        "generated_at": plan.generated_at,
    }


def _summarize_pages(pages: list[PageResult]) -> dict[str, Any]:
    by_tier: dict[str, int] = {}
    by_model: dict[str, int] = {}
    for page in pages:
        by_tier[page.tier.value] = by_tier.get(page.tier.value, 0) + 1
        by_model[page.model_used] = by_model.get(page.model_used, 0) + 1
    ok_pages = sum(1 for page in pages if page.status == "ok")
    error_pages = sum(1 for page in pages if page.status == "error")
    return {
        "pages": len(pages),
        "ok_pages": ok_pages,
        "error_pages": error_pages,
        "by_tier": by_tier,
        "by_model": by_model,
        "characters": sum(len(page.text) for page in pages),
    }


def _pipeline_status(pages: list[PageResult], *, expected_pages: int) -> str:
    if not pages or all(page.status == "error" for page in pages):
        return "error"
    if len(pages) != expected_pages or any(page.status == "error" for page in pages):
        return "partial"
    return "ok"


def _batch_status(results: list[PipelineResult]) -> str:
    if not results:
        return "ok"
    if all(result.status == "ok" for result in results):
        return "ok"
    if all(result.status == "error" for result in results):
        return "error"
    return "partial"


def _extract_response_metadata(response: Any) -> dict[str, Any]:
    payload = _response_to_mapping(response)
    usage = payload.get("usage")
    if isinstance(usage, dict):
        return {"usage": dict(usage)}
    model_dump = getattr(usage, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump()
        if isinstance(dumped, dict):
            return {"usage": dumped}
    return {}


def _response_to_mapping(response: Any) -> dict[str, Any]:
    if isinstance(response, dict):
        return response
    if hasattr(response, "model_dump"):
        dumped = response.model_dump()
        if isinstance(dumped, dict):
            return dumped
    return {
        field: getattr(response, field)
        for field in ("id", "model", "object", "choices", "usage")
        if getattr(response, field, None) is not None
    }


def _resize_to_max_edge(image: Any, max_long_edge: int) -> Any:
    if max_long_edge <= 0:
        return image
    longest = max(image.width, image.height)
    if longest <= max_long_edge:
        return image
    scale = max_long_edge / longest
    size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
    return image.resize(size)


def _safe_key(value: str) -> str:
    return value.replace("/", "_").replace("\\", "_").strip("._") or "unknown"


def _now_iso() -> str:
    return datetime.now().isoformat()


__all__ = [
    "AdaptivePipeline",
    "PageResult",
    "PipelineResult",
    "pipeline_result_to_dict",
    "run_adaptive_pipeline",
]
