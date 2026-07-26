"""High-level PDF classification helpers."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

from .content import ContentMode, inspect_pdf_content
from .integrity import validate_pdf_integrity
from .limits import DEFAULT_MAX_FILE_BYTES, PdfResourceLimitError, PdfTimeoutError
from .text import analyze_pdf_text
from .text_class import PdfTextSignals, classify_pdf_text_metadata

PDF_CLASSES = {
    "missing_pdf",
    "invalid_pdf",
    "text_pdf",
    "image_or_unextractable_pdf",
    "error",
}


def classify_pdf_document(
    pdf_path: Path | str,
    *,
    text_metadata: dict[str, Any] | None = None,
    integrity_metadata: dict[str, Any] | None = None,
    analyze_text: bool = True,
    include_integrity_hashes: bool = False,
    max_pages: int | None = 3,
    timeout_seconds: float = 30,
    max_file_bytes: int = DEFAULT_MAX_FILE_BYTES,
) -> dict[str, Any]:
    """Classify a PDF by integrity and text-layer extractability."""
    path = Path(pdf_path)
    if not path.exists():
        return {
            "path": str(path),
            "filename": path.name,
            "document_class": "missing_pdf",
            "confidence": 1.0,
            "valid_pdf": False,
            "text_extractable": False,
            "reasons": ["file_missing"],
        }

    integrity = integrity_metadata or validate_pdf_integrity(
        path,
        include_hashes=include_integrity_hashes,
        use_reader=False,
        max_file_bytes=max_file_bytes,
        timeout_seconds=timeout_seconds,
    )
    if not integrity.get("valid"):
        return {
            "path": str(path),
            "filename": path.name,
            "document_class": "invalid_pdf",
            "confidence": 0.95,
            "valid_pdf": False,
            "text_extractable": False,
            "reasons": _failed_checks(integrity),
            "integrity": integrity,
        }

    text = text_metadata
    if text is None and analyze_text:
        text = analyze_pdf_text(
            path,
            max_pages=max_pages,
            timeout_seconds=timeout_seconds,
            max_file_bytes=max_file_bytes,
        )

    if text and text.get("status") == "error":
        return {
            "path": str(path),
            "filename": path.name,
            "document_class": "error",
            "confidence": 0.6,
            "valid_pdf": True,
            "text_extractable": False,
            "reasons": [str(text.get("error") or "text_analysis_error")],
            "integrity": integrity,
            "text": text,
        }

    text_extractable = bool((text or {}).get("text_extractable"))
    total_chars = int((text or {}).get("total_chars") or 0)
    content = None
    if analyze_text:
        try:
            content = inspect_pdf_content(
                path,
                max_pages=max_pages,
                timeout_seconds=timeout_seconds,
                max_file_bytes=max_file_bytes,
            )
        except (PdfResourceLimitError, PdfTimeoutError, RuntimeError, ValueError):
            return {
                "path": str(path),
                "filename": path.name,
                "document_class": "error",
                "confidence": 0.0,
                "valid_pdf": True,
                "text_extractable": False,
                "reasons": ["content_analysis_error"],
                "integrity": integrity,
                "text": text,
            }
    if text_extractable:
        confidence = 0.95 if total_chars >= 30 else 0.82
        document_class = "text_pdf"
        reasons = ["extractable_text_layer"]
    else:
        confidence = 0.76 if text is not None else 0.55
        document_class = "image_or_unextractable_pdf"
        reasons = ["no_extractable_text_detected" if text is not None else "text_not_analyzed"]

    return {
        "path": str(path),
        "filename": path.name,
        "document_class": document_class,
        "confidence": confidence,
        "valid_pdf": True,
        "text_extractable": text_extractable,
        **(
            {
                "content_mode": content.mode,
                "content_pages": [asdict(page) for page in content.pages],
                "pdf_text_class": _legacy_pdf_text_class(content.mode, text_extractable),
            }
            if content is not None
            else {}
        ),
        "needs_ocr": not text_extractable,
        "total_chars": total_chars,
        "reasons": reasons,
        "integrity": integrity,
        "text": text,
    }


def _legacy_pdf_text_class(content_mode: ContentMode, text_extractable: bool) -> str:
    decision = classify_pdf_text_metadata(
        PdfTextSignals(
            text_extractable=text_extractable,
            has_digital_evidence=content_mode is ContentMode.NATIVE_TEXT,
            has_images=content_mode in {ContentMode.IMAGE_ONLY, ContentMode.IMAGE_WITH_TEXT_LAYER},
        )
    )
    return decision.pdf_text_class


def classify_pdf_from_metadata(
    *,
    integrity: dict[str, Any] | None = None,
    text: dict[str, Any] | None = None,
    layout: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Classify from precomputed metadata without touching the filesystem."""
    reasons: list[str] = []
    if integrity is not None and not integrity.get(
        "valid", integrity.get("overall_status") == "pass"
    ):
        return {
            "document_class": "invalid_pdf",
            "confidence": 0.95,
            "reasons": _failed_checks(integrity),
        }
    if text is not None and text.get("status") == "error":
        return {
            "document_class": "error",
            "confidence": 0.6,
            "reasons": [str(text.get("error") or "text_error")],
        }
    if text is not None and text.get("text_extractable"):
        layout_class = _layout_class(layout)
        if layout_class:
            reasons.append(f"layout:{layout_class}")
        reasons.append("extractable_text_layer")
        layout_confidence = layout.get("confidence") if isinstance(layout, dict) else None
        return {
            "document_class": layout_class or "text_pdf",
            "confidence": float(layout_confidence or 0.9) if layout_class else 0.9,
            "text_extractable": True,
            "reasons": reasons,
        }
    if text is not None:
        return {
            "document_class": "image_or_unextractable_pdf",
            "confidence": 0.76,
            "text_extractable": False,
            "reasons": ["no_extractable_text_detected"],
        }
    return {"document_class": "unknown", "confidence": 0.0, "reasons": ["insufficient_metadata"]}


def _failed_checks(integrity: dict[str, Any]) -> list[str]:
    checks_payload = integrity.get("checks")
    checks: dict[str, Any] = checks_payload if isinstance(checks_payload, dict) else {}
    reasons = [
        name
        for name, check in checks.items()
        if isinstance(check, dict) and check.get("status") == "fail"
    ]
    return reasons or ["integrity_failed"]


def _layout_class(layout: dict[str, Any] | None) -> str:
    if not isinstance(layout, dict):
        return ""
    nested_payload = layout.get("layout")
    nested: dict[str, Any] = nested_payload if isinstance(nested_payload, dict) else layout
    value = str(nested.get("document_class") or "")
    return "" if value in {"", "unknown_text", "error"} else value


def classify_manifest(
    *,
    manifest_path: Path | str,
    output_dir: Path | str,
    max_pages: int | None = 3,
    method: str = "extract-text",
    only_missing: bool = False,
    force: bool = False,
    limit: int | None = None,
    peti_root: Path | str | None = None,
    timeout_seconds: float = 30,
    max_file_bytes: int = DEFAULT_MAX_FILE_BYTES,
) -> dict[str, Any]:
    """Classify manifest rows using the focused batch adapter."""
    from .classification_manifest import classify_manifest as run

    return run(
        manifest_path=manifest_path,
        output_dir=output_dir,
        max_pages=max_pages,
        method=method,
        only_missing=only_missing,
        force=force,
        limit=limit,
        peti_root=peti_root,
        timeout_seconds=timeout_seconds,
        max_file_bytes=max_file_bytes,
    )
