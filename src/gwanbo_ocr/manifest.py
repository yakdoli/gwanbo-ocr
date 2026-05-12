from __future__ import annotations

import json
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_PETI_ROOT = Path("/root/peti")
DEFAULT_THEMES = ("searchThema", "pety")


@dataclass(frozen=True)
class ManifestEntry:
    id: str
    theme: str
    date: str
    year: str
    title: str
    category: str
    agency: str
    content_id: str
    toc_id: str
    status: str
    url: str
    pdf_path: str
    pdf_abs_path: str
    pdf_status: str
    pdf_exists: bool
    size_bytes: int | None
    sha256: str
    pages: int | None
    text_extractable: bool | None
    text_pages: int | None
    total_chars: int | None
    scanned_pages: int | None
    page_error_count: int | None
    layout_class: str
    table_count: int | None
    metadata_key: str
    source_metadata_path: str
    text_metadata_key: str
    layout_metadata_key: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "theme": self.theme,
            "date": self.date,
            "year": self.year,
            "title": self.title,
            "category": self.category,
            "agency": self.agency,
            "content_id": self.content_id,
            "toc_id": self.toc_id,
            "status": self.status,
            "url": self.url,
            "pdf_path": self.pdf_path,
            "pdf_abs_path": self.pdf_abs_path,
            "pdf_status": self.pdf_status,
            "pdf_exists": self.pdf_exists,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
            "pages": self.pages,
            "text_extractable": self.text_extractable,
            "text_pages": self.text_pages,
            "total_chars": self.total_chars,
            "scanned_pages": self.scanned_pages,
            "page_error_count": self.page_error_count,
            "layout_class": self.layout_class,
            "table_count": self.table_count,
            "metadata_key": self.metadata_key,
            "source_metadata_path": self.source_metadata_path,
            "text_metadata_key": self.text_metadata_key,
            "layout_metadata_key": self.layout_metadata_key,
        }


class JsonObjectStream:
    """Stream a top-level JSON object as key/value pairs."""

    def __init__(self, path: Path, chunk_size: int = 1024 * 1024) -> None:
        self.path = path
        self.chunk_size = chunk_size
        self.decoder = json.JSONDecoder()
        self.buffer = ""
        self.pos = 0
        self.eof = False

    def __iter__(self) -> Iterator[tuple[str, Any]]:
        with self.path.open("r", encoding="utf-8") as handle:
            self.handle = handle
            self._fill()
            self._consume_char("{")
            while True:
                self._skip_ws()
                if self._peek_char() == "}":
                    self.pos += 1
                    return
                key = self._decode_next()
                if not isinstance(key, str):
                    raise ValueError(f"Expected string key in {self.path}")
                self._consume_char(":")
                value = self._decode_next()
                yield key, value
                self._skip_ws()
                marker = self._peek_char()
                if marker == ",":
                    self.pos += 1
                    continue
                if marker == "}":
                    self.pos += 1
                    return
                raise ValueError(f"Expected ',' or '}}' in {self.path}")

    def _fill(self) -> None:
        if self.eof:
            return
        chunk = self.handle.read(self.chunk_size)
        if not chunk:
            self.eof = True
            return
        if self.pos:
            self.buffer = self.buffer[self.pos :]
            self.pos = 0
        self.buffer += chunk

    def _ensure_buffer(self) -> None:
        while self.pos >= len(self.buffer) and not self.eof:
            self._fill()

    def _skip_ws(self) -> None:
        while True:
            self._ensure_buffer()
            while self.pos < len(self.buffer) and self.buffer[self.pos].isspace():
                self.pos += 1
            if self.pos < len(self.buffer) or self.eof:
                return

    def _peek_char(self) -> str:
        self._skip_ws()
        self._ensure_buffer()
        if self.pos >= len(self.buffer):
            raise ValueError(f"Unexpected end of JSON in {self.path}")
        return self.buffer[self.pos]

    def _consume_char(self, expected: str) -> None:
        actual = self._peek_char()
        if actual != expected:
            raise ValueError(f"Expected {expected!r} in {self.path}, got {actual!r}")
        self.pos += 1

    def _decode_next(self) -> Any:
        self._skip_ws()
        while True:
            try:
                value, end = self.decoder.raw_decode(self.buffer, self.pos)
            except json.JSONDecodeError:
                if self.eof:
                    raise
                self._fill()
                continue
            self.pos = end
            return value


def iter_json_object(path: str | Path) -> Iterator[tuple[str, Any]]:
    yield from JsonObjectStream(Path(path))


def iter_manifest_entries(
    peti_root: str | Path = DEFAULT_PETI_ROOT,
    *,
    themes: Sequence[str] = DEFAULT_THEMES,
    include_missing: bool = False,
    validate_files: bool = True,
    max_entries: int | None = None,
) -> Iterator[ManifestEntry]:
    root = Path(peti_root)
    yielded = 0
    for theme in themes:
        metadata_path = root / "artifacts" / theme / "metadata" / "metadata.json"
        if not metadata_path.is_file():
            raise FileNotFoundError(metadata_path)

        text_index = _load_sidecar_index(
            root / "artifacts" / theme / "text_metadata" / "metadata.json"
        )
        layout_index = _load_sidecar_index(
            root / "artifacts" / theme / "layout_metadata" / "metadata.json"
        )

        for metadata_key, metadata in iter_json_object(metadata_path):
            if not isinstance(metadata, Mapping):
                continue
            entry = _entry_from_metadata(
                root=root,
                theme=theme,
                metadata_key=metadata_key,
                metadata=metadata,
                source_metadata_path=metadata_path,
                text_index=text_index,
                layout_index=layout_index,
                include_missing=include_missing,
                validate_files=validate_files,
            )
            if entry is None:
                continue
            yield entry
            yielded += 1
            if max_entries is not None and yielded >= max_entries:
                return


def build_manifest(
    peti_root: str | Path = DEFAULT_PETI_ROOT,
    *,
    themes: Sequence[str] = DEFAULT_THEMES,
    include_missing: bool = False,
    validate_files: bool = True,
    max_entries: int | None = None,
) -> list[dict[str, Any]]:
    entries = [
        entry.to_dict()
        for entry in iter_manifest_entries(
            peti_root,
            themes=themes,
            include_missing=include_missing,
            validate_files=validate_files,
            max_entries=max_entries,
        )
    ]
    entries.sort(key=lambda item: (item["theme"], item["date"], item["id"]))
    return entries


def manifest_summary(entries: Iterable[ManifestEntry | Mapping[str, Any]]) -> dict[str, Any]:
    total = 0
    existing = 0
    by_theme: dict[str, int] = {}
    by_year: dict[str, int] = {}
    by_status: dict[str, int] = {}
    total_bytes = 0
    total_pages = 0
    text_extractable = 0

    for entry in entries:
        item = _as_mapping(entry)
        total += 1
        theme = str(item.get("theme") or "")
        year = str(item.get("year") or "")
        status = str(item.get("pdf_status") or "")
        by_theme[theme] = by_theme.get(theme, 0) + 1
        by_year[year] = by_year.get(year, 0) + 1
        by_status[status] = by_status.get(status, 0) + 1
        if item.get("pdf_exists"):
            existing += 1
        if item.get("text_extractable") is True:
            text_extractable += 1
        total_bytes += int(item.get("size_bytes") or 0)
        total_pages += int(item.get("pages") or 0)

    return {
        "total": total,
        "pdf_existing": existing,
        "pdf_missing": total - existing,
        "text_extractable": text_extractable,
        "total_bytes": total_bytes,
        "total_pages": total_pages,
        "by_theme": dict(sorted(by_theme.items())),
        "by_year": dict(sorted(by_year.items())),
        "by_pdf_status": dict(sorted(by_status.items())),
    }


def write_manifest(
    entries: Iterable[ManifestEntry | Mapping[str, Any]],
    output_path: str | Path,
    *,
    peti_root: str | Path = DEFAULT_PETI_ROOT,
    jsonl: bool = True,
) -> Path:
    path = Path(output_path)
    assert_not_under_peti(path, peti_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        if jsonl:
            for entry in entries:
                row = _as_mapping(entry)
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
                handle.write("\n")
        else:
            rows = [_as_mapping(entry) for entry in entries]
            json.dump(rows, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
    return path


def build_peti_manifest(
    *,
    peti_root: str | Path = DEFAULT_PETI_ROOT,
    output_path: str | Path,
    sources: Sequence[str] = DEFAULT_THEMES,
    limit: int | None = None,
    include_issue_pdfs: bool = True,
) -> dict[str, Any]:
    """Build a JSONL manifest from /root/peti without writing under that tree.

    ``include_issue_pdfs`` is accepted for CLI compatibility. Issue PDFs are
    represented in `/root/peti` metadata when their item JSON points at them;
    this builder does not mutate or repair those source records.
    """
    root = Path(peti_root)
    output = Path(output_path)
    assert_not_under_peti(output, root)
    output.parent.mkdir(parents=True, exist_ok=True)

    counts: dict[str, Any] = {
        "total": 0,
        "pdf_existing": 0,
        "pdf_missing": 0,
        "text_extractable": 0,
        "by_theme": {},
        "include_issue_pdfs": include_issue_pdfs,
    }
    with output.open("w", encoding="utf-8") as handle:
        for entry in iter_manifest_entries(
            root,
            themes=tuple(sources),
            include_missing=True,
            validate_files=True,
            max_entries=limit,
        ):
            row = entry.to_dict()
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
            counts["total"] += 1
            if row.get("pdf_exists"):
                counts["pdf_existing"] += 1
            else:
                counts["pdf_missing"] += 1
            if row.get("text_extractable") is True:
                counts["text_extractable"] += 1
            theme = str(row.get("theme") or "")
            counts["by_theme"][theme] = counts["by_theme"].get(theme, 0) + 1

    summary = {
        "status": "ok",
        "peti_root": str(root),
        "manifest": str(output),
        **counts,
    }
    summary_path = output.with_suffix(output.suffix + ".summary.json")
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    summary["summary_path"] = str(summary_path)
    return summary


def assert_not_under_peti(output_path: str | Path, peti_root: str | Path) -> None:
    root = Path(peti_root).resolve(strict=False)
    path = Path(output_path).resolve(strict=False)
    try:
        path.relative_to(root)
    except ValueError:
        return
    raise ValueError(f"Refusing to write manifest output under read-only peti root: {path}")


def _entry_from_metadata(
    *,
    root: Path,
    theme: str,
    metadata_key: str,
    metadata: Mapping[str, Any],
    source_metadata_path: Path,
    text_index: Mapping[str, tuple[str, Mapping[str, Any]]],
    layout_index: Mapping[str, tuple[str, Mapping[str, Any]]],
    include_missing: bool,
    validate_files: bool,
) -> ManifestEntry | None:
    item_id = _string(metadata.get("id") or metadata.get("toc_id") or metadata_key)
    text_lookup = text_index.get(item_id)
    text_key = text_lookup[0] if text_lookup is not None else ""
    text_meta: Mapping[str, Any] = text_lookup[1] if text_lookup is not None else {}
    layout_lookup = layout_index.get(item_id)
    layout_key = layout_lookup[0] if layout_lookup is not None else ""
    layout_meta: Mapping[str, Any] = layout_lookup[1] if layout_lookup is not None else {}
    pdf_payload = metadata.get("pdf")
    pdf_info: Mapping[str, Any] = pdf_payload if isinstance(pdf_payload, Mapping) else {}
    pdf_path = _best_pdf_path(root, theme, metadata, text_meta)
    pdf_abs_path = str((root / pdf_path).resolve(strict=False)) if pdf_path else ""
    pdf_exists = Path(pdf_abs_path).is_file() if validate_files and pdf_abs_path else False
    if validate_files and not pdf_exists and not include_missing:
        return None

    date = _string(metadata.get("date") or _date_from_metadata(metadata))
    year = _string(metadata.get("year") or date[:4] or metadata.get("stored_field_year"))
    text_extractable = _optional_bool(
        text_meta.get("text_extractable")
        if text_meta
        else _deep_get(metadata, ("ocr", "extracted_metadata", "text_extractable"))
    )
    layout_payload = layout_meta.get("layout")
    layout: Mapping[str, Any] = layout_payload if isinstance(layout_payload, Mapping) else {}
    metrics_payload = layout.get("metrics")
    layout_metrics: Mapping[str, Any] = (
        metrics_payload if isinstance(metrics_payload, Mapping) else {}
    )
    ocr_meta = _deep_get(metadata, ("ocr", "extracted_metadata"))
    if not isinstance(ocr_meta, Mapping):
        ocr_meta = {}

    return ManifestEntry(
        id=item_id,
        theme=_string(metadata.get("theme") or theme),
        date=date,
        year=year,
        title=_string(metadata.get("title") or metadata.get("stored_field_subject")),
        category=_string(metadata.get("category") or metadata.get("stored_category_name")),
        agency=_string(metadata.get("agency") or metadata.get("stored_organ_nm")),
        content_id=_string(metadata.get("content_id")),
        toc_id=_string(metadata.get("toc_id") or metadata.get("stored_toc_seq")),
        status=_string(metadata.get("status")),
        url=_string(metadata.get("url") or metadata.get("source_url")),
        pdf_path=pdf_path,
        pdf_abs_path=pdf_abs_path,
        pdf_status=_string(pdf_info.get("status") or text_meta.get("status")),
        pdf_exists=pdf_exists,
        size_bytes=_optional_int(text_meta.get("size_bytes") or pdf_info.get("size_bytes")),
        sha256=_string(pdf_info.get("sha256")),
        pages=_optional_int(text_meta.get("pages") or ocr_meta.get("pages")),
        text_extractable=text_extractable,
        text_pages=_optional_int(text_meta.get("text_pages") or ocr_meta.get("text_pages")),
        total_chars=_optional_int(text_meta.get("total_chars") or ocr_meta.get("total_chars")),
        scanned_pages=_optional_int(text_meta.get("scanned_pages")),
        page_error_count=_optional_int(text_meta.get("page_error_count")),
        layout_class=_string(layout.get("document_class")),
        table_count=_optional_int(
            layout_meta.get("table_count") or layout_metrics.get("table_count")
        ),
        metadata_key=metadata_key,
        source_metadata_path=str(source_metadata_path),
        text_metadata_key=text_key,
        layout_metadata_key=layout_key,
    )


def _load_sidecar_index(path: Path) -> dict[str, tuple[str, Mapping[str, Any]]]:
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, Mapping):
        return {}

    index: dict[str, tuple[str, Mapping[str, Any]]] = {}
    for key, value in data.items():
        if not isinstance(value, Mapping):
            continue
        item_id = _sidecar_item_id(key, value)
        if item_id:
            index[item_id] = (str(key), value)
    return index


def _sidecar_item_id(key: Any, value: Mapping[str, Any]) -> str:
    for candidate in (
        value.get("id"),
        value.get("toc_id"),
        Path(_string(value.get("filename"))).stem,
        Path(_string(value.get("pdf_path"))).stem,
        Path(_string(value.get("path"))).stem,
        Path(str(key)).stem,
    ):
        text = _string(candidate)
        if text:
            return text
    return ""


def _best_pdf_path(
    root: Path,
    theme: str,
    metadata: Mapping[str, Any],
    text_meta: Mapping[str, Any],
) -> str:
    pdf_payload = metadata.get("pdf")
    pdf_info: Mapping[str, Any] = pdf_payload if isinstance(pdf_payload, Mapping) else {}
    candidates = [
        text_meta.get("pdf_path"),
        text_meta.get("path"),
        pdf_info.get("path"),
        metadata.get("stored_pdf_file_path"),
    ]
    normalized = [
        path
        for candidate in candidates
        if (path := _normalize_pdf_path(root, theme, _string(candidate)))
    ]
    for path in normalized:
        if (root / path).is_file():
            return path
    return normalized[0] if normalized else ""


def _normalize_pdf_path(root: Path, theme: str, path: str) -> str:
    if not path:
        return ""
    path = path.replace("\\", "/").strip()
    if path.startswith("/ndata/"):
        return _ndata_to_artifact_path(theme, path)
    source = Path(path)
    if source.is_absolute():
        try:
            return str(source.resolve(strict=False).relative_to(root.resolve(strict=False)))
        except ValueError:
            return path
    parts = path.split("/")
    if len(parts) >= 3 and parts[0] == "artifacts" and parts[1] == "pdfs":
        return "/".join(["artifacts", theme, *parts[1:]])
    if len(parts) >= 3 and parts[0] == "artifacts" and parts[1] == "ocr_ready":
        return "/".join(["artifacts", theme, *parts[1:]])
    return path


def _ndata_to_artifact_path(theme: str, path: str) -> str:
    filename = Path(path).name
    if not filename:
        return ""
    return f"artifacts/{theme}/pdfs/{filename}"


def _date_from_metadata(metadata: Mapping[str, Any]) -> str:
    regdate = _string(metadata.get("keyword_field_regdate"))
    if len(regdate) == 8 and regdate.isdigit():
        return f"{regdate[:4]}-{regdate[4:6]}-{regdate[6:8]}"
    year = _string(metadata.get("stored_field_year"))
    month = _string(metadata.get("stored_field_month")).zfill(2)
    day = _string(metadata.get("stored_field_day")).zfill(2)
    if year and month.strip("0") and day.strip("0"):
        return f"{year}-{month}-{day}"
    return ""


def _as_mapping(entry: ManifestEntry | Mapping[str, Any]) -> Mapping[str, Any]:
    if isinstance(entry, ManifestEntry):
        return entry.to_dict()
    return entry


def _deep_get(mapping: Mapping[str, Any], path: Sequence[str]) -> Any:
    value: Any = mapping
    for key in path:
        if not isinstance(value, Mapping):
            return None
        value = value.get(key)
    return value


def _string(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def _optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_bool(value: Any) -> bool | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "y"}:
            return True
        if lowered in {"0", "false", "no", "n"}:
            return False
    return bool(value)
