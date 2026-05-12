from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_SAMPLE_SEED = "gwanbo-ocr-session-d"
DEFAULT_SAMPLE_FIELDS = (
    "id",
    "theme",
    "date",
    "year",
    "category",
    "agency",
    "title",
    "pdf_path",
    "pages",
    "size_bytes",
    "text_extractable",
    "layout_class",
    "table_count",
)
DEFAULT_TOKEN_FIELDS = ("title", "category", "agency", "year")
TOKEN_PATTERN = re.compile(r"[0-9A-Za-z\u3131-\u318e\uac00-\ud7a3]+")


@dataclass(frozen=True)
class SampleSuite:
    seed: str
    requested_size: int
    generated_size: int
    strata: tuple[str, ...]
    samples: tuple[dict[str, Any], ...]
    summary: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "requested_size": self.requested_size,
            "generated_size": self.generated_size,
            "strata": list(self.strata),
            "summary": self.summary,
            "samples": list(self.samples),
        }


def stable_digest(value: Any, seed: str = DEFAULT_SAMPLE_SEED) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(f"{seed}\0{payload}".encode()).hexdigest()


def sample_entries(
    entries: Iterable[Mapping[str, Any] | Any],
    *,
    size: int,
    seed: str = DEFAULT_SAMPLE_SEED,
    strata: Sequence[str] = ("theme", "year"),
    min_per_stratum: int = 1,
    require_existing_pdf: bool = True,
) -> list[dict[str, Any]]:
    if size <= 0:
        return []

    rows = [_as_mapping(entry) for entry in entries]
    if require_existing_pdf:
        rows = [row for row in rows if bool(row.get("pdf_exists", True))]

    groups: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = _stratum_key(row, strata)
        groups.setdefault(key, []).append(row)

    for key, group in groups.items():
        group.sort(key=lambda row: stable_digest(_identity(row), f"{seed}:{key}"))

    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    remaining = size
    if min_per_stratum > 0:
        group_order = sorted(
            groups,
            key=lambda key: stable_digest({"stratum": key}, seed),
        )
        for key in group_order:
            for row in groups[key][:min_per_stratum]:
                if remaining <= 0:
                    break
                row_id = _identity(row)
                if row_id in selected_ids:
                    continue
                selected.append(row)
                selected_ids.add(row_id)
                remaining -= 1
            if remaining <= 0:
                break

    pool = [
        row
        for group in groups.values()
        for row in group
        if _identity(row) not in selected_ids
    ]
    pool.sort(key=lambda row: stable_digest(_identity(row), seed))
    for row in pool:
        if len(selected) >= size:
            break
        selected.append(row)
        selected_ids.add(_identity(row))

    selected.sort(key=lambda row: stable_digest(_identity(row), f"{seed}:output"))
    return selected


def build_sample_suite(
    entries: Iterable[Mapping[str, Any] | Any],
    *,
    size: int,
    seed: str = DEFAULT_SAMPLE_SEED,
    strata: Sequence[str] = ("theme", "year"),
    min_per_stratum: int = 1,
    require_existing_pdf: bool = True,
    fields: Sequence[str] = DEFAULT_SAMPLE_FIELDS,
    token_fields: Sequence[str] = DEFAULT_TOKEN_FIELDS,
) -> SampleSuite:
    selected = sample_entries(
        entries,
        size=size,
        seed=seed,
        strata=strata,
        min_per_stratum=min_per_stratum,
        require_existing_pdf=require_existing_pdf,
    )
    samples = tuple(
        _sample_payload(index, row, fields, strata, token_fields)
        for index, row in enumerate(selected, start=1)
    )
    return SampleSuite(
        seed=seed,
        requested_size=size,
        generated_size=len(samples),
        strata=tuple(strata),
        samples=samples,
        summary=_suite_summary(samples, strata),
    )


def write_sample_suite(suite: SampleSuite, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(suite.to_dict(), handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    return path


def critical_tokens_for_entry(
    entry: Mapping[str, Any] | Any,
    *,
    fields: Sequence[str] = DEFAULT_TOKEN_FIELDS,
) -> list[str]:
    row = _as_mapping(entry)
    tokens: list[str] = []
    for field in fields:
        value = row.get(field)
        if value is None:
            continue
        tokens.extend(tokenize_critical_text(str(value)))
    return sorted(dict.fromkeys(tokens))


def tokenize_critical_text(text: str) -> list[str]:
    return [match.group(0).casefold() for match in TOKEN_PATTERN.finditer(text)]


def _sample_payload(
    index: int,
    row: Mapping[str, Any],
    fields: Sequence[str],
    strata: Sequence[str],
    token_fields: Sequence[str],
) -> dict[str, Any]:
    payload = {field: row.get(field) for field in fields if field in row}
    payload["sample_index"] = index
    payload["sample_id"] = stable_digest(_identity(row))[:16]
    payload["stratum"] = {
        field: _derived_field(row, field)
        for field in strata
    }
    payload["critical_tokens"] = critical_tokens_for_entry(row, fields=token_fields)
    return payload


def _suite_summary(
    samples: Sequence[Mapping[str, Any]],
    strata: Sequence[str],
) -> dict[str, Any]:
    by_theme: dict[str, int] = {}
    by_year: dict[str, int] = {}
    by_stratum: dict[str, int] = {}
    for sample in samples:
        theme = str(sample.get("theme") or "")
        year = str(sample.get("year") or "")
        by_theme[theme] = by_theme.get(theme, 0) + 1
        by_year[year] = by_year.get(year, 0) + 1
        stratum = sample.get("stratum")
        if isinstance(stratum, Mapping):
            key = "|".join(f"{field}={stratum.get(field, '')}" for field in strata)
            by_stratum[key] = by_stratum.get(key, 0) + 1
    return {
        "by_theme": dict(sorted(by_theme.items())),
        "by_year": dict(sorted(by_year.items())),
        "by_stratum": dict(sorted(by_stratum.items())),
    }


def _stratum_key(row: Mapping[str, Any], strata: Sequence[str]) -> tuple[str, ...]:
    if not strata:
        return ("all",)
    return tuple(str(_derived_field(row, field) or "") for field in strata)


def _derived_field(row: Mapping[str, Any], field: str) -> Any:
    if field == "year" and not row.get("year"):
        return str(row.get("date") or "")[:4]
    return row.get(field)


def _identity(row: Mapping[str, Any]) -> str:
    return str(row.get("id") or row.get("pdf_path") or row)


def _as_mapping(entry: Mapping[str, Any] | Any) -> dict[str, Any]:
    if isinstance(entry, Mapping):
        return dict(entry)
    if hasattr(entry, "to_dict"):
        return dict(entry.to_dict())
    return dict(vars(entry))
