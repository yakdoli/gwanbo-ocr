from __future__ import annotations

import re
import unicodedata
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

WORD_PATTERN = re.compile(r"\S+")
TOKEN_PATTERN = re.compile(r"[0-9A-Za-z\u3131-\u318e\uac00-\ud7a3]+")


@dataclass(frozen=True)
class F1Result:
    precision: float
    recall: float
    f1: float
    tp: int
    fp: int
    fn: int

    def to_dict(self) -> dict[str, float | int]:
        return {
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
        }


def normalize_text(
    text: Any,
    *,
    nfkc: bool = True,
    collapse_whitespace: bool = True,
    casefold: bool = False,
) -> str:
    normalized = "" if text is None else str(text)
    if nfkc:
        normalized = unicodedata.normalize("NFKC", normalized)
    if casefold:
        normalized = normalized.casefold()
    if collapse_whitespace:
        normalized = " ".join(normalized.split())
    return normalized


def levenshtein_distance(a: Sequence[Any], b: Sequence[Any]) -> int:
    if a == b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, item_a in enumerate(a, start=1):
        current = [i]
        for j, item_b in enumerate(b, start=1):
            insert_cost = current[j - 1] + 1
            delete_cost = previous[j] + 1
            replace_cost = previous[j - 1] + (item_a != item_b)
            current.append(min(insert_cost, delete_cost, replace_cost))
        previous = current
    return previous[-1]


def cer(reference: Any, hypothesis: Any, *, normalize: bool = True) -> float:
    ref = normalize_text(reference) if normalize else str(reference or "")
    hyp = normalize_text(hypothesis) if normalize else str(hypothesis or "")
    if not ref:
        return 0.0 if not hyp else 1.0
    return levenshtein_distance(ref, hyp) / len(ref)


def wer(reference: Any, hypothesis: Any, *, normalize: bool = True) -> float:
    ref_text = normalize_text(reference) if normalize else str(reference or "")
    hyp_text = normalize_text(hypothesis) if normalize else str(hypothesis or "")
    ref = WORD_PATTERN.findall(ref_text)
    hyp = WORD_PATTERN.findall(hyp_text)
    if not ref:
        return 0.0 if not hyp else 1.0
    return levenshtein_distance(ref, hyp) / len(ref)


def precision_recall_f1(tp: int, fp: int, fn: int) -> F1Result:
    precision = tp / (tp + fp) if tp + fp else (1.0 if fn == 0 else 0.0)
    recall = tp / (tp + fn) if tp + fn else (1.0 if fp == 0 else 0.0)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return F1Result(precision=precision, recall=recall, f1=f1, tp=tp, fp=fp, fn=fn)


def critical_token_f1(
    reference_tokens: Any,
    hypothesis_tokens: Any,
    *,
    casefold: bool = True,
) -> F1Result:
    return multiset_f1(
        _coerce_tokens(reference_tokens, casefold=casefold),
        _coerce_tokens(hypothesis_tokens, casefold=casefold),
    )


def table_f1(
    reference_table: Any,
    hypothesis_table: Any,
    *,
    casefold: bool = False,
) -> F1Result:
    reference_cells = [
        normalize_text(cell, casefold=casefold)
        for cell in flatten_table_cells(reference_table)
        if normalize_text(cell, casefold=casefold)
    ]
    hypothesis_cells = [
        normalize_text(cell, casefold=casefold)
        for cell in flatten_table_cells(hypothesis_table)
        if normalize_text(cell, casefold=casefold)
    ]
    return multiset_f1(reference_cells, hypothesis_cells)


def multiset_f1(reference: Iterable[str], hypothesis: Iterable[str]) -> F1Result:
    ref_counter = Counter(reference)
    hyp_counter = Counter(hypothesis)
    tp = sum((ref_counter & hyp_counter).values())
    fp = sum((hyp_counter - ref_counter).values())
    fn = sum((ref_counter - hyp_counter).values())
    return precision_recall_f1(tp, fp, fn)


def flatten_table_cells(table: Any) -> list[str]:
    if table is None:
        return []
    if isinstance(table, str):
        return [table]
    if isinstance(table, Mapping):
        for key in ("cells", "rows", "data", "table"):
            if key in table:
                return flatten_table_cells(table[key])
        if "text" in table:
            return [str(table["text"])]
        return [str(value) for value in table.values() if value is not None]
    if isinstance(table, Iterable):
        cells: list[str] = []
        for item in table:
            cells.extend(flatten_table_cells(item))
        return cells
    return [str(table)]


def evaluate_document(
    reference_text: Any,
    hypothesis_text: Any,
    *,
    reference_tables: Any = None,
    hypothesis_tables: Any = None,
    reference_critical_tokens: Any = None,
    hypothesis_critical_tokens: Any = None,
) -> dict[str, Any]:
    token_ref = (
        reference_critical_tokens
        if reference_critical_tokens is not None
        else extract_critical_tokens(reference_text)
    )
    token_hyp = (
        hypothesis_critical_tokens
        if hypothesis_critical_tokens is not None
        else extract_critical_tokens(hypothesis_text)
    )
    table = table_f1(reference_tables, hypothesis_tables)
    critical = critical_token_f1(token_ref, token_hyp)
    return {
        "cer": cer(reference_text, hypothesis_text),
        "wer": wer(reference_text, hypothesis_text),
        "table_f1": table.f1,
        "table_precision": table.precision,
        "table_recall": table.recall,
        "critical_token_f1": critical.f1,
        "critical_token_precision": critical.precision,
        "critical_token_recall": critical.recall,
        "table_counts": table.to_dict(),
        "critical_token_counts": critical.to_dict(),
    }


def extract_critical_tokens(text: Any) -> list[str]:
    normalized = normalize_text(text, casefold=True)
    return TOKEN_PATTERN.findall(normalized)


def _coerce_tokens(value: Any, *, casefold: bool) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [normalize_text(token, casefold=casefold) for token in TOKEN_PATTERN.findall(value)]
    if isinstance(value, Mapping):
        tokens: list[str] = []
        for item in value.values():
            tokens.extend(_coerce_tokens(item, casefold=casefold))
        return tokens
    if isinstance(value, Iterable):
        return [normalize_text(token, casefold=casefold) for token in value]
    return [normalize_text(value, casefold=casefold)]
