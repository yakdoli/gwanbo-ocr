from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from gwanbo_ocr.metrics import cer, critical_token_f1, table_f1, wer


def test_cer_and_wer_use_reference_length() -> None:
    assert cer("abcd", "abxd") == 0.25
    assert wer("alpha beta gamma", "alpha gamma") == 1 / 3
    assert cer("", "") == 0.0
    assert wer("", "extra") == 1.0


def test_table_f1_scores_cell_multisets() -> None:
    result = table_f1([["agency", "date"], ["A", "2026"]], [["agency", "date"], ["B", "2026"]])

    assert result.tp == 3
    assert result.fp == 1
    assert result.fn == 1
    assert result.f1 == 0.75


def test_critical_token_f1_casefolds_and_supports_hangul() -> None:
    result = critical_token_f1(["Gwanbo", "notice"], ["gwanbo", "notice"])

    assert result.f1 == 1.0

    hangul = critical_token_f1("\uad00\ubcf4 title", "\uad00\ubcf4 title")
    assert hangul.tp == 2
