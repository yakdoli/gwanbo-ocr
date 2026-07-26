from __future__ import annotations

import json
import os
import sys
from collections.abc import Callable
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gwanbo_ocr.pdf.io import (
    ensure_directory,
    file_sha256,
    iter_pdf_paths,
    limited,
    normalize_text,
    pdf_key_from_path,
    pdf_key_from_row,
    read_json,
    read_jsonl,
    relative_to_or_str,
    resolve_path,
    resolve_pdf_path,
    write_json_atomic,
    write_json_lines_atomic,
    write_jsonl_atomic,
)


def test_read_json_returns_none_for_missing_file(tmp_path: Path) -> None:
    result = read_json(tmp_path / "no_such_file.json")
    assert result is None


def test_read_json_returns_none_for_malformed_json(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text("{not valid json", encoding="utf-8")
    assert read_json(bad) is None


def test_read_json_returns_parsed_object(tmp_path: Path) -> None:
    p = tmp_path / "data.json"
    p.write_text('{"key": 42}', encoding="utf-8")
    assert read_json(p) == {"key": 42}


def test_write_json_atomic_creates_file_with_valid_json(tmp_path: Path) -> None:
    target = tmp_path / "sub" / "out.json"
    write_json_atomic(target, {"a": 1, "b": [1, 2]})
    assert target.exists()
    assert json.loads(target.read_text(encoding="utf-8")) == {"a": 1, "b": [1, 2]}


def test_write_json_atomic_overwrites_existing(tmp_path: Path) -> None:
    target = tmp_path / "out.json"
    write_json_atomic(target, {"v": 1})
    write_json_atomic(target, {"v": 2})
    assert json.loads(target.read_text(encoding="utf-8")) == {"v": 2}


def test_read_jsonl_yields_decoded_records(tmp_path: Path) -> None:
    lines = tmp_path / "data.jsonl"
    lines.write_text('{"a": 1}\n\n{"b": 2}\n', encoding="utf-8")
    rows = list(read_jsonl(lines))
    assert rows == [{"a": 1}, {"b": 2}]


def test_write_jsonl_atomic_roundtrips(tmp_path: Path) -> None:
    target = tmp_path / "out.jsonl"
    rows = [{"id": i, "v": f"val{i}"} for i in range(5)]
    write_jsonl_atomic(target, rows)
    assert list(read_jsonl(target)) == rows


@pytest.mark.parametrize(
    "write",
    [
        lambda path: write_json_atomic(path, {"safe": True}),
        lambda path: write_jsonl_atomic(path, ({"safe": True},)),
        lambda path: write_json_lines_atomic(path, ('{"safe":true}',)),
    ],
)
def test_atomic_writers_do_not_follow_predictable_temp_symlinks(
    tmp_path: Path, write: Callable[[Path], None]
) -> None:
    victim = tmp_path / "victim.txt"
    victim.write_text("safe", encoding="utf-8")
    target = tmp_path / "output.json"
    target.with_name(f"{target.name}.{os.getpid()}.tmp").symlink_to(victim)

    write(target)

    assert victim.read_text(encoding="utf-8") == "safe"


def test_normalize_text_collapses_whitespace() -> None:
    assert normalize_text("  hello   world  ") == "hello world"
    assert normalize_text("a\t\tb\n\nc") == "a b c"
    assert normalize_text("") == ""


def test_resolve_path_absolute_is_returned_as_is(tmp_path: Path) -> None:
    result = resolve_path(tmp_path)
    assert result == tmp_path.resolve()


def test_resolve_path_relative_resolved_against_base(tmp_path: Path) -> None:
    (tmp_path / "child.txt").write_text("x", encoding="utf-8")
    resolved = resolve_path("child.txt", base_dir=tmp_path)
    assert resolved == (tmp_path / "child.txt").resolve()


def test_resolve_pdf_path_prefers_absolute_manifest_path(tmp_path: Path) -> None:
    absolute = tmp_path / "absolute.pdf"
    relative = tmp_path / "relative.pdf"
    row = {"pdf_abs_path": str(absolute), "pdf_path": str(relative)}

    assert resolve_pdf_path(row) == absolute


def test_resolve_pdf_path_uses_peti_root_for_artifact_relative_path(tmp_path: Path) -> None:
    row = {"pdf_path": "artifacts/searchThema/pdfs/2024/doc.pdf"}

    assert resolve_pdf_path(row, peti_root=tmp_path) == (
        tmp_path / "artifacts/searchThema/pdfs/2024/doc.pdf"
    ).resolve(strict=False)


def test_pdf_key_from_row_uses_metadata_before_path() -> None:
    row = {"metadata_key": "a\\b/c", "pdf_path": "artifacts/x.pdf"}

    assert pdf_key_from_row(row) == "a/b/c"


def test_relative_to_or_str_returns_posix_string(tmp_path: Path) -> None:
    path = tmp_path / "a" / "b" / "c.txt"
    result = relative_to_or_str(path, tmp_path)
    assert result == "a/b/c.txt"


def test_relative_to_or_str_returns_str_when_not_under_root(tmp_path: Path) -> None:
    path = Path("/some/absolute/path.txt")
    result = relative_to_or_str(path, tmp_path)
    assert result == str(path)


def test_pdf_key_from_path_strips_extension_and_makes_relative(tmp_path: Path) -> None:
    pdf_dir = tmp_path / "pdfs"
    pdf_path = pdf_dir / "2024" / "20240101" / "doc-001.pdf"
    assert pdf_key_from_path(pdf_path, pdf_dir) == "2024/20240101/doc-001"


def test_file_sha256_is_deterministic(tmp_path: Path) -> None:
    f = tmp_path / "f.bin"
    f.write_bytes(b"hello world")
    h1 = file_sha256(f)
    h2 = file_sha256(f)
    assert h1 == h2
    assert len(h1) == 64


def test_file_sha256_differs_for_different_content(tmp_path: Path) -> None:
    a = tmp_path / "a.bin"
    b = tmp_path / "b.bin"
    a.write_bytes(b"aaa")
    b.write_bytes(b"bbb")
    assert file_sha256(a) != file_sha256(b)


def test_iter_pdf_paths_yields_sorted_pdfs(tmp_path: Path) -> None:
    (tmp_path / "b.pdf").write_bytes(b"%PDF")
    (tmp_path / "a.pdf").write_bytes(b"%PDF")
    (tmp_path / "other.txt").write_bytes(b"text")
    paths = list(iter_pdf_paths(tmp_path))
    assert [p.name for p in paths] == ["a.pdf", "b.pdf"]


def test_iter_pdf_paths_handles_missing_directory(tmp_path: Path) -> None:
    paths = list(iter_pdf_paths(tmp_path / "nonexistent"))
    assert paths == []


def test_limited_restricts_iteration() -> None:
    items = [Path(f"/f{i}.pdf") for i in range(10)]
    result = list(limited(items, 3))
    assert len(result) == 3


def test_limited_passes_all_when_none(tmp_path: Path) -> None:
    items = [Path(f"/f{i}.pdf") for i in range(5)]
    assert list(limited(items, None)) == items


def test_ensure_directory_creates_nested(tmp_path: Path) -> None:
    target = tmp_path / "a" / "b" / "c"
    result = ensure_directory(target)
    assert result == target
    assert target.is_dir()
