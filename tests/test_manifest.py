from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gwanbo_ocr.manifest import build_manifest, iter_manifest_entries, write_manifest


def test_build_manifest_merges_peti_metadata_read_only(tmp_path: Path) -> None:
    peti = tmp_path / "peti"
    search_pdf = peti / "artifacts" / "searchThema" / "pdfs" / "2001" / "20010102" / "search-id.pdf"
    pety_pdf = peti / "artifacts" / "pety" / "pdfs" / "1999" / "19990103" / "pety-id.pdf"
    search_pdf.parent.mkdir(parents=True)
    pety_pdf.parent.mkdir(parents=True)
    search_pdf.write_bytes(b"%PDF search")
    pety_pdf.write_bytes(b"%PDF pety")

    _write_json(
        peti / "artifacts" / "searchThema" / "metadata" / "metadata.json",
        {
            "search-id": {
                "id": "search-id",
                "theme": "searchThema",
                "date": "2001-01-02",
                "title": "Search title",
                "category": "notice",
                "agency": "agency",
                "pdf": {
                    "status": "completed",
                    "path": "artifacts/searchThema/pdfs/2001/20010102/search-id.pdf",
                    "size_bytes": 11,
                    "sha256": "abc",
                },
            }
        },
    )
    _write_json(
        peti / "artifacts" / "searchThema" / "text_metadata" / "metadata.json",
        {
            "2001/20010102/search-id": {
                "filename": "search-id.pdf",
                "pdf_path": "artifacts/searchThema/pdfs/2001/20010102/search-id.pdf",
                "pages": 2,
                "text_extractable": True,
                "text_pages": 2,
                "total_chars": 123,
            }
        },
    )
    _write_json(
        peti / "artifacts" / "searchThema" / "layout_metadata" / "metadata.json",
        {
            "2001/20010102/search-id": {
                "filename": "search-id.pdf",
                "layout": {
                    "document_class": "table",
                    "metrics": {"table_count": 1},
                },
            }
        },
    )

    _write_json(
        peti / "artifacts" / "pety" / "metadata" / "metadata.json",
        {
            "pety-id": {
                "id": "pety-id",
                "theme": "pety",
                "date": "1999-01-03",
                "title": "Pety title",
                "category": "public",
                "pdf": {
                    "status": "completed",
                    "path": "artifacts/pdfs/1999/19990103/pety-id.pdf",
                },
            }
        },
    )
    _write_json(
        peti / "artifacts" / "pety" / "text_metadata" / "metadata.json",
        {
            "1999/19990103/pety-id": {
                "filename": "pety-id.pdf",
                "pdf_path": "artifacts/pety/pdfs/1999/19990103/pety-id.pdf",
                "pages": 3,
                "text_extractable": False,
            }
        },
    )

    before = sorted(path.relative_to(peti) for path in peti.rglob("*"))
    manifest = build_manifest(peti)
    after = sorted(path.relative_to(peti) for path in peti.rglob("*"))

    assert before == after
    assert [item["id"] for item in manifest] == ["pety-id", "search-id"]

    search_item = next(item for item in manifest if item["id"] == "search-id")
    assert search_item["pdf_exists"] is True
    assert search_item["pages"] == 2
    assert search_item["text_extractable"] is True
    assert search_item["layout_class"] == "table"
    assert search_item["table_count"] == 1

    pety_item = next(item for item in manifest if item["id"] == "pety-id")
    assert pety_item["pdf_path"] == "artifacts/pety/pdfs/1999/19990103/pety-id.pdf"
    assert pety_item["pdf_exists"] is True


def test_manifest_can_include_missing_when_requested(tmp_path: Path) -> None:
    peti = tmp_path / "peti"
    _write_json(
        peti / "artifacts" / "searchThema" / "metadata" / "metadata.json",
        {
            "missing-id": {
                "id": "missing-id",
                "theme": "searchThema",
                "keyword_field_regdate": "20010102",
                "pdf": {
                    "status": "pending",
                    "path": "artifacts/searchThema/pdfs/2001/20010102/missing-id.pdf",
                },
            }
        },
    )

    assert list(iter_manifest_entries(peti, themes=("searchThema",))) == []
    entries = list(
        iter_manifest_entries(
            peti,
            themes=("searchThema",),
            include_missing=True,
        )
    )
    assert len(entries) == 1
    assert entries[0].date == "2001-01-02"
    assert entries[0].pdf_exists is False


def test_write_manifest_refuses_to_write_under_peti_root(tmp_path: Path) -> None:
    peti = tmp_path / "peti"
    peti.mkdir()
    try:
        write_manifest([], peti / "manifest.jsonl", peti_root=peti)
    except ValueError:
        return
    raise AssertionError("write_manifest should reject outputs under peti root")


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
