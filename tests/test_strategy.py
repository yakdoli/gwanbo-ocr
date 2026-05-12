from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from typer.testing import CliRunner

from gwanbo_ocr.cli import app
from gwanbo_ocr.strategy import (
    assign_strategy,
    cluster_profiles,
    evaluate_clusters,
    feature_signature,
    form_score_bucket,
    page_count_bucket,
    table_density_bucket,
)

runner = CliRunner()


def test_feature_buckets_are_deterministic() -> None:
    row = {
        "theme": "searchThema",
        "year": "2024",
        "category": "고시",
        "text_mode": "text",
        "layout_class": "table_heavy",
        "pages": 12,
        "table_text_ratio": 0.51,
        "form_score": 0.3,
    }

    assert page_count_bucket(row["pages"]) == "11-50"
    assert table_density_bucket(row["table_text_ratio"], 2) == "high"
    assert form_score_bucket(row["form_score"]) == "medium"
    assert feature_signature(row) == {
        "theme": "searchThema",
        "year": "2024",
        "category": "고시",
        "text_mode": "text",
        "layout_class": "table_heavy",
        "page_count_bucket": "11-50",
        "table_density_bucket": "high",
        "form_score_bucket": "medium",
    }


def test_assign_strategy_matrix() -> None:
    table_strategy, _, _ = assign_strategy(
        {
            "text_mode": "text",
            "layout_class": "table_heavy",
            "table_density_bucket": "high",
        }
    )
    body_strategy, _, _ = assign_strategy(
        {
            "text_mode": "text",
            "layout_class": "body_text",
            "table_density_bucket": "none",
        }
    )
    paddle_strategy, _, _ = assign_strategy(
        {
            "text_mode": "image",
            "page_count_bucket": "1",
            "form_score_bucket": "none",
        }
    )
    invalid_strategy, _, _ = assign_strategy({"text_mode": "missing"})

    assert table_strategy == "native_pdfplumber_table"
    assert body_strategy == "native_text_body"
    assert paddle_strategy == "ocr_paddle_simple"
    assert invalid_strategy == "skip_invalid"


def test_cluster_profiles_groups_by_signature_and_evaluates(tmp_path: Path) -> None:
    profiles = tmp_path / "profiles.jsonl"
    rows = [
        {
            "schema_version": "pdf-profile/v1",
            "pdf_key": "a",
            "theme": "searchThema",
            "year": "2024",
            "category": "고시",
            "text_mode": "text",
            "layout_class": "body_text",
            "pages": 1,
            "table_text_ratio": 0,
            "table_count": 0,
            "form_score": 0,
        },
        {
            "schema_version": "pdf-profile/v1",
            "pdf_key": "b",
            "theme": "searchThema",
            "year": "2024",
            "category": "고시",
            "text_mode": "text",
            "layout_class": "body_text",
            "pages": 1,
            "table_text_ratio": 0,
            "table_count": 0,
            "form_score": 0,
        },
        {
            "schema_version": "pdf-profile/v1",
            "pdf_key": "c",
            "theme": "searchThema",
            "year": "2024",
            "category": "공고",
            "text_mode": "image",
            "layout_class": "image_or_unextractable_pdf",
            "pages": 2,
            "table_text_ratio": 0,
            "table_count": 0,
            "form_score": 0,
        },
    ]
    profiles.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )

    cluster_summary = cluster_profiles(profiles_path=profiles, output_dir=tmp_path / "clusters")
    clusters = [
        json.loads(line)
        for line in (tmp_path / "clusters" / "cluster_manifest.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    eval_summary = evaluate_clusters(
        clusters_path=tmp_path / "clusters" / "cluster_manifest.jsonl",
        output_dir=tmp_path / "eval",
    )

    assert cluster_summary["clusters"] == 2
    assert sorted(cluster["count"] for cluster in clusters) == [1, 2]
    assert {cluster["assigned_strategy"] for cluster in clusters} == {
        "native_text_body",
        "ocr_paddle_simple",
    }
    assert eval_summary["evaluated_clusters"] == 2


def test_strategy_cli_help() -> None:
    cluster = runner.invoke(app, ["strategy", "cluster", "--help"])
    evaluate = runner.invoke(app, ["strategy", "evaluate", "--help"])

    assert cluster.exit_code == 0
    assert "--profiles" in cluster.output
    assert evaluate.exit_code == 0
    assert "--clusters" in evaluate.output
