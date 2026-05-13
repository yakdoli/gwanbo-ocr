#!/usr/bin/env python
"""Generate a reviewer-friendly static HTML package from clustering outputs.

Usage example:
python scripts/generate_cluster_review_report.py \
  --clusters /tmp/gwanbo-ocr-clustering-max/clusters/cluster_manifest.jsonl \
  --profiles /tmp/gwanbo-ocr-clustering-max/profiles/manifest.jsonl \
  --output /tmp/gwanbo-ocr-cluster-review \
  --top-clusters 120 \
  --thumbnail-per-cluster 1
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import fitz
from PIL import Image, ImageDraw, ImageFont


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, 1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                yield json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid jsonl at line {line_no}: {path}") from exc


def draw_bar_chart(
    title: str,
    values: dict[str, int],
    out_path: Path,
    rotate_x: int = 28,
    width: int = 1300,
    height: int = 720,
) -> None:
    if not values:
        return

    pairs = sorted(values.items(), key=lambda kv: kv[1], reverse=True)
    font = ImageFont.load_default()
    max_value = max(values.values())
    left = 290
    bottom = height - 90
    top = 80
    right = width - 40
    bar_gap = 12
    max_bars = min(30, len(pairs))
    top_pairs = pairs[:max_bars]
    if max_bars == 0:
        return

    bar_height = max(24, min(40, (bottom - top) // max_bars - bar_gap))
    chart_height = max_bars * (bar_height + bar_gap) + 40
    img_height = max(height, top + chart_height + 20)
    image = Image.new("RGB", (width, img_height), "white")
    draw = ImageDraw.Draw(image)

    plot_w = right - left
    draw.rectangle((left, top, right, bottom), outline="#222", width=1)
    draw.text((30, 20), title, fill="#111111", font=font)

    for idx, (label, count) in enumerate(top_pairs):
        y1 = top + 20 + idx * (bar_height + bar_gap)
        y2 = y1 + bar_height
        x2 = left + int((count / max_value) * plot_w)
        draw.rectangle((left, y1, x2, y2), fill="#1f77b4", outline="#1f77b4")
        y_text = y1 + 2
        display_label = label[:44] + ("..." if len(label) > 44 else "")
        draw.text((20, y_text), display_label, fill="#111111", font=font)
        draw.text((x2 + 6, y_text), str(count), fill="#222222", font=font)

    # Axis ticks
    tick_steps = 5
    for i in range(tick_steps + 1):
        x = left + int(plot_w * (i / tick_steps))
        draw.line((x, bottom, x, bottom + 8), fill="#666", width=1)
        val = int((max_value * i) / tick_steps)
        draw.text((x - 20, bottom + 14), str(val), fill="#444", font=font)

    # rotate x labels not used (for readability in dense bar lists, keep horizontal)
    image.save(out_path, format="PNG", optimize=True)


def draw_heatmap(
    data: list[tuple[str, str, str, int]],
    out_path: Path,
    title: str,
    top_years: int = 12,
    top_layouts: int = 8,
) -> None:
    years = sorted({year for _, year, _, _ in data})[-top_years:]
    layouts = sorted({layout for _, _, layout, _ in data})[:top_layouts]

    # Rebuild matrix grouped by year/layout in data tuples.
    matrix: dict[tuple[str, str], int] = {(year, layout): 0 for year in years for layout in layouts}
    for _category, year, layout, count in data:
        if year in years and layout in layouts:
            matrix[(year, layout)] = matrix.get((year, layout), 0) + count

    w_col = 140
    h_row = 48
    left = 160
    top = 80
    width = left + len(layouts) * w_col + 40
    height = top + len(years) * h_row + 80
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    draw.text((20, 20), title, fill="#111", font=font)
    max_cell = max(matrix.values() or [1])

    for lx, layout in enumerate(layouts):
        x = left + lx * w_col
        draw.text((x + 10, 40), layout, fill="#111", font=font)

    for y, year in enumerate(years):
        y0 = top + y * h_row + 8
        draw.text((20, y0 + 12), year, fill="#111", font=font)
        for lx, layout in enumerate(layouts):
            count = matrix[(year, layout)]
            x0 = left + lx * w_col
            x1 = x0 + w_col - 12
            y1 = y0 + h_row - 8
            ratio = count / max_cell
            intensity = int(255 - (ratio * 170))
            fill = (255, intensity, intensity)
            draw.rectangle((x0, y0, x1, y1), fill=fill, outline="#333")
            draw.text((x0 + 8, y0 + 14), str(count), fill="#111", font=font)

    image.save(out_path, format="PNG", optimize=True)


def make_thumbnail(
    pdf_path: Path,
    out_path: Path,
    page_number: int = 0,
    max_size: tuple[int, int] = (640, 900),
    quality: int = 95,
) -> str:
    try:
        if not pdf_path.exists():
            raise FileNotFoundError(f"missing {pdf_path}")
        doc = fitz.open(pdf_path)
        if len(doc) == 0:
            raise RuntimeError(f"empty pdf {pdf_path}")
        page_idx = min(max(0, page_number), len(doc) - 1)
        page = doc[page_idx]
        target_width, target_height = max_size
        width = max(1, page.rect.width)
        height = max(1, page.rect.height)
        scale = min(target_width / width, target_height / height)
        matrix = fitz.Matrix(scale * 2.2, scale * 2.2)
        pix = page.get_pixmap(matrix=matrix, alpha=False)
        image = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        image.thumbnail(max_size)
        image.save(out_path, format="JPEG", quality=quality, optimize=True, progressive=True)
        return "ok"
    except Exception as exc:  # pragma: no cover - defensive branch
        base = Image.new("RGB", max_size, "#f3f3f3")
        draw = ImageDraw.Draw(base)
        font = ImageFont.load_default()
        draw.text((20, max_size[1] // 2 - 24), "thumbnail unavailable", fill="#555", font=font)
        draw.text((20, max_size[1] // 2), str(exc)[:90], fill="#777", font=font)
        base.save(
            out_path,
            format="JPEG",
            quality=max(10, min(quality, 100)),
            optimize=True,
            progressive=True,
        )
        return "error"


def safe(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def generate_report(args: argparse.Namespace) -> None:
    clusters_path = Path(args.clusters)
    profiles_path = Path(args.profiles)
    output_root = Path(args.output)

    clusters_root = output_root / "assets"
    images_root = clusters_root / "images"
    data_root = clusters_root / "data"
    clusters_root.mkdir(parents=True, exist_ok=True)
    images_root.mkdir(parents=True, exist_ok=True)
    data_root.mkdir(parents=True, exist_ok=True)

    profiles_by_key: dict[str, dict[str, Any]] = {}
    for profile in iter_jsonl(profiles_path):
        key = safe(profile.get("pdf_key"))
        if key:
            profiles_by_key[key] = profile

    strategy_cluster_counts: Counter[str] = Counter()
    strategy_pdf_counts: Counter[str] = Counter()
    category_year_layout_counts: Counter[tuple[str, str, str]] = Counter()
    year_counts: Counter[str] = Counter()
    layout_counts: Counter[str] = Counter()
    theme_counts: Counter[str] = Counter()
    category_counts: Counter[str] = Counter()
    clusters: list[dict[str, Any]] = []

    for row in iter_jsonl(clusters_path):
        strategy = safe(row.get("assigned_strategy") or "unknown")
        strategy_cluster_counts[strategy] += 1
        strategy_pdf_counts[strategy] += int(row.get("count", 0))
        feature = (
            row.get("feature_signature") if isinstance(row.get("feature_signature"), dict) else {}
        )
        category = safe(feature.get("category", row.get("dominant_category", "미지정")))
        year = safe(feature.get("year", row.get("year", "미지정")))
        layout = safe(feature.get("layout_class", "미지정"))
        theme = safe(feature.get("theme", row.get("theme", "")) or "미지정")
        count = int(row.get("count", 0))
        sample_keys = row.get("sample_pdf_keys") or []

        category_year_layout_counts[(category, year, layout)] += count
        year_counts[year] += 1
        layout_counts[layout] += 1
        theme_counts[theme] += 1
        category_counts[category] += 1

        clusters.append(
            {
                "cluster_id": row.get("cluster_id"),
                "dominant_category": category,
                "year": year,
                "layout_class": layout,
                "theme": theme,
                "strategy": strategy,
                "count": count,
                "sample_pdf_keys": sample_keys,
                "confidence": row.get("confidence"),
                "reasons": row.get("reasons"),
                "feature_signature": feature,
            }
        )

    clusters.sort(key=lambda item: item["count"], reverse=True)
    top_clusters = clusters[: args.top_clusters]

    # Generate thumbnails for representative samples.
    cluster_cards = []
    for _rank, cluster in enumerate(top_clusters, 1):
        rep_key = safe((cluster["sample_pdf_keys"] or [None])[0])
        if rep_key and rep_key in profiles_by_key:
            profile = profiles_by_key[rep_key]
            pdf_path = Path(profile.get("pdf_abs_path") or profile.get("pdf_path") or "")
            result = make_thumbnail(
                pdf_path,
                images_root / f"{cluster['cluster_id']}.jpg",
                max_size=(args.thumbnail_max_width, args.thumbnail_max_height),
                quality=args.thumbnail_jpeg_quality,
            )
        else:
            make_thumbnail(
                Path("/non-existent.pdf"),
                images_root / f"{cluster['cluster_id']}.jpg",
                max_size=(args.thumbnail_max_width, args.thumbnail_max_height),
                quality=args.thumbnail_jpeg_quality,
            )
            result = "missing"
        cluster_cards.append(
            {
                **cluster,
                "thumbnail": f"./images/{cluster['cluster_id']}.jpg",
                "thumbnail_status": result,
                "sample_pdf_key": rep_key,
            }
        )

    strategy_counts = {k: int(v) for k, v in strategy_cluster_counts.items() if k}
    year_counts_by_name = dict(sorted(year_counts.items(), key=lambda kv: kv[0]))
    theme_counts_by_name = dict(sorted(theme_counts.items(), key=lambda kv: kv[1], reverse=True))
    category_counts_by_name = dict(
        sorted(category_counts.items(), key=lambda kv: kv[1], reverse=True)
    )
    layout_counts_by_name = dict(sorted(layout_counts.items(), key=lambda kv: kv[1], reverse=True))

    cat_year_layout_top = sorted(
        (
            (cat, year, layout, count)
            for (cat, year, layout), count in category_year_layout_counts.items()
        ),
        key=lambda item: item[3],
        reverse=True,
    )
    cat_year_layout_top = cat_year_layout_top[: args.top_category_year_layout]

    heatmap_data = [(cat, year, layout, count) for cat, year, layout, count in cat_year_layout_top]
    draw_heatmap(
        heatmap_data,
        images_root / "category_year_layout_heatmap.png",
        title="관보타입×연도×레이아웃 상위 조합",
        top_years=12,
        top_layouts=min(8, len({layout for _, _, layout, _ in heatmap_data} or {1})),
    )
    draw_bar_chart("전략별 클러스터 수", strategy_counts, images_root / "strategy_distribution.png")
    draw_bar_chart("연도별 클러스터 수", year_counts_by_name, images_root / "year_distribution.png")

    report = {
        "status": "ok",
        "clusters_count": len(clusters),
        "profiles_count": len(profiles_by_key),
        "strategy_cluster_counts": strategy_counts,
        "strategy_pdf_counts": {k: int(v) for k, v in strategy_pdf_counts.items()},
        "year_counts": year_counts_by_name,
        "layout_counts": layout_counts_by_name,
        "theme_counts": theme_counts_by_name,
        "category_counts": category_counts_by_name,
        "category_year_layout_top": [
            {"category": category, "year": year, "layout_class": layout, "clusters": count}
            for category, year, layout, count in cat_year_layout_top
        ],
        "top_clusters": cluster_cards,
        "cluster_source": str(clusters_path),
        "profile_source": str(profiles_path),
        "output": str(output_root),
    }

    (data_root / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    cluster_cards_html = "".join(
        f"""
          <article class=\"cluster-card\">
            <img src=\"{item["thumbnail"]}\" alt=\"{item["cluster_id"]}\" />
            <div class=\"cluster-meta\">
              <h3>{item["dominant_category"]} ({item["year"]})</h3>
              <p>클러스터: <strong>{item["cluster_id"]}</strong></p>
              <p>레이아웃: <strong>{item["layout_class"]}</strong> | 전략: <strong>{item["strategy"]}</strong></p>
              <p>샘플 수: <strong>{item["count"]}</strong>개 | theme: {item["theme"]}</p>
              <p>표본 PDF: <code>{item["sample_pdf_key"] or "-"}</code></p>
            </div>
          </article>
        """
        for item in cluster_cards
    )

    category_year_layout_rows = "".join(
        f"<tr><td>{safe(category)}</td><td>{safe(year)}</td><td>{safe(layout)}</td><td>{count}</td></tr>"
        for category, year, layout, count in cat_year_layout_top
    )

    html = f"""<!doctype html>
<html lang=\"ko\">
  <head>
    <meta charset=\"utf-8\" />
    <title>클러스터링 결과 검토 리포트</title>
    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
    <link rel=\"stylesheet\" href=\"./styles.css\" />
  </head>
  <body>
    <main class=\"container\">
      <header>
        <h1>클러스터링 결과 검토 리포트</h1>
        <p>총 클러스터: {len(clusters):,}개, 전체 프로필: {len(profiles_by_key):,}개</p>
      </header>
      <section class=\"cards\">
        <article class=\"card\">
          <h2>전략 분포</h2>
          <img src=\"./assets/images/strategy_distribution.png\" alt=\"strategy_distribution\" />
        </article>
        <article class=\"card\">
          <h2>연도 분포</h2>
          <img src=\"./assets/images/year_distribution.png\" alt=\"year_distribution\" />
        </article>
        <article class=\"card\">
          <h2>연도×레이아웃 Heatmap(상위)</h2>
          <img src=\"./assets/images/category_year_layout_heatmap.png\" alt=\"category_year_layout_heatmap\" />
        </article>
      </section>
      <section class=\"card\">
        <h2>대표 클러스터</h2>
        <div class=\"cluster-grid\">
          {cluster_cards_html}
        </div>
      </section>
      <section class=\"card\">
        <h2>관보 타입×연도×레이아웃 상위 분포</h2>
        <table>
          <thead>
            <tr>
              <th>관보 타입</th>
              <th>연도</th>
              <th>레이아웃</th>
              <th>클러스터 수</th>
            </tr>
          </thead>
          <tbody>
            {category_year_layout_rows}
          </tbody>
        </table>
      </section>
    </main>
  </body>
</html>
"""

    css = """
    :root {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, "Apple SD Gothic Neo", "Malgun Gothic", sans-serif;
      color: #111827;
      background: #f4f6fb;
    }
    body { margin: 0; padding: 32px; }
    .container { max-width: 1280px; margin: 0 auto; }
    .cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin-bottom: 20px; }
    .card, .cluster-card {
      background: #fff;
      border: 1px solid #dbe0eb;
      border-radius: 12px;
      padding: 16px;
    }
    h1, h2, h3 { margin: 0.2rem 0 0.8rem; }
    img { max-width: 100%; border-radius: 8px; border: 1px solid #ddd; }
    table { width: 100%; border-collapse: collapse; font-size: 14px; }
    th, td { border: 1px solid #dce0e8; padding: 8px; text-align: left; }
    th { background: #f0f4ff; }
    .cluster-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 16px; }
    .cluster-card img { width: 100%; aspect-ratio: 3 / 4; object-fit: cover; margin-bottom: 10px; background: #f8f9fc; }
    .cluster-meta p { margin: 0.25rem 0; font-size: 14px; line-height: 1.5; }
    code { background: #f4f4f6; padding: 2px 6px; border-radius: 6px; font-size: 12px; }
    .card:last-child { overflow-x: auto; }
    """

    (clusters_root / "index.html").write_text(html, encoding="utf-8")
    (clusters_root / "styles.css").write_text(css, encoding="utf-8")
    print(f"status: ok -> {output_root}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate static reviewer HTML for cluster statistics."
    )
    parser.add_argument("--clusters", required=True, help="cluster_manifest.jsonl path")
    parser.add_argument("--profiles", required=True, help="pdf-profile manifest.jsonl path")
    parser.add_argument(
        "--output", default="/tmp/gwanbo-ocr-cluster-review", help="output directory"
    )
    parser.add_argument(
        "--top-clusters", type=int, default=80, help="number of clusters to render in preview"
    )
    parser.add_argument(
        "--top-category-year-layout",
        type=int,
        default=120,
        help="number of category/year/layout rows to list in table",
    )
    parser.add_argument("--thumbnail-max-width", type=int, default=960, help="max thumbnail width")
    parser.add_argument(
        "--thumbnail-max-height", type=int, default=1280, help="max thumbnail height"
    )
    parser.add_argument(
        "--thumbnail-jpeg-quality",
        type=int,
        default=98,
        help="jpeg quality for thumbnails",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generate_report(args)


if __name__ == "__main__":
    main()
