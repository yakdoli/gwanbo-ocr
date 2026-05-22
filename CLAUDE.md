# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Purpose

`gwanbo-ocr` is a PDF metadata extraction, classification, layout analysis, OCR/VLM benchmarking, and scoring pipeline for Korean government gazette (관보) PDFs stored in the read-only `/root/peti` project.

**Key constraints**:
- `/root/peti/artifacts` is read-only input. Never write to it. All outputs go under `runs/<run_id>/` inside this repo, or `/tmp/gwanbo-ocr-*` for smoke runs. `runs/` is gitignored.
- vLLM, torch, transformers, ROCm, PaddlePaddle, and PaddleOCR are **never** installed inside `.venv`. They run as external Docker services.
- Python `>=3.12` is enforced by `pyproject.toml`.
- Ignore `.claude/worktrees/`, `.venv/`, caches when searching.

## Environment Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
# Add extras as needed:
pip install -e ".[pdf]"          # PDF parsing (PyMuPDF, pdfplumber, pypdf, etc.)
pip install -e ".[qwen]"         # OpenAI-compatible VLM client
pip install -e ".[services]"     # FastAPI service wrappers
pip install -e ".[pdf,qwen,dev]"
```

## Common Commands

```bash
# Lint
ruff check src tests scripts
ruff format --check src tests scripts

# Type check
pyrefly check --summary=none

# All tests
pytest

# Focused test file(s)
pytest tests/test_manifest.py
pytest tests/pdf/test_pdf_session_b.py
pytest tests/bench/test_run.py tests/bench/test_score.py tests/bench/test_report.py -v

# Single test by name
pytest tests/bench/test_vllm_runner.py -k "test_vllm_runner_rejects_schema_echo"

# Clean vLLM residue from .venv (dry run first, then apply)
scripts/clean_venv_vllm_residue.sh --venv .venv
scripts/clean_venv_vllm_residue.sh --venv .venv --apply
```

There is no Makefile, pre-commit config, or GitHub Actions workflow. Ruff config: line length 100, py312, rules `E,F,I,UP,B`; project ignores `B008` and `E501`; tests ignore `E402`.

## CLI Pipeline (gwanbo-ocr)

Full pipeline in run order:

```bash
# 1. Build manifest from /root/peti artifacts
gwanbo-ocr manifest build --peti-root /root/peti --output runs/<run_id>/pdf_manifest.jsonl

# 2. Classify PDFs (text_pdf / image_or_unextractable_pdf / invalid_pdf / missing_pdf)
gwanbo-ocr pdf classify \
  --input runs/<run_id>/pdf_manifest.jsonl \
  --output runs/<run_id>/classification \
  --max-pages 3 --workers 8

# 3. Layout analysis (only text_pdf entries → table_heavy / body_text / multi_column_text)
gwanbo-ocr pdf layout \
  --classification runs/<run_id>/classification/manifest.jsonl \
  --output runs/<run_id>/layout \
  --table-strategy auto

# 4. Render PDF pages to PNG
gwanbo-ocr pdf render \
  --input runs/<run_id>/classification/manifest.jsonl \
  --output runs/<run_id>/images \
  --dpi 200 --max-long-edge 2400

# 5. Profile PDFs for clustering (lightweight pdf-profile/v1 rows)
gwanbo-ocr pdf profile \
  --input runs/<run_id>/pdf_manifest.jsonl \
  --output runs/<run_id>/profiles \
  --max-pages 3 --workers 8 --sample-per-bucket 20

# 6. Cluster profiles and assign parsing strategies
gwanbo-ocr strategy cluster \
  --profiles runs/<run_id>/profiles/manifest.jsonl \
  --output runs/<run_id>/clusters
gwanbo-ocr strategy evaluate \
  --clusters runs/<run_id>/clusters/cluster_manifest.jsonl \
  --output runs/<run_id>/strategy_eval

# 7. Benchmark run / score
gwanbo-ocr bench run \
  --suite runs/<run_id>/images/manifest.jsonl \
  --runner qwen36_baseline \
  --base-url http://127.0.0.1:8000/v1 \
  --run-dir runs/<run_id>/bench/qwen36_baseline

gwanbo-ocr bench score \
  --run runs/<run_id>/bench/qwen36_baseline \
  --output runs/<run_id>/reports/qwen36_baseline

# 8. Peer extraction review (multi-method comparison)
gwanbo-ocr peer run \
  --input runs/<run_id>/pdf_manifest.jsonl \
  --output runs/<run_id>/peer_review

gwanbo-ocr peer score --review runs/<run_id>/peer_review

# 9. MarkItDown conversion (single file or manifest)
gwanbo-ocr convert file <pdf> --output <out.md> --mode plain
gwanbo-ocr convert manifest --input runs/<run_id>/pdf_manifest.jsonl --output runs/<run_id>/markdown
```

## Architecture

### Pipeline Stages

1. **`manifest build`** — Walks `/root/peti/artifacts/{pety,searchThema}/` and emits JSONL `SampleRow` records with `sample_id`, `pdf_path`, `sha256`, `pages`, `text_extractable`, `layout_class`, `table_count`, etc. Merges sidecars from `metadata/`, `text_metadata/`, `layout_metadata/`.

2. **`pdf classify`** — Writes per-PDF classification sidecars (`schema_version`, `pdf_key`, `integrity`, `native_text`, `decision`).

3. **`pdf layout`** — Runs only on `text_pdf` entries; produces `document_class` via pdfplumber table/column analysis.

4. **`pdf render`** — PDF pages → PNG via PyMuPDF (200 DPI default, 2400 max-long-edge).

5. **`pdf profile` / `strategy cluster` / `strategy evaluate`** — Builds deterministic `pdf-profile/v1` rows, groups into layout clusters, assigns parsing strategies (`native_text_body`, `native_pdfplumber_table`, `ocr_paddle_simple`, `ocr_vlm_structured`, `peer_review_escalation`, `skip_invalid`).

6. **`bench run` / `bench score`** — Sends rendered pages to an OpenAI-compatible `/v1/chat/completions` endpoint; results are `RunRecord` objects; scores CER, WER, table cell F1, critical-token F1, and throughput.

7. **`peer run` / `peer score`** — Runs multiple extraction methods in parallel (native text, pdfplumber, MarkItDown, PaddleOCR, PaddleOCR-VL, VLM) and cross-compares results; `decide_extraction` picks `preferred_text_source` and flags `needs_ocr`.

8. **`convert`** — Converts PDFs to Markdown via MarkItDown (plain or ocr-llm mode); supports HTTP service backend (`GWANBO_MARKITDOWN_SERVICE_URL`).

### Key Source Files

| File | Role |
|------|------|
| `src/gwanbo_ocr/cli.py` | Typer CLI entrypoint; all subcommands |
| `src/gwanbo_ocr/manifest.py` | Manifest build; refuses writes under peti root |
| `src/gwanbo_ocr/pdf/io.py` | Shared JSON/JSONL atomic writes and `/root/peti` path resolution |
| `src/gwanbo_ocr/pdf/integrity.py` | PDF header, EOF, hash, page-count checks |
| `src/gwanbo_ocr/pdf/text.py` | Native text extraction metadata |
| `src/gwanbo_ocr/pdf/classification.py` | Per-PDF classification sidecar |
| `src/gwanbo_ocr/pdf/layout.py` | Layout/table analysis for text PDFs |
| `src/gwanbo_ocr/pdf/profile.py` | Lightweight `pdf-profile/v1` rows for clustering |
| `src/gwanbo_ocr/render.py` | PDF → PNG rendering via PyMuPDF |
| `src/gwanbo_ocr/strategy.py` | Deterministic layout clustering and strategy evaluation |
| `src/gwanbo_ocr/prompts.py` | OCR/VLM transcription prompt (JSON-only, anti-schema-echo) |
| `src/gwanbo_ocr/runners/vllm.py` | `VllmChatRunner`: builds chat payload, attaches image as data URL |
| `src/gwanbo_ocr/runners/preflight.py` | OpenAI-compatible endpoint/model preflight validation |
| `src/gwanbo_ocr/bench/run.py` | Benchmark run orchestration and runner/model resolution |
| `src/gwanbo_ocr/bench/score.py` | Benchmark scoring |
| `src/gwanbo_ocr/bench/report.py` | Throughput report helpers |
| `src/gwanbo_ocr/peers/` | Peer extraction orchestration; `METHOD_PREFERENCE` ranking |
| `src/gwanbo_ocr/conversion.py` | MarkItDown PDF-to-Markdown conversion (plain / ocr-llm modes) |
| `src/gwanbo_ocr/services.py` | FastAPI service HTTP clients (MarkItDown, PaddleOCR) |
| `src/gwanbo_ocr/metrics.py` | CER/WER/token/table scoring helpers |
| `src/gwanbo_ocr/sampling.py` | Deterministic stratified sampling (`DEFAULT_SAMPLE_SEED`) |
| `src/gwanbo_ocr/docker_vllm.py` | MI300X/ROCm Docker launch config |
| `configs/models.yaml` | Model aliases, Docker knobs, PaddleOCR settings |

**Compatibility shims** (do not extend): `runners/vllm_chat.py`, `runners/paddleocr.py`, `pdf/classify.py` — these re-export from canonical modules.

### `/root/peti` Artifact Layout (read-only)

```
artifacts/
├── searchThema/
│   ├── metadata/           # per-item JSON (id, theme, date, category, pdf.*)
│   ├── text_metadata/      # pages, text_extractable, total_chars
│   ├── layout_metadata/    # document_class, table metrics
│   ├── pdfs/               # YYYY/YYYYMMDD/<id>.pdf
│   └── issue_pdfs/         # fallback PDFs
└── pety/
    ├── metadata/
    ├── text_metadata/
    └── pdfs/
```

### vLLM Runtime (MI300X / ROCm Docker)

The default baseline runs inside `vllm/vllm-openai-rocm:latest` on an AMD MI300X with 192 GB HBM, exposing an OpenAI-compatible endpoint at `http://127.0.0.1:8000/v1`.

Default VLM alias: `qwen36_baseline` → `Qwen/Qwen3.6-35B-A3B-FP8`. All model aliases resolve through `configs/models.yaml`.

Docker container names differ by call site (`vllm-ocr-batch` vs `gwanbo-vllm`) — check the relevant call site before managing containers.

Docker launch flags: `--network=host`, `/dev/kfd`, `/dev/dri`, `--ipc=host`, `--group-add=video`, `--enforce-eager`, `--disable-custom-all-reduce`.

### OCR Result Schema

Each runner result includes: `status`, `text`, `tables`, `blocks`, `raw_response`, `latency_ms`, `usage`, `model_id`, `prompt_version`, `image_sha256`, `error`.

### Test Conventions

- CLI tests use `typer.testing.CliRunner` with `from gwanbo_ocr.cli import app`.
- Tests insert `src/` into `sys.path`; editable install is not required.
- PDF/render tests skip when optional deps (PyMuPDF, pdfplumber, MarkItDown) are absent.
- A real `/root/peti` smoke test in `tests/test_cli.py` skips if `/root/peti` is absent.

### Acceptance Targets

- Gold suite completion: 100%
- Timeout/error rate: ≤ 1%
- Clean text median normalized CER: ≤ 1%
- Scanned median CER: ≤ 6%
- Critical-token F1: ≥ 0.97
- Table cell F1: ≥ 0.85
