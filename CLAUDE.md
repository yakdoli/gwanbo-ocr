# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Purpose

`gwanbo-ocr` is a PDF metadata extraction, classification, layout analysis, OCR/VLM benchmarking, and scoring pipeline for Korean government gazette (관보) PDFs stored in the read-only `/root/peti` project.

**Key constraint**: `/root/peti/artifacts` is read-only input. Never write to it. All outputs go under `runs/<run_id>/` inside this repo, or `/tmp/gwanbo-ocr-*` for smoke runs.

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

vLLM, torch, transformers, ROCm, PaddlePaddle, and PaddleOCR are **never** installed inside `.venv`. They run as external Docker services.

## Common Commands

```bash
# Lint
ruff check src tests scripts
ruff format --check src tests scripts

# Type check
mypy src
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

## CLI Pipeline (gwanbo-ocr)

Core OCR benchmark flow:

```bash
gwanbo-ocr manifest build --peti-root /root/peti --output runs/<run_id>/pdf_manifest.jsonl

gwanbo-ocr pdf classify \
  --input runs/<run_id>/pdf_manifest.jsonl \
  --output runs/<run_id>/classification \
  --max-pages 3 --workers 8

gwanbo-ocr pdf layout \
  --classification runs/<run_id>/classification/manifest.jsonl \
  --output runs/<run_id>/layout \
  --table-strategy auto

gwanbo-ocr pdf render \
  --input runs/<run_id>/classification/manifest.jsonl \
  --output runs/<run_id>/images \
  --dpi 200 --max-long-edge 2400

gwanbo-ocr bench run \
  --suite runs/<run_id>/images/manifest.jsonl \
  --runner qwen36_baseline \
  --base-url http://127.0.0.1:8000/v1 \
  --run-dir runs/<run_id>/bench/qwen36_baseline

gwanbo-ocr bench score \
  --run runs/<run_id>/bench/qwen36_baseline \
  --output runs/<run_id>/reports/qwen36_baseline
```

## Architecture

### Pipeline Stages

1. **`manifest build`** — Walks `/root/peti/artifacts/{pety,searchThema}/` and emits a JSONL where each row is a `SampleRow` containing `sample_id`, `pdf_path`, `sha256`, `pages`, `text_extractable`, `layout_class`, `table_count`, etc. Merges metadata from three sidecar directories: `metadata/`, `text_metadata/`, `layout_metadata/`.

2. **`pdf classify`** — Reads the manifest; for each PDF writes a classification sidecar with fields `schema_version`, `pdf_key`, `integrity`, `native_text`, `decision`. Decision values: `text_pdf`, `image_or_unextractable_pdf`, `invalid_pdf`, `missing_pdf`.

3. **`pdf layout`** — Only runs on `text_pdf` entries. Analyses page metrics, line/column density, table candidates via pdfplumber, and produces `document_class`: `table_heavy`, `body_text`, or `multi_column_text`.

4. **`pdf render`** — Converts PDF pages to PNG using PyMuPDF (default 200 DPI, 2400 max-long-edge) for OCR/VLM input.

5. **`bench run` / `bench score`** — Sends rendered page images to an OpenAI-compatible `/v1/chat/completions` endpoint. Results are `RunRecord` objects; scoring computes CER, WER, table cell F1, critical-token F1, and throughput metrics.

6. **`pdf profile` / `strategy cluster` / `strategy evaluate`** — Combines manifest metadata with lightweight PDF features into `pdf-profile/v1`, builds deterministic layout clusters, and summarizes assigned parsing strategies for representative evaluation before large OCR runs.

### Key Source Files

| File | Role |
|------|------|
| `src/gwanbo_ocr/cli.py` | Typer CLI entrypoint; all subcommands |
| `src/gwanbo_ocr/manifest.py` | Manifest build; reads `/root/peti` read-only |
| `src/gwanbo_ocr/pdf/integrity.py` | PDF header, EOF, hash, page-count checks |
| `src/gwanbo_ocr/pdf/text.py` | Native text extraction metadata |
| `src/gwanbo_ocr/pdf/classification.py` | Per-PDF classification sidecar |
| `src/gwanbo_ocr/pdf/layout.py` | Layout/table analysis for text PDFs |
| `src/gwanbo_ocr/render.py` | PDF → PNG rendering via PyMuPDF |
| `src/gwanbo_ocr/prompts.py` | OCR/VLM transcription prompt (JSON-only, anti-schema-echo) |
| `src/gwanbo_ocr/runners/vllm.py` | `VllmChatRunner`: builds chat payload, attaches image as data URL |
| `src/gwanbo_ocr/runners/paddle.py` | PaddleOCR adapter |
| `src/gwanbo_ocr/runners/preflight.py` | OpenAI-compatible endpoint/model preflight validation |
| `src/gwanbo_ocr/bench/run.py` | Benchmark run orchestration and runner/model resolution |
| `src/gwanbo_ocr/bench/score.py` | Benchmark scoring |
| `src/gwanbo_ocr/bench/report.py` | Throughput report helpers |
| `src/gwanbo_ocr/peers/` | Peer extraction/review orchestration and adapters |
| `src/gwanbo_ocr/metrics.py` | CER/WER/token/table scoring helpers |
| `src/gwanbo_ocr/sampling.py` | Deterministic stratified sampling (`DEFAULT_SAMPLE_SEED`) |
| `src/gwanbo_ocr/docker_vllm.py` | MI300X/ROCm Docker launch config |
| `configs/models.yaml` | Model aliases, Docker knobs, PaddleOCR settings |

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

The default baseline runs inside `vllm/vllm-openai-rocm:latest` on an AMD MI300X with 192 GB HBM. The process exposes an OpenAI-compatible endpoint at `http://127.0.0.1:8000/v1`.

Docker launch flags used: `--network=host`, `/dev/kfd`, `/dev/dri`, `--ipc=host`, `--group-add=video`, `--enforce-eager`, `--disable-custom-all-reduce`.

All model aliases (`qwen36_baseline`, `qwen3_vl`, `bizonai_ocr`, `exaone45_33b`) share the same `VllmChatRunner` adapter and resolve through `configs/models.yaml`.

### OCR Result Schema

Each runner result includes: `status`, `text`, `tables`, `blocks`, `raw_response`, `latency_ms`, `usage`, `model_id`, `prompt_version`, `image_sha256`, `error`.

### Acceptance Targets

- Gold suite completion: 100%
- Timeout/error rate: ≤ 1%
- Clean text median normalized CER: ≤ 1%
- Scanned median CER: ≤ 6%
- Critical-token F1: ≥ 0.97
- Table cell F1: ≥ 0.85
