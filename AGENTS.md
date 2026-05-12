# AGENTS.md

Compact guidance for OpenCode sessions working in this repo.

## Hard Constraints

- `/root/peti/artifacts` is read-only input. Never write, mutate, or delete anything under it.
- Pipeline outputs belong under `runs/<run_id>/` or `/tmp/gwanbo-ocr-*`; `runs/` is gitignored.
- Do not install vLLM, torch, transformers, ROCm/CUDA runtimes in `.venv`; vLLM runs as an external Docker/OpenAI-compatible service.
- Python `>=3.12` is enforced by `pyproject.toml`.
- Ignore `.claude/worktrees/`, `.venv/`, caches, and `.opencode/node_modules/` when searching; they contain duplicate snapshots or dependencies.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python -m pip install -e ".[pdf,qwen,dev]"   # PDF + VLM workflows
python -m pip install -e ".[paddleocr]"      # PaddleOCR workflows
```

If vLLM/Torch residue was accidentally installed into `.venv`, dry-run before applying cleanup:

```bash
scripts/clean_venv_vllm_residue.sh --venv .venv
scripts/clean_venv_vllm_residue.sh --venv .venv --apply
```

## Verification

There is no Makefile, pre-commit config, or GitHub Actions workflow; use `pyproject.toml` and `.opencode/commands/verify.md` as source of truth.

Run in this order:

```bash
ruff check src tests
ruff format --check src tests
.venv/bin/mypy src
.venv/bin/pytest -ra --tb=short
```

Focused checks:

```bash
.venv/bin/pytest tests/test_cli.py -v
.venv/bin/pytest tests/pdf/test_pdf_session_b.py -v
.venv/bin/pytest tests/bench/test_vllm_runner.py -k "test_vllm_runner_rejects_schema_echo" -v
```

Ruff config: line length 100, py312, rules `E,F,I,UP,B`; project ignores `B008` and `E501`; tests ignore `E402` for local `sys.path` insertion.

## Test Conventions

- Tests insert `src/` into `sys.path`; editable install is not required for imports.
- CLI tests use `typer.testing.CliRunner` with `from gwanbo_ocr.cli import app`.
- Some PDF/render tests skip optional dependency paths when PyMuPDF/pdfplumber/MarkItDown are missing.
- A real `/root/peti` smoke test in `tests/test_cli.py` skips if `/root/peti` is absent.

## Entrypoints

- Console script: `gwanbo-ocr = gwanbo_ocr.cli:app`.
- CLI groups: `manifest`, `pdf`, `bench`, `peer`, `strategy`.
- PDF commands include `validate`, `classify`, `layout`, `render`, and `profile`.
- Benchmark commands are `bench run` and `bench score`; peer review commands are `peer run` and `peer score`.

Common pipeline:

```bash
gwanbo-ocr manifest build --peti-root /root/peti --output runs/<run_id>/pdf_manifest.jsonl
gwanbo-ocr pdf classify --input runs/<run_id>/pdf_manifest.jsonl --output runs/<run_id>/classification --max-pages 3 --workers 8
gwanbo-ocr pdf layout --classification runs/<run_id>/classification/manifest.jsonl --output runs/<run_id>/layout --table-strategy auto
gwanbo-ocr pdf render --input runs/<run_id>/classification/manifest.jsonl --output runs/<run_id>/images --dpi 200 --max-long-edge 2400
gwanbo-ocr pdf profile --input runs/<run_id>/pdf_manifest.jsonl --output runs/<run_id>/profiles --max-pages 3 --workers 8 --sample-per-bucket 20
gwanbo-ocr strategy cluster --profiles runs/<run_id>/profiles/manifest.jsonl --output runs/<run_id>/clusters
gwanbo-ocr strategy evaluate --clusters runs/<run_id>/clusters/cluster_manifest.jsonl --output runs/<run_id>/strategy_eval
gwanbo-ocr bench run --suite runs/<run_id>/images/manifest.jsonl --runner qwen36_baseline --base-url http://127.0.0.1:8000/v1 --run-dir runs/<run_id>/bench/qwen36_baseline
gwanbo-ocr bench score --run runs/<run_id>/bench/qwen36_baseline --output runs/<run_id>/reports/qwen36_baseline
```

## Runtime And Models

- Model aliases and integration defaults live in `configs/models.yaml`.
- Default VLM alias: `qwen36_baseline` -> `Qwen/Qwen3.6-35B-A3B-FP8` at `http://127.0.0.1:8000/v1`.
- `src/gwanbo_ocr/runners/vllm.py` uses the `openai` package by default, not raw `httpx`; tests may inject a fake `client`.
- `src/gwanbo_ocr/runners/vllm_chat.py`, `src/gwanbo_ocr/runners/paddleocr.py`, and `src/gwanbo_ocr/pdf/classify.py` are compatibility shims.
- Docker defaults are split between `configs/models.yaml` and `src/gwanbo_ocr/docker_vllm.py`; container names may differ (`vllm-ocr-batch` vs `gwanbo-vllm`), so check the call site before managing containers.

## Module Map

- `src/gwanbo_ocr/manifest.py`: builds `/root/peti` manifests and refuses output under the peti root.
- `src/gwanbo_ocr/pdf/io.py`: shared JSON/JSONL atomic writes and `/root/peti` PDF path resolution.
- `src/gwanbo_ocr/pdf/classification.py`: PDF integrity/native-text classification sidecars.
- `src/gwanbo_ocr/pdf/layout.py`: text-PDF layout and table analysis.
- `src/gwanbo_ocr/pdf/profile.py`: lightweight `pdf-profile/v1` rows for clustering.
- `src/gwanbo_ocr/render.py`: PyMuPDF PDF-to-PNG rendering.
- `src/gwanbo_ocr/strategy.py`: deterministic layout clustering and strategy evaluation.
- `src/gwanbo_ocr/bench.py`: benchmark run and score orchestration.
- `src/gwanbo_ocr/prompts.py`: JSON-only OCR/VLM transcription prompt and schema-echo guard.
- `src/gwanbo_ocr/docker_vllm.py`: MI300X/ROCm Docker command builder.
