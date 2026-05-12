# gwanbo-ocr

PDF metadata extraction, classification, rendering, OCR/VLM benchmarking, and
scoring pipeline for `/root/peti` Gwanbo artifacts.

See [docs/project_overview.md](docs/project_overview.md) for the architecture,
data contracts, runtime policy, and benchmark strategy.

## Environment

Use a local virtual environment and install only the dependency groups needed
for the workflow you are running:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Optional extras:

```bash
python -m pip install -e ".[pdf]"          # PDF rendering helpers
python -m pip install -e ".[qwen]"         # OpenAI-compatible Qwen client helpers
python -m pip install -e ".[paddleocr]"    # PaddleOCR package
python -m pip install -e ".[pdf,qwen,dev]"
```

The project metadata intentionally keeps vLLM, torch, transformers, and local
model-server runtimes out of base and optional dependencies. vLLM is expected
to run as an external Docker service exposing an OpenAI-compatible endpoint,
for example `http://127.0.0.1:8000/v1`.

## Console Entry Point

The package exposes the CLI entry point:

```bash
gwanbo-ocr
```

The entry point is declared as `gwanbo_ocr.cli:app`.

Common flow:

```bash
gwanbo-ocr manifest build \
  --peti-root /root/peti \
  --output runs/<run_id>/pdf_manifest.jsonl

gwanbo-ocr pdf classify \
  --input runs/<run_id>/pdf_manifest.jsonl \
  --output runs/<run_id>/classification \
  --max-pages 3 \
  --workers 8

gwanbo-ocr pdf layout \
  --classification runs/<run_id>/classification/manifest.jsonl \
  --output runs/<run_id>/layout \
  --table-strategy auto

gwanbo-ocr pdf render \
  --input runs/<run_id>/classification/manifest.jsonl \
  --output runs/<run_id>/images \
  --dpi 200 \
  --max-long-edge 2400

gwanbo-ocr bench run \
  --suite runs/<run_id>/images/manifest.jsonl \
  --runner qwen36_baseline \
  --base-url http://127.0.0.1:8000/v1 \
  --run-dir runs/<run_id>/bench/qwen36_baseline

gwanbo-ocr bench score \
  --run runs/<run_id>/bench/qwen36_baseline \
  --output runs/<run_id>/reports/qwen36_baseline
```

## Model Configuration

Default integration settings live in `configs/models.yaml`. That file records
PDF rendering defaults, PaddleOCR settings, OpenAI-compatible model aliases,
and Docker launch settings for the MI300X/ROCm vLLM runtime. The default
baseline alias is `qwen36_baseline`, which resolves to
`Qwen/Qwen3.6-35B-A3B-FP8`.

## Cleaning vLLM Residue

Use the cleanup helper after experimenting with local vLLM/Torch/Transformers
installs in the project virtual environment:

```bash
scripts/clean_venv_vllm_residue.sh --venv .venv
scripts/clean_venv_vllm_residue.sh --venv .venv --apply
```

The first command is a dry run. The second command uninstalls matching packages
and removes leftover site-packages directories from the selected virtual
environment.
