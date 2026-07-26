# gwanbo-ocr

Focused tools for classifying PDF text availability and producing auditable OCR text and metadata corrections.

## Requirements and installation

Python 3.12 or newer is required.

```bash
python -m pip install -e ".[pdf,dev]"
```

## PDF classification

Classify PDFs listed in a JSONL manifest by their observable page content:

```bash
gwanbo-ocr pdf classify \
  --input manifest.jsonl \
  --output classification \
  --max-pages 3 \
  --peti-root corpus \
  --timeout-seconds 30 \
  --max-file-bytes 268435456
```

`--input` and `--output` are required. `--peti-root` supplies the trusted root for relative PDF paths; the other options use the defaults shown above.

## OCR correction

Normalize OCR text from a JSONL file while preserving the raw text and writing correction audit data:

```bash
gwanbo-ocr pdf correct \
  --input ocr.jsonl \
  --output corrected.jsonl
```

Optional low-confidence span review uses an OpenAI-compatible VLM endpoint:

```bash
gwanbo-ocr pdf correct \
  --input ocr.jsonl \
  --output corrected.jsonl \
  --vlm-base-url http://localhost:8000/v1 \
  --vlm-model vision-model \
  --vlm-crop-root crops \
  --vlm-timeout 30
```

`--vlm-base-url` and `--vlm-model` must be supplied together, and enabling VLM review also requires `--vlm-crop-root`. Remote endpoints require HTTPS; plain HTTP is accepted only for localhost. If authentication is needed, set `OPENAI_API_KEY` in the process environment.

## Development checks

```bash
.venv/bin/ruff check --no-cache src tests
.venv/bin/ruff format --check --no-cache src tests
.venv/bin/pyrefly check --summary=none
.venv/bin/pytest -ra --tb=short -p no:cacheprovider
```

## Current scope

The package currently supports only focused PDF text classification and auditable OCR text and metadata correction. It does not provide a general document-processing pipeline or model-serving runtime.
