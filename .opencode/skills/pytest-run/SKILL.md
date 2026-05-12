---
name: pytest-run
description: Run the gwanbo-ocr test suite with proper venv activation
license: MIT
compatibility: opencode
---

## What I do
- Run `.venv/bin/pytest -ra --tb=short` for full test suite
- Run `.venv/bin/pytest tests/path/file.py -k "name" -v` for focused tests
- Report failures with tracebacks

## When to use me
After code changes that need test verification.

## Key tests
- `tests/bench/test_vllm_runner.py` — VLM runner + schema-echo rejection
- `tests/test_cli.py` — CLI smoke tests via Typer CliRunner
- `tests/pdf/test_pdf_session_b.py` — PDF pipeline tests
- `tests/test_manifest.py` — Manifest build
