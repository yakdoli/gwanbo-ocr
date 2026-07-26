# PROJECT KNOWLEDGE BASE

## OVERVIEW

Python 3.12 package and Typer CLI for PDF text classification plus auditable OCR text/metadata correction. It is intentionally not a general OCR engine or model-serving runtime.

## STRUCTURE

```text
gwanbo-ocr/
├── src/gwanbo_ocr/cli.py          # thin Typer command surface
├── src/gwanbo_ocr/pdf/            # classification, integrity, correction, VLM review
├── tests/pdf/                     # security and PDF-path regressions
├── tests/test_cli.py              # CLI contract
├── tests/test_packaging.py        # optional-dependency contract
└── pyproject.toml                 # package, tooling, coverage floor
```

## WHERE TO LOOK

| Task | Location | Notes |
|------|----------|-------|
| Add/change CLI options | `src/gwanbo_ocr/cli.py` | Keep business logic outside Typer callbacks |
| Classify a document | `src/gwanbo_ocr/pdf/classification.py` | Observable content + integrity metadata |
| Correct JSONL rows | `src/gwanbo_ocr/pdf/correct.py` | Atomic output and audit fields |
| Change deterministic rules | `src/gwanbo_ocr/pdf/correction.py` | Preserve raw evidence |
| Change metadata gates | `src/gwanbo_ocr/pdf/metadata_correction.py` | Fail closed on ambiguous inputs |
| Change VLM review | `src/gwanbo_ocr/pdf/openai_review.py` | Security/resource limits are part of the API |

## CONVENTIONS

- CLI commands are `gwanbo-ocr pdf classify` and `gwanbo-ocr pdf correct`.
- Keep core correction importable without the optional `pdf` dependency group.
- Use frozen Pydantic models for manifest and correction result contracts.
- Preserve `raw_text`; write corrected text and audit metadata separately.
- Write JSONL outputs atomically and enforce path, size, timeout, and response limits before expensive work.
- Treat `--peti-root` and `--vlm-crop-root` as trusted-root boundaries, not convenience paths.

## ANTI-PATTERNS

- Do not silently replace source OCR evidence.
- Do not hash/read oversized files before size gates fail.
- Do not allow remote plain-HTTP VLM endpoints; only localhost may use HTTP.
- Do not enable VLM review without paired base URL/model and a crop root.
- Do not move PDF logic into CLI callbacks or weaken fail-closed tests.

## COMMANDS

```bash
python -m pip install -e ".[pdf,dev]"
.venv/bin/ruff check --no-cache src tests
.venv/bin/ruff format --check --no-cache src tests
.venv/bin/pyrefly check --summary=none
.venv/bin/pytest -ra --tb=short -p no:cacheprovider
```

CI also enforces at least 70% coverage.
