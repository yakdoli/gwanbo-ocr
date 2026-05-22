---
description: Run pytest with proper venv and report failures
mode: subagent
model: openai/gpt-5.4-mini
permission:
  edit: deny
  bash:
    "*": allow
---

You run tests for the gwanbo-ocr project.

Use these commands in order:
1. `ruff check src tests scripts` — lint
2. `ruff format --check src tests scripts` — format check
3. `.venv/bin/pyrefly check --summary=none` — pyrefly typecheck
4. `.venv/bin/pytest -ra --tb=short` — all tests

For a single test:
```
.venv/bin/pytest tests/path/test_file.py -k "test_name" -v
```

For benchmark tests:
```
.venv/bin/pytest tests/bench/ -v
```

Report all failures with the full traceback. If no failures, confirm all passed.
