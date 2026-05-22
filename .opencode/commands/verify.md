---
description: Run full verification: ruff lint → format → pyrefly → pytest
agent: test
---

Run the complete verification pipeline:

```bash
ruff check src tests scripts && \
ruff format --check src tests scripts && \
.venv/bin/pyrefly check --summary=none && \
.venv/bin/pytest -ra --tb=short
```

Report all failures. If everything passes, confirm with a summary.
