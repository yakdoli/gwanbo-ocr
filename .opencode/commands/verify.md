---
description: Run full verification: ruff lint → format → mypy typecheck → pytest
agent: test
---

Run the complete verification pipeline:

```bash
ruff check src tests && \
ruff format --check src tests && \
.venv/bin/mypy src && \
.venv/bin/pytest -ra --tb=short
```

Report all failures. If everything passes, confirm with a summary.
