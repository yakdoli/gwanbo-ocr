# Architecture Refactor Design

**Date:** 2026-05-12  
**Scope:** Full architecture cleanup — bench/ package, peers/ package, CLI thinning, test restructure

> **Status (2026-05-13):** Historical design record. The main architecture refactor described here has already been implemented in recent commits; retained legacy module/test references below are context, not current instructions.

---

## Context

The gwanbo-ocr pipeline has grown through several feature additions (strategy pipeline, routing-aware benchmarking, peer review, cluster profiling). As a result:

- `bench.py` (598 lines) mixes benchmark execution, throughput analytics, reporting, preflight validation, and model resolution.
- `cli.py` (531 lines) contains a 167-line `strategy_pipeline()` function that directly orchestrates a 7-stage pipeline — domain logic living inside the CLI layer.
- `peer_review.py` (856 lines) bundles five distinct extraction methods together with comparison and scoring logic.

**Goal:** Assign single responsibilities to each module, enable each unit to be understood and tested independently, and align the test structure with the new module layout.

---

## New Module Structure

```
src/gwanbo_ocr/
├── cli.py                    # Thin Typer adapter only (~330 lines)
├── bench/
│   ├── __init__.py           # Re-exports: RunRecord, run_benchmark, score_benchmark, summarize_throughput, format_throughput_report
│   ├── run.py                # RunRecord, run_benchmark, _run_benchmark_task, resolve_runner_model
│   ├── score.py              # score_benchmark
│   └── report.py             # summarize_throughput, format_throughput_report
├── strategy.py               # Existing + run_pipeline() (absorbs cli.strategy_pipeline logic)
├── peers/
│   ├── __init__.py           # Comparison/scoring public API: run_peer_review, score_peer_results, etc.
│   ├── native.py             # extract_native_text and helpers
│   ├── pdfplumber.py         # extract_pdfplumber and helpers
│   ├── markitdown.py         # extract_markitdown and helpers
│   ├── paddle.py             # extract_paddle and helpers
│   └── vlm.py                # extract_vlm and helpers
├── runners/
│   ├── preflight.py          # preflight_openai_endpoint, _read_json_response, _served_model_ids
│   └── ... (existing: base, protocol, vllm, paddle, paddleocr, vllm_chat)
└── ... (manifest, metrics, sampling, prompts, render, docker_vllm, pdf/ — unchanged)
```

---

## Responsibility Assignments

### `bench/run.py`
- `RunRecord` — frozen dataclass for per-task results
- `run_benchmark(suite, runner, run_dir, ...)` — concurrent task orchestration with preflight
- `_run_benchmark_task(...)` — single-task execution with strategy-based routing (Paddle vs vLLM)
- `resolve_runner_model(runner_alias)` — reads configs/models.yaml, maps alias → model ID

### `bench/score.py`
- `score_benchmark(run_dir, output_dir, ...)` — loads RunRecords, evaluates against reference, computes CER/WER/F1

### `bench/report.py`
- `summarize_throughput(run_dir)` — aggregates latencies, compute rates, route metrics, status counts
- `format_throughput_report(summary)` — renders Markdown throughput report

### `bench/__init__.py`
- Declares the package's canonical public API via `from .run import ...`, `from .score import ...`, `from .report import ...`.
- All internal callers (cli.py, tests) are updated to import from the specific submodule (`bench.run`, `bench.score`, `bench.report`) rather than relying on `__init__.py` re-exports. `__init__.py` is the public package surface, not a backward-compat shim.

### `runners/preflight.py`
- `preflight_openai_endpoint(base_url, model_id, timeout_s)` — checks reachability + model availability
- `_read_json_response(resp)`, `_served_model_ids(base_url, timeout_s)` — internal helpers

### `strategy.py` (extended)
- Existing: `cluster_pdfs`, `evaluate_strategy`, `StrategyResult`, etc.
- **Added:** `run_pipeline(args...)` — the 7-stage orchestration logic currently in `cli.strategy_pipeline()`. CLI's `strategy_pipeline` command becomes a thin call to `strategy.run_pipeline(...)`.

### `peers/__init__.py`
- Public API: `run_peer_review`, `score_peer_results`, `format_peer_report`
- Comparison and aggregation logic that spans methods

### `peers/{native,pdfplumber,markitdown,paddle,vlm}.py`
- Each file owns exactly one extraction method and its private helpers
- Existing function signatures are preserved exactly — no interface changes during this refactor

### `cli.py`
- Typer command definitions and parameter parsing only
- Every command delegates immediately to the appropriate domain function
- `strategy_pipeline` command: parse params → call `strategy.run_pipeline(...)` → exit

---

## Data Flow (unchanged)

```
manifest build → pdf classify → pdf layout → pdf render
                                                    ↓
                                         bench run (bench/run.py)
                                                    ↓
                                         bench score (bench/score.py)
                                                    ↓
                                         report (bench/report.py)

strategy cluster → strategy evaluate → strategy pipeline (strategy.run_pipeline)
peer run → peer score  (peers/)
```

---

## Test Structure

```
tests/
├── bench/
│   ├── test_run.py           # run_benchmark, _run_benchmark_task, RunRecord
│   ├── test_score.py         # score_benchmark
│   ├── test_report.py        # summarize_throughput, format_throughput_report
│   └── test_vllm_runner.py   # existing, keep as-is
├── peers/
│   ├── test_native.py
│   ├── test_pdfplumber.py
│   ├── test_markitdown.py
│   ├── test_paddle.py
│   ├── test_vlm.py
│   └── test_compare.py       # comparison/scoring in peers/__init__.py
├── runners/
│   └── test_preflight.py     # preflight tests (split from test_vllm_runner.py)
├── test_strategy.py          # updated: add run_pipeline tests
├── test_cli.py               # updated: import paths only
└── ... (test_manifest, test_render, test_pdf_*, test_docker_vllm — unchanged)
```

**No backward-compat re-exports or shims.** All import paths updated directly to new locations.

---

## Migration Rules

1. Move code — no logic changes during migration. Refactor and logic fixes are separate commits.
2. `bench/__init__.py` re-exports the public API so `from gwanbo_ocr.bench import run_benchmark` still works — but this is a real re-export, not a deprecated shim.
3. Delete `peer_review.py` once `peers/` is complete and all tests pass.
4. Delete the old `bench.py` once `bench/` package is complete and all tests pass.
5. Each migration step: move → update imports → run `pytest` + `ruff check` + `mypy`.

---

## Build Sequence

1. `runners/preflight.py` — extract from `bench.py`, update bench.py import, run tests
2. `bench/` package — create package, split run/score/report, update `bench/__init__.py`, update all callers, delete `bench.py`
3. `strategy.py` — add `run_pipeline()`, thin out `cli.strategy_pipeline`
4. `peers/` package — create package, split by method, update `peers/__init__.py`, update callers, delete `peer_review.py`
5. Test restructure — split `test_bench.py` → `tests/bench/`, split `test_peer_review.py` → `tests/peers/`, add `tests/runners/test_preflight.py`
6. Final verification — `pytest`, `ruff check src tests`, `mypy src`

---

## Verification

After each step:
```bash
pytest
ruff check src tests
mypy src
```

Full suite after all steps:
```bash
pytest --tb=short -q
ruff format --check src tests
mypy src
```

Acceptance: all existing tests pass, no ruff errors, no mypy errors. No behavior changes.
