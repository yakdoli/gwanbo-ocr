#!/usr/bin/env bash
# Adaptive OCR Benchmark — Sequential load/unload via lemonade CLI
# Usage: bash scripts/bench_adaptive.sh [LIMIT]
# Lemonade Max Models/Type=1, so we load one model at a time.

set -euo pipefail

LIMIT="${1:-3}"
BASE_URL="http://127.0.0.1:13305/api/v1"
SUITE="runs/bench_1540_4models/images/manifest.jsonl"
OUTPUT_BASE="runs/adaptive_bench"
VENV=".venv"

# Model definitions: lemonade_name | runner_alias | output_dir
declare -a MODELS=(
	"LightOnOCR-2-1B-vLLM|lightonocr2_1b_lemonade_vllm|lightonocr2_1b"
	"chandra-ocr-2-vLLM|chandra_ocr_2_lemonade_vllm|chandra_ocr2"
	"BizOnAI-OCR-vLLM|bizonai_ocr_lemonade_vllm|bizonai_ocr"
)

echo "========================================"
echo "Adaptive OCR Benchmark"
echo "Limit: ${LIMIT} images per model"
echo "Suite: ${SUITE}"
echo "========================================"

# Activate venv
source "${VENV}/bin/activate"

for ENTRY in "${MODELS[@]}"; do
	IFS='|' read -r LEMONADE_MODEL RUNNER_ALIAS OUTPUT_NAME <<<"${ENTRY}"
	RUN_DIR="${OUTPUT_BASE}/${OUTPUT_NAME}_1540_limit${LIMIT}"

	echo ""
	echo "========================================"
	echo "Loading: ${LEMONADE_MODEL}"
	echo "========================================"

	# Unload any existing model first
	lemonade unload 2>/dev/null || true
	sleep 2

	# Load model
	echo "Loading ${LEMONADE_MODEL} (this may take a few minutes)..."
	lemonade load "${LEMONADE_MODEL}" --ctx-size 16384
	echo "Load complete."

	# Wait for model to be ready
	sleep 5

	# Verify model is loaded
	echo "Verifying model is loaded..."
	curl -s --connect-timeout 10 "${BASE_URL}/models" | python -c "
import sys, json
data = json.load(sys.stdin)
models = [m['id'] for m in data.get('data', [])]
print(f'Active models: {models}')
if not any('${LEMONADE_MODEL}'.lower() in m.lower() for m in models):
    print(f'WARNING: ${LEMONADE_MODEL} not found in active models!')
    sys.exit(1)
"

	# Run benchmark
	echo ""
	echo "Running benchmark: ${RUNNER_ALIAS} → ${RUN_DIR}"
	python -c "
import json
from gwanbo_ocr.bench.run import run_benchmark

result = run_benchmark(
    suite='${SUITE}',
    runner_name='${RUNNER_ALIAS}',
    run_dir='${RUN_DIR}',
    base_url='${BASE_URL}',
    concurrency=1,
    limit=${LIMIT},
    preflight_vllm=False,
)
print(f'Benchmark status: {result.get(\"status\")}')
print(f'Tasks: {result.get(\"tasks\")}')
print(f'Results: {result.get(\"results\")}')
"

	# Print results
	echo ""
	echo "Results for ${LEMONADE_MODEL}:"
	python -c "
import json
results_file = '${RUN_DIR}/results.jsonl'
success = 0
fail = 0
total_duration = 0.0
total_chars = 0
with open(results_file) as f:
    for line in f:
        r = json.loads(line.strip())
        if not isinstance(r, dict):
            continue
        status = r.get('status', 'unknown')
        dur = r.get('duration_s', 0) or 0
        text_len = len(r.get('text', '') or '')
        total_duration += dur
        if status == 'ok':
            success += 1
            total_chars += text_len
            print(f'  ✅ p{r.get(\"page_number\",\"?\")} {dur:.1f}s {text_len} chars')
        else:
            fail += 1
            err = (r.get('error', '') or '')[:100]
            print(f'  ❌ p{r.get(\"page_number\",\"?\")} {dur:.1f}s ERROR: {err}')
print(f'  Summary: {success} ok, {fail} fail, {total_duration:.1f}s total, {total_chars} total chars')
if success > 0:
    print(f'  Avg: {total_duration/success:.1f}s/item, {total_chars//success if success else 0} chars/item')
"

	# Unload
	echo "Unloading ${LEMONADE_MODEL}..."
	lemonade unload 2>/dev/null || true
	sleep 3

done

echo ""
echo "========================================"
echo "All benchmarks complete!"
echo "========================================"
