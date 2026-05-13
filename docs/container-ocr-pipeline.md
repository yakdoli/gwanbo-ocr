# Container OCR Pipeline

This project keeps heavyweight OCR/VLM runtimes outside the host `.venv`.
`gwanbo-ocr` orchestrates work from the app container or host CLI, while
PaddleOCR, PaddleOCR-VL, MarkItDown OCR, and vLLM run as separate services.

## Services

`docker-compose.ocr.yml` defines the full runtime:

- `gwanbo-app`: CLI orchestrator. Mounts `/root/peti` read-only and `runs/` read-write.
- `vllm-qwen`: OpenAI-compatible Qwen VLM at `http://127.0.0.1:8000/v1`.
- `paddleocr-vl-vllm`: OpenAI-compatible PaddleOCR-VL model server at `http://127.0.0.1:8118/v1`.
- `markitdown-ocr-api`: MarkItDown + `markitdown-ocr` service at `http://127.0.0.1:8081`.
- `paddleocr-api`: PaddleOCR/PaddleOCR-VL service at `http://127.0.0.1:8082`.

`/root/peti/artifacts` is input only. Write pipeline outputs under `runs/<run_id>/`.

## Start Services

```bash
docker compose -f docker-compose.ocr.yml up -d markitdown-ocr-api paddleocr-api
docker compose -f docker-compose.ocr.yml --profile vlm --profile paddle-vl up -d vllm-qwen paddleocr-vl-vllm
```

Health checks:

```bash
curl -fsS http://127.0.0.1:8081/health
curl -fsS http://127.0.0.1:8082/health
curl -fsS http://127.0.0.1:8000/v1/models
curl -fsS http://127.0.0.1:8118/v1/models
```

## Convert

Plain MarkItDown:

```bash
gwanbo-ocr convert file /path/to/doc.pdf --output runs/demo/markdown --mode plain
```

OCR+LLM through the service:

```bash
gwanbo-ocr convert file /path/to/doc.pdf \
  --output runs/demo/markdown \
  --mode ocr-llm \
  --service-url http://127.0.0.1:8081 \
  --llm-base-url http://127.0.0.1:8000/v1 \
  --llm-model Qwen/Qwen3.6-35B-A3B-FP8
```

## Peer Review

```bash
gwanbo-ocr peer run \
  --manifest runs/demo/pdf_manifest.jsonl \
  --output runs/demo/peer_review \
  --markitdown-ocr-llm \
  --markitdown-service-url http://127.0.0.1:8081 \
  --markitdown-llm-base-url http://127.0.0.1:8000/v1 \
  --markitdown-llm-model Qwen/Qwen3.6-35B-A3B-FP8 \
  --paddle \
  --paddle-service-url http://127.0.0.1:8082 \
  --paddle-vl \
  --paddle-vl-service-url http://127.0.0.1:8082 \
  --paddle-vl-server-url http://127.0.0.1:8118/v1 \
  --paddle-vl-model PaddlePaddle/PaddleOCR-VL-1.5
```

Peer runs write comparison artifacts under `runs/<run_id>/peer_review/samples/`
or, from `strategy pipeline`, under `runs/<run_id>/samples/`.

## Strategy Pipeline

```bash
gwanbo-ocr strategy pipeline \
  --manifest runs/demo/pdf_manifest.jsonl \
  --output runs/demo \
  --base-url http://127.0.0.1:8000/v1 \
  --markitdown-ocr-llm \
  --markitdown-service-url http://127.0.0.1:8081 \
  --markitdown-llm-base-url http://127.0.0.1:8000/v1 \
  --markitdown-llm-model Qwen/Qwen3.6-35B-A3B-FP8 \
  --paddle \
  --paddle-service-url http://127.0.0.1:8082 \
  --paddle-vl \
  --paddle-vl-service-url http://127.0.0.1:8082 \
  --paddle-vl-server-url http://127.0.0.1:8118/v1 \
  --paddle-vl-model PaddlePaddle/PaddleOCR-VL-1.5
```

Sampling artifacts include:

- `source.json`: original manifest metadata and rendered image paths.
- `peer_samples.json`: per-peer status and text sample.
- `peer_samples.md`: human-readable source/result comparison.
- `diff_summary.json`: pairwise similarity, warnings, and decision summary.
