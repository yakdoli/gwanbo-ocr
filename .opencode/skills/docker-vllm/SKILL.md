---
name: docker-vllm
description: Manage the vLLM Docker container for OCR/VLM inference
license: MIT
compatibility: opencode
metadata:
  infra: docker-rocm
  target: mi300x
---

## What I do
- Check if vLLM container is running
- View container logs
- Start/stop/restart the vLLM container
- Verify health at `http://127.0.0.1:8000/v1`

## When to use me
When working on OCR/VLM benchmarks or any code that calls the vLLM endpoint.

## Docker config
- Image: `vllm/vllm-openai-rocm:latest`
- Container: `gwanbo-vllm` (or `vllm-ocr-batch`)
- Exposes: `http://127.0.0.1:8000/v1`
- Docker MCP is disabled by default because it mounts the host Docker socket.
- Enable Docker MCP only for explicit container-management work, otherwise use bash with docker CLI.
- Never mount or mutate `/root/peti/artifacts`; it is read-only input.
