# gwanbo-ocr Project Overview

## 목적

`gwanbo-ocr`는 `/root/peti` 프로젝트가 보유한 관보 PDF 아티팩트를 읽기
전용 입력으로 사용해, PDF 메타데이터를 추출하고 OCR/VLM 파싱 전략을
벤치마크하기 위한 실행 도구입니다.

핵심 목표는 다음 세 가지입니다.

- PDF가 native text 추출 가능한 문서인지, 이미지 기반 스캔 문서인지
  분류한다.
- text 추출 가능한 PDF는 레이아웃/표 패턴을 분석해 직접 추출 전략을
  수립한다.
- text 추출이 어렵거나 이미지 기반인 PDF는 페이지 이미지로 렌더링한 뒤
  PaddleOCR 또는 OpenAI-compatible VLM endpoint를 통해 전사하고 모델
  조합을 비교한다.

## 저장소 역할

이 저장소는 구현과 실험 실행을 담당합니다.

- 구현 저장소: `/root/gwanbo-ocr`
- 읽기 전용 입력: `/root/peti/artifacts`
- 실행 산출물: `runs/<run_id>/`
- 임시 smoke 산출물: `/tmp/gwanbo-ocr-*`

`/root/peti`의 item JSON/PDF는 수정하지 않습니다. manifest, classification
sidecar, 렌더 이미지, benchmark result, report는 모두 이 저장소 밖의 원본을
참조하는 파생 산출물로 취급합니다.

## 아키텍처

파이프라인은 다섯 단계로 나뉩니다.

1. `manifest build`
   `/root/peti/artifacts` 아래의 item metadata와 PDF 경로를 수집해 JSONL
   manifest를 생성합니다. PDF path, sha256, page count, category/date/id 같은
   후속 처리 키를 한 줄 단위 sample row로 정규화합니다.

2. `pdf classify`
   PDF 무결성, page count, native text 추출량, 텍스트 밀도 등을 확인해
   `text_pdf`, `image_or_unextractable_pdf`, `invalid_pdf`, `missing_pdf` 등으로
   분류합니다.

3. `pdf layout`
   text 추출 가능한 PDF만 대상으로 page metrics, line density, column 추정,
   table 후보, form/table-heavy 여부를 분석합니다. 이 단계는 이미지 OCR을
   쓰기 전에 가능한 deterministic extraction 전략을 찾기 위한 단계입니다.

4. `pdf render`
   OCR/VLM 입력용으로 PDF page를 PNG로 렌더링합니다. 기본 구현은 PyMuPDF를
   사용하며 DPI, 최대 long edge, selected page/max page 제한을 지원합니다.

5. `bench run` / `bench score`
   렌더링된 image manifest를 입력으로 VLM/PaddleOCR runner를 실행하고,
   throughput 및 accuracy metric report를 생성합니다.

6. `pdf profile` / `strategy cluster` / `strategy evaluate`
   대용량 OCR 실행 전에 원본 metadata와 경량 PDF feature를 결합해
   `pdf-profile/v1`을 만들고, `연도 x PDF 종류 x 레이아웃` 단위의
   deterministic cluster를 생성합니다. 각 cluster는 기본 parsing strategy를
   배정받고, representative sample 평가를 위한 `strategy-eval/v1` row로
   요약됩니다.

## 런타임 정책

vLLM은 이 프로젝트의 Python dependency로 설치하지 않습니다. vLLM, torch,
transformers, ROCm/CUDA runtime 계열은 Docker endpoint가 책임집니다.

기본 baseline endpoint:

- Base URL: `http://127.0.0.1:8000/v1`
- Runner alias: `qwen36_baseline`
- Model id: `Qwen/Qwen3.6-35B-A3B-FP8`

MI300X/ROCm Docker 실행 설정은 `configs/models.yaml`과
`src/gwanbo_ocr/docker_vllm.py`에 기록되어 있습니다. 현재 기준 Docker image는
`vllm/vllm-openai-rocm:latest`이고, host network, `/dev/kfd`, `/dev/dri`,
`--ipc=host`, `--group-add=video`, eager mode, custom all-reduce disable 설정을
사용합니다.

`.venv`에 vLLM 관련 패키지가 섞였을 때는 다음 스크립트로 정리합니다.

```bash
scripts/clean_venv_vllm_residue.sh --venv .venv --apply
```

## 모델/runner 구성

모델 별칭은 `configs/models.yaml`의 `vision_language_models`에서 관리합니다.

- `qwen36_baseline`: 현재 Docker baseline
- `qwen3_vl`: Qwen3 VL instruct 계열 후보
- `bizonai_ocr`: `ONTHEIT/BizOnAI-OCR`
- `exaone45_33b`: `LGAI-EXAONE/EXAONE-4.5-33B`
- `paddleocr`: local optional OCR runner

Qwen/BizOnAI/EXAONE은 동일한 `VllmChatRunner` adapter를 사용합니다. runner는
OpenAI-compatible `/v1/chat/completions` payload를 만들고, page image를 data
URL로 첨부해 JSON-only transcription prompt를 전송합니다.

## 주요 모듈

- `src/gwanbo_ocr/cli.py`: Typer 기반 CLI entrypoint
- `src/gwanbo_ocr/manifest.py`: `/root/peti` manifest 생성
- `src/gwanbo_ocr/pdf/integrity.py`: PDF header, EOF, hash, page count 검증
- `src/gwanbo_ocr/pdf/text.py`: native text 추출 메타데이터
- `src/gwanbo_ocr/pdf/classification.py`: PDF 분류 sidecar 생성
- `src/gwanbo_ocr/pdf/layout.py`: text PDF layout/table 분석
- `src/gwanbo_ocr/pdf/profile.py`: manifest row별 PDF feature profile 생성
- `src/gwanbo_ocr/render.py`: PDF page PNG 렌더링
- `src/gwanbo_ocr/strategy.py`: layout cluster 생성 및 parsing strategy 평가
- `src/gwanbo_ocr/prompts.py`: OCR/VLM transcription prompt
- `src/gwanbo_ocr/runners/vllm.py`: OpenAI-compatible VLM runner
- `src/gwanbo_ocr/runners/paddle.py`: PaddleOCR adapter
- `src/gwanbo_ocr/bench.py`: benchmark 실행 및 throughput report
- `src/gwanbo_ocr/metrics.py`: CER/WER/token/table scoring helpers
- `src/gwanbo_ocr/sampling.py`: deterministic sample suite 생성

## 데이터 계약

Sample row는 후속 단계가 공통으로 읽는 기본 단위입니다.

- `sample_id`
- `source`
- `bucket`
- `id`
- `date`
- `category`
- `metadata_path`
- `pdf_path`
- `sha256`
- `pages`
- `text_extractable`
- `selected_pages`
- `strata`
- `truth`

Classification sidecar는 다음 상위 필드를 갖습니다.

- `schema_version`
- `pdf_key`
- `integrity`
- `native_text`
- `decision`

OCR/VLM result는 다음 상위 필드를 갖습니다.

- `status`
- `text`
- `tables`
- `blocks`
- `raw_response`
- `latency_ms`
- `usage`
- `model_id`
- `prompt_version`
- `image_sha256`
- `error`

PDF profile row는 cluster 입력으로 다음 상위 필드를 갖습니다.

- `schema_version`: `pdf-profile/v1`
- `pdf_key`, `id`, `theme`, `year`, `category`, `agency`
- `pdf_path`, `pdf_abs_path`, `pdf_exists`
- `size_bytes`, `pages`, `integrity_status`
- `text_extractable`, `text_mode`, `total_chars`
- `layout_class`, `table_count`, `table_text_ratio`, `form_score`, `text_quality`
- `error`

Layout cluster row는 다음 상위 필드를 갖습니다.

- `schema_version`: `layout-cluster/v1`
- `cluster_id`
- `year`, `theme`, `dominant_category`
- `feature_signature`
- `count`, `sample_pdf_keys`
- `assigned_strategy`, `confidence`, `reasons`
- `profile_summary`

## 일반 실행 흐름

```bash
gwanbo-ocr manifest build \
  --peti-root /root/peti \
  --output runs/<run_id>/pdf_manifest.jsonl

gwanbo-ocr pdf classify \
  --input runs/<run_id>/pdf_manifest.jsonl \
  --output runs/<run_id>/classification \
  --max-pages 3 \
  --workers 8

gwanbo-ocr pdf layout \
  --classification runs/<run_id>/classification/manifest.jsonl \
  --output runs/<run_id>/layout \
  --table-strategy auto

gwanbo-ocr pdf render \
  --input runs/<run_id>/classification/manifest.jsonl \
  --output runs/<run_id>/images \
  --dpi 200 \
  --max-long-edge 2400

gwanbo-ocr pdf profile \
  --input runs/<run_id>/pdf_manifest.jsonl \
  --output runs/<run_id>/profiles \
  --max-pages 3 \
  --workers 8 \
  --sample-per-bucket 20

gwanbo-ocr strategy cluster \
  --profiles runs/<run_id>/profiles/manifest.jsonl \
  --output runs/<run_id>/clusters

gwanbo-ocr strategy evaluate \
  --clusters runs/<run_id>/clusters/cluster_manifest.jsonl \
  --output runs/<run_id>/strategy_eval

gwanbo-ocr bench run \
  --suite runs/<run_id>/images/manifest.jsonl \
  --runner qwen36_baseline \
  --base-url http://127.0.0.1:8000/v1 \
  --run-dir runs/<run_id>/bench/qwen36_baseline

gwanbo-ocr bench score \
  --run runs/<run_id>/bench/qwen36_baseline \
  --output runs/<run_id>/reports/qwen36_baseline
```

## 검증 기준

현재 구현은 unit/smoke 검증을 먼저 안정화하고, gold label이 준비되면 acceptance
metric을 적용하는 구조입니다.

기준 metric:

- PDF integrity/text/layout 단위 테스트
- JSONL I/O 및 deterministic sampling 테스트
- VLM payload construction 및 schema-echo rejection 테스트
- `/v1/models` healthcheck
- rendered page 1건 이상 실제 image request smoke
- CER/WER
- table cell F1
- critical-token F1
- throughput/latency report

목표 acceptance 기준:

- gold suite completion 100%
- timeout/error <= 1%
- clean text median normalized CER <= 1%
- scanned median CER <= 6%
- critical-token F1 >= 0.97
- table cell F1 >= 0.85
