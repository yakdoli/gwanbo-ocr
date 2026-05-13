# MarkItDown 통합 가이드

## 개요

**MarkItDown**은 Microsoft의 오픈소스 도구로, PDF, DOCX, PPTX 등 다양한 문서 형식을 마크다운으로 변환합니다. `gwanbo-ocr` 프로젝트에서는 PDF 메타데이터 추출 및 문서 변환에 사용됩니다.

- **GitHub**: https://github.com/microsoft/markitdown
- **현재 버전**: `markitdown[pdf]>=0.1.3,<0.2`

## 설치

### 로컬 환경

```bash
# Python 3.12+ 필수
python3 -m venv .venv
source .venv/bin/activate

# PDF 지원 포함
pip install -e ".[pdf]"

# 또는 전체 설치
pip install -e ".[pdf,qwen,dev]"
```

### Docker 환경

#### 단순 CLI 이미지 빌드

```bash
docker build -f Dockerfile.markitdown -t markitdown:latest .

# 사용 예
docker run --rm -i markitdown:latest < input.pdf > output.md
```

#### Docker Compose (권장)

```bash
# 기본 CLI 서비스
docker-compose -f docker-compose.markitdown.yml up markitdown

# HTTP 서버 모드 (배치 처리용)
docker-compose -f docker-compose.markitdown.yml --profile server up markitdown-server
```

## 사용법

### 1. Python API 직접 사용

#### 기본 변환

```python
import markitdown

# 파일 경로에서 변환
md = markitdown.MarkItDown()
result = md.convert("document.pdf")
print(result.text_content)

# 파일 스트림에서 변환
with open("document.pdf", "rb") as f:
    result = md.convert_stream(f, file_path="document.pdf")
    print(result.text_content)
```

#### gwanbo-ocr 파이프라인과 통합

```python
from gwanbo_ocr.manifest import build_manifest
from pathlib import Path
import markitdown

# 매니페스트 생성
manifest_path = Path("runs/my-run/pdf_manifest.jsonl")
build_manifest(peti_root="/root/peti", output_path=manifest_path)

# 각 PDF 변환
md = markitdown.MarkItDown()
with open(manifest_path) as f:
    for line in f:
        sample = json.loads(line)
        pdf_path = sample["pdf_path"]

        result = md.convert(str(pdf_path))

        # 마크다운 저장
        md_path = Path(f"runs/my-run/markdown/{sample['sample_id']}.md")
        md_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.write_text(result.text_content)
```

### 2. CLI 사용

#### 단일 파일 변환

```bash
markitdown document.pdf > document.md

markitdown document.docx > document.md

# 스트림 처리
cat document.pdf | markitdown > document.md
```

#### Docker CLI 사용

```bash
docker run --rm \
  -v /root/gwanbo-ocr/runs:/workspace/runs:ro \
  -v /root/gwanbo-ocr/outputs:/workspace/outputs:rw \
  markitdown:latest /workspace/runs/my-run/sample.pdf > output.md

# 배치 처리
for pdf in runs/my-run/pdfs/*.pdf; do
  docker run --rm -i markitdown:latest < "$pdf" > "outputs/$(basename $pdf .pdf).md"
done
```

### 3. HTTP API 서버 사용

#### 서버 시작

```bash
docker-compose -f docker-compose.markitdown.yml --profile server up -d markitdown-server

# 또는 로컬에서
python scripts/markitdown_server.py
```

#### 파일 업로드 변환

```bash
curl -X POST "http://localhost:8080/convert/file" \
  -F "file=@document.pdf" \
  -F "include_metadata=true"
```

#### 경로 기반 변환

```bash
curl -X POST "http://localhost:8080/convert/path" \
  -H "Content-Type: application/json" \
  -d '{
    "file_path": "/workspace/runs/my-run/sample.pdf",
    "output_path": "/workspace/outputs/sample.md",
    "include_metadata": true
  }'
```

#### 배치 변환

```bash
curl -X POST "http://localhost:8080/convert/batch" \
  -H "Content-Type: application/json" \
  -d '{
    "file_paths": [
      "/workspace/runs/my-run/sample1.pdf",
      "/workspace/runs/my-run/sample2.pdf"
    ],
    "output_dir": "/workspace/outputs"
  }'
```

#### 헬스 체크

```bash
curl "http://localhost:8080/health"
```

## 시스템 의존성

MarkItDown의 PDF 지원을 위해 다음 라이브러리가 필요합니다:

### 시스템 패키지 (Linux)

```bash
# Ubuntu/Debian
sudo apt-get install libmupdf-dev libpdf-dev

# Fedora/RHEL
sudo dnf install mupdf-devel

# macOS
brew install mupdf
```

### 자동 설치 (Docker)

Dockerfile에서 자동으로 설치됩니다:
```dockerfile
RUN apt-get install libmupdf-dev libpdf-dev
```

## 통합 워크플로우

### 예시 1: PDF 분류 + 마크다운 변환

```bash
# 1. 매니페스트 생성
gwanbo-ocr manifest build \
  --peti-root /root/peti \
  --output runs/my-run/pdf_manifest.jsonl

# 2. PDF 분류
gwanbo-ocr pdf classify \
  --input runs/my-run/pdf_manifest.jsonl \
  --output runs/my-run/classification

# 3. 분류된 PDF를 마크다운으로 변환
python << 'EOF'
import json
from pathlib import Path
import markitdown

md = markitdown.MarkItDown()
output_dir = Path("runs/my-run/markdown")
output_dir.mkdir(exist_ok=True)

with open("runs/my-run/classification/manifest.jsonl") as f:
    for line in f:
        sample = json.loads(line)
        if sample.get("classification", {}).get("decision") == "text_pdf":
            result = md.convert(sample["pdf_path"])
            md_file = output_dir / f"{sample['sample_id']}.md"
            md_file.write_text(result.text_content)
            print(f"Converted: {sample['sample_id']}")
EOF
```

### 예시 2: 배치 처리 (Docker Compose)

```bash
# Docker Compose로 서버 시작
docker-compose -f docker-compose.markitdown.yml --profile server up -d

# 파이썬 스크립트로 배치 변환
python << 'EOF'
import json
import requests
from pathlib import Path

API_URL = "http://localhost:8080"

# 변환할 PDF 목록 수집
pdf_files = list(Path("/root/gwanbo-ocr/runs/my-run").glob("*.pdf"))

# API에 배치 변환 요청
response = requests.post(
    f"{API_URL}/convert/batch",
    json={
        "file_paths": [str(p) for p in pdf_files],
        "output_dir": "/root/gwanbo-ocr/outputs"
    }
)

print(json.dumps(response.json(), indent=2))
EOF
```

### 예시 3: OCR 벤치마크와 통합

```bash
# 1. 레이아웃 분석
gwanbo-ocr pdf layout \
  --classification runs/my-run/classification/manifest.jsonl \
  --output runs/my-run/layout

# 2. 텍스트 추출 (MarkItDown 사용)
python << 'EOF'
import json
from pathlib import Path
import markitdown

md = markitdown.MarkItDown()
markdown_dir = Path("runs/my-run/markdown")
markdown_dir.mkdir(exist_ok=True)

with open("runs/my-run/layout/manifest.jsonl") as f:
    for line in f:
        sample = json.loads(line)
        result = md.convert(sample["pdf_path"])

        # 추출된 텍스트 저장
        md_file = markdown_dir / f"{sample['sample_id']}.md"
        md_file.write_text(result.text_content)

        # 메타데이터 저장
        meta_file = markdown_dir / f"{sample['sample_id']}.meta.json"
        meta_file.write_text(json.dumps({
            "extraction_method": "markitdown",
            "document_class": sample.get("layout_class"),
            "has_tables": sample.get("table_count", 0) > 0,
        }, indent=2))
EOF

# 3. 렌더링 및 벤치마킹 (기존 파이프라인)
gwanbo-ocr pdf render \
  --input runs/my-run/classification/manifest.jsonl \
  --output runs/my-run/images

gwanbo-ocr bench run \
  --suite runs/my-run/images/manifest.jsonl \
  --runner qwen36_baseline \
  --base-url http://127.0.0.1:8000/v1 \
  --run-dir runs/my-run/bench/qwen36_baseline
```

## 성능 고려사항

### 리소스 요구사항

| 작업 | CPU | 메모리 | 예상 시간 |
|------|-----|--------|---------|
| 단일 PDF (10p) | 1 코어 | 512MB | < 1초 |
| 중간 크기 (100p) | 1 코어 | 1GB | 1-5초 |
| 배치 (1000 파일) | 2+ 코어 | 4GB+ | 1-5분 |

### 최적화 팁

1. **병렬 처리**: 배치 작업은 Docker 서버 사용 권장
2. **메모리**: 대용량 PDF는 스트림 처리 사용
3. **캐싱**: 같은 PDF 재처리 시 결과 캐시
4. **디스크**: 출력 경로는 충분한 여유 공간 필수

## 디버깅

### 시스템 의존성 확인

```bash
# libmupdf 확인
python -c "import markitdown; print(markitdown.__version__)"

# PDF 지원 확인
python << 'EOF'
import markitdown
md = markitdown.MarkItDown()
try:
    result = md.convert("test.pdf")
    print("PDF support: OK")
except Exception as e:
    print(f"PDF support: FAILED - {e}")
EOF
```

### Docker 빌드 문제

```bash
# 빌드 로그 확인
docker build --progress=plain -f Dockerfile.markitdown -t markitdown:latest .

# 컨테이너 내에서 테스트
docker run --rm -it markitdown:latest python << 'EOF'
import markitdown
print(markitdown.__version__)
EOF
```

### API 서버 문제

```bash
# 서버 로그 확인
docker-compose -f docker-compose.markitdown.yml --profile server logs markitdown-server

# 수동 테스트
curl -v http://localhost:8080/health
```

## 지원 형식

### 입력 형식
- **PDF**: ✅ (PyMuPDF 사용)
- **Word**: DOCX, DOC
- **PowerPoint**: PPTX, PPT
- **이미지**: PNG, JPG (OCR)
- **기타**: HTML, Markdown, Text

### 출력 형식
- **Markdown**: 기본 출력 형식
- **메타데이터**: JSON (옵션)
- **이미지**: Base64 인코딩 (옵션)

## 라이선스

- **MarkItDown**: MIT License
- **gwanbo-ocr**: (프로젝트 라이선스 참고)

## 참고 자료

- [MarkItDown GitHub](https://github.com/microsoft/markitdown)
- [MarkItDown PyPI](https://pypi.org/project/markitdown/)
- [gwanbo-ocr CLAUDE.md](../CLAUDE.md)
- [gwanbo-ocr AGENTS.md](../AGENTS.md)
