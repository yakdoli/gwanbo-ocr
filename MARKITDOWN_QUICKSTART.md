# MarkItDown Docker 빌드 및 사용 가이드

## 📋 목차

1. [설치](#설치)
2. [Docker 빌드](#docker-빌드)
3. [CLI 명령어](#cli-명령어)
4. [Docker Compose 사용](#docker-compose-사용)
5. [HTTP API 사용](#http-api-사용)
6. [통합 예제](#통합-예제)

---

## 설치

### 1️⃣ 로컬 환경

```bash
# 프로젝트 디렉토리로 이동
cd /root/gwanbo-ocr

# Python 3.12+ 필수
python3 -m venv .venv
source .venv/bin/activate

# PDF 지원 포함 설치
pip install -e ".[pdf]"
```

### 2️⃣ 필수 시스템 라이브러리

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y libmupdf-dev libpdf-dev

# Fedora/RHEL
sudo dnf install -y mupdf-devel

# macOS
brew install mupdf
```

---

## Docker 빌드

### 기본 MarkItDown 이미지

```bash
# 빌드
docker build -f Dockerfile.markitdown -t markitdown:latest .

# 테스트
docker run --rm -i markitdown:latest --help
```

### HTTP 서버 이미지

```bash
# 빌드
docker build -f Dockerfile.markitdown.server -t markitdown-server:latest .

# 테스트
docker run --rm -p 8081:8080 markitdown-server:latest
```

### 모든 이미지 빌드

```bash
docker-compose -f docker-compose.markitdown.yml build
```

---

## CLI 명령어

### 단일 파일 변환

```bash
# 로컬에서
gwanbo-ocr convert file \
  document.pdf \
  --output output.md

# Docker에서
docker run --rm \
  -v $(pwd):/workspace \
  markitdown:latest /workspace/document.pdf > output.md
```

### 매니페스트 배치 변환

```bash
# 로컬에서
gwanbo-ocr convert manifest \
  --input runs/my-run/pdf_manifest.jsonl \
  --output runs/my-run/markdown \
  --workers 4

# 옵션
gwanbo-ocr convert manifest \
  --input runs/my-run/pdf_manifest.jsonl \
  --output runs/my-run/markdown \
  --key sample_id \
  --pdf-path-field pdf_path \
  --workers 4 \
  --limit 100 \
  --skip-errors \
  --force
```

### CLI 도움말

```bash
gwanbo-ocr convert --help
gwanbo-ocr convert file --help
gwanbo-ocr convert manifest --help
```

---

## Docker Compose 사용

### CLI 모드

```bash
# 컨테이너 시작
docker-compose -f docker-compose.markitdown.yml up -d markitdown

# 사용
docker-compose -f docker-compose.markitdown.yml exec markitdown \
  markitdown /workspace/runs/sample.pdf > output.md

# 중지
docker-compose -f docker-compose.markitdown.yml down
```

### HTTP 서버 모드

```bash
# 서버 시작
docker-compose -f docker-compose.markitdown.yml --profile server up -d markitdown-server

# 헬스 체크
curl http://localhost:8081/health

# 중지
docker-compose -f docker-compose.markitdown.yml --profile server down
```

---

## HTTP API 사용

### API 서버 시작

```bash
# Docker Compose 사용
docker-compose -f docker-compose.markitdown.yml --profile server up -d markitdown-server

# 또는 로컬에서
uvicorn scripts.markitdown_server:app --host 0.0.0.0 --port 8081
```

### API 문서

브라우저에서 다음 주소 방문:
- **Swagger UI**: http://localhost:8081/docs
- **ReDoc**: http://localhost:8081/redoc

### 1. 파일 업로드 변환

```bash
curl -X POST "http://localhost:8081/convert/file" \
  -F "file=@document.pdf" \
  | jq '.markdown_content'
```

### 2. 경로 기반 변환

```bash
curl -X POST "http://localhost:8081/convert/path" \
  -H "Content-Type: application/json" \
  -d '{
    "file_path": "/workspace/runs/sample.pdf",
    "output_path": "/workspace/outputs/sample.md"
  }' | jq '.'
```

### 3. 배치 변환

```bash
curl -X POST "http://localhost:8081/convert/batch" \
  -H "Content-Type: application/json" \
  -d '{
    "file_paths": [
      "/workspace/runs/sample1.pdf",
      "/workspace/runs/sample2.pdf"
    ],
    "output_dir": "/workspace/outputs"
  }' | jq '.'
```

### 4. 헬스 체크

```bash
curl http://localhost:8081/health | jq '.'
```

---

## 통합 예제

### 예제 1: 전체 파이프라인

```bash
#!/bin/bash

RUN_ID="my-ocr-run"
PETI_ROOT="/root/peti"

# 1. 매니페스트 생성
echo "📝 Creating manifest..."
gwanbo-ocr manifest build \
  --peti-root $PETI_ROOT \
  --output runs/$RUN_ID/pdf_manifest.jsonl \
  --limit 10

# 2. PDF 분류
echo "🔍 Classifying PDFs..."
gwanbo-ocr pdf classify \
  --input runs/$RUN_ID/pdf_manifest.jsonl \
  --output runs/$RUN_ID/classification \
  --max-pages 3 \
  --workers 4

# 3. 마크다운 변환
echo "📄 Converting to Markdown..."
gwanbo-ocr convert manifest \
  --input runs/$RUN_ID/classification/manifest.jsonl \
  --output runs/$RUN_ID/markdown \
  --workers 4 \
  --skip-errors

# 4. 결과 확인
echo "✅ Done! Check output in runs/$RUN_ID/markdown/"
ls -la runs/$RUN_ID/markdown/ | head -10
```

### 예제 2: Docker 배치 처리

```bash
#!/bin/bash

# 서버 시작
docker-compose -f docker-compose.markitdown.yml --profile server up -d

# Python 스크립트로 배치 처리
python << 'EOF'
import json
import requests
from pathlib import Path

API_URL = "http://localhost:8081"

# PDF 목록 수집
pdfs = [str(p) for p in Path("/root/gwanbo-ocr/runs").rglob("*.pdf")][:10]

# 배치 변환 요청
response = requests.post(
    f"{API_URL}/convert/batch",
    json={
        "file_paths": pdfs,
        "output_dir": "/root/gwanbo-ocr/outputs"
    }
)

result = response.json()
print(json.dumps(result, indent=2))
print(f"\n✅ {len(result['results'])} files processed")
EOF

# 서버 중지
docker-compose -f docker-compose.markitdown.yml --profile server down
```

### 예제 3: Python 직접 사용

```python
import markitdown
from pathlib import Path

# 단일 파일 변환
md = markitdown.MarkItDown()
result = md.convert("document.pdf")
print(result.text_content)

# 결과 저장
output_file = Path("output.md")
output_file.write_text(result.text_content)

# 배치 처리
output_dir = Path("markdown_outputs")
output_dir.mkdir(exist_ok=True)

for pdf_file in Path("pdfs").glob("*.pdf"):
    result = md.convert(str(pdf_file))
    md_file = output_dir / f"{pdf_file.stem}.md"
    md_file.write_text(result.text_content)
    print(f"✅ {md_file}")
```

---

## 성능 최적화

### 병렬 처리

```bash
# 4개 워커로 병렬 처리
gwanbo-ocr convert manifest \
  --input pdf_manifest.jsonl \
  --output markdown_output \
  --workers 4
```

### 대용량 파일

```python
import markitdown
import io

# 스트림 처리로 메모리 효율성 증대
md = markitdown.MarkItDown()
with open("large.pdf", "rb") as f:
    result = md.convert_stream(f, file_path="large.pdf")
    print(result.text_content)
```

---

## 문제 해결

### PDF 지원 확인

```bash
python -c "
import markitdown
md = markitdown.MarkItDown()
try:
    result = md.convert('test.pdf')
    print('✅ PDF support OK')
except Exception as e:
    print(f'❌ PDF support FAILED: {e}')
"
```

### Docker 빌드 디버깅

```bash
docker build --progress=plain -f Dockerfile.markitdown -t markitdown:latest .
```

### 로그 확인

```bash
# Docker Compose 로그
docker-compose -f docker-compose.markitdown.yml logs -f

# 특정 서비스
docker-compose -f docker-compose.markitdown.yml logs -f markitdown-server
```

---

## 참고자료

- [MarkItDown GitHub](https://github.com/microsoft/markitdown)
- [전체 통합 가이드](docs/markitdown-integration.md)
- [gwanbo-ocr CLAUDE.md](CLAUDE.md)
- [Docker Compose 설정](docker-compose.markitdown.yml)

---

## 빠른 시작 체크리스트

- [ ] 필수 시스템 라이브러리 설치
- [ ] Python 3.12+ 설치
- [ ] `pip install -e ".[pdf]"` 실행
- [ ] `gwanbo-ocr convert file --help` 실행
- [ ] 첫 PDF 변환 테스트
- [ ] Docker 빌드 (옵션)
- [ ] Docker Compose 테스트 (옵션)

---

**🎉 이제 MarkItDown을 사용할 준비가 되었습니다!**
