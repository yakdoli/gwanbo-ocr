# MarkItDown 통합 아키텍처 및 배포 가이드

## 📐 아키텍처 개요

```
┌─────────────────────────────────────────────────────────────┐
│                     gwanbo-ocr Pipeline                      │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────┐       ┌──────────────┐                    │
│  │   Manifest   │──────▶│  PDF Input   │                    │
│  │    Build     │       │  Manifest    │                    │
│  └──────────────┘       └──────────────┘                    │
│         │                      │                            │
│         └──────────┬───────────┘                            │
│                    │                                        │
│         ┌──────────▼──────────┐                            │
│         │  PDF Classify       │  ◀──── Classification      │
│         │  (integrity, text)  │       Results             │
│         └──────────┬──────────┘                            │
│                    │                                        │
│    ┌───────────────┼───────────────┐                      │
│    │               │               │                      │
│    ▼               ▼               ▼                      │
│ ┌────────┐   ┌─────────────┐   ┌─────────┐              │
│ │ Layout │   │ MarkItDown  │◀──│ Convert │              │
│ │Analysis│   │  Converter  │   │  Stage  │              │
│ └────────┘   └─────────────┘   └─────────┘              │
│              (THIS MODULE)                              │
│    │               │               │                      │
│    └───────────────┼───────────────┘                      │
│                    │                                        │
│         ┌──────────▼──────────┐                            │
│         │  PDF Render         │  ◀──── Rendered          │
│         │  (images for OCR)   │       Images             │
│         └──────────┬──────────┘                            │
│                    │                                        │
│         ┌──────────▼──────────┐                            │
│         │  VLM/OCR Benchmark  │  ◀──── OCR Results       │
│         │  (Qwen, etc.)       │                          │
│         └──────────┬──────────┘                            │
│                    │                                        │
│         ┌──────────▼──────────┐                            │
│         │  Score & Report     │  ◀──── Metrics           │
│         │                      │       (CER, WER, F1)    │
│         └─────────────────────┘                            │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

---

## 🏗️ 컴포넌트 구조

### 1. CLI Layer (`gwanbo_ocr/cli.py`)

```
convert_app
├── convert file      # 단일 파일 변환
└── convert manifest  # 배치 변환
```

**특징:**
- Typer 기반 CLI 프레임워크
- 병렬 처리 지원 (ThreadPoolExecutor)
- 에러 처리 및 스킵 옵션

### 2. Docker Layer

#### Dockerfile.markitdown (CLI 이미지)
- 기본 Python 3.12 slim 이미지
- MarkItDown[pdf] 의존성
- 시스템 라이브러리 (libmupdf-dev, libpdf-dev)
- 비-루트 사용자 (markitdown:1000)

#### Dockerfile.markitdown.server (HTTP API 이미지)
- FastAPI 기반 HTTP 서버
- 멀티파트 파일 업로드 지원
- 배치 변환 엔드포인트
- 상태 체크 및 헬스 체크

#### docker-compose.markitdown.yml
```yaml
services:
  markitdown:          # CLI 모드 (기본)
  markitdown-server:   # HTTP 서버 모드 (profile: server)
```

### 3. API Layer (`scripts/markitdown_server.py`)

**Endpoints:**
- `POST /convert/file` - 파일 업로드 변환
- `POST /convert/path` - 경로 기반 변환
- `POST /convert/batch` - 배치 변환
- `GET /health` - 헬스 체크
- `GET /` - API 정보

---

## 🚀 배포 옵션

### 옵션 1️⃣: 로컬 Python (최소 리소스)

```bash
pip install -e ".[pdf]"
gwanbo-ocr convert manifest --manifest pdf_manifest.jsonl --output markdown/
```

**용도:** 개발, 소규모 배치
**리소스:** CPU 1-2개, 메모리 512MB-1GB
**속도:** 순차 처리

### 옵션 2️⃣: Docker CLI (독립형)

```bash
docker build -f Dockerfile.markitdown -t markitdown:latest .
docker run --rm -v /data:/workspace markitdown:latest /workspace/input.pdf > output.md
```

**용도:** 개발 환경, CI/CD
**리소스:** CPU 1개, 메모리 512MB
**속도:** 직렬 처리

### 옵션 3️⃣: Docker Compose CLI (벌크)

```bash
docker-compose -f docker-compose.markitdown.yml up -d markitdown

# 컨테이너 내에서 gwanbo-ocr 명령 실행
docker-compose exec markitdown gwanbo-ocr convert manifest \
  --manifest /workspace/pdf_manifest.jsonl \
  --output /workspace/markdown \
  --workers 4
```

**용제:** 배치 처리, 리소스 제한 환경
**리소스:** CPU 2개, 메모리 4GB
**속도:** 병렬 처리 (4 workers)

### 옵션 4️⃣: HTTP API 서버 (서비스)

```bash
docker-compose -f docker-compose.markitdown.yml --profile server up -d

curl -X POST http://localhost:8080/convert/batch \
  -H "Content-Type: application/json" \
  -d '{"file_paths": [...], "output_dir": "..."}'
```

**용도:** 마이크로서비스, 여러 애플리케이션 공유
**리소스:** CPU 2개, 메모리 4GB
**속도:** 병렬 처리, 요청 기반

---

## 📊 성능 비교

| 옵션 | 처리량 | 지연시간 | 리소스 | 복잡도 |
|-----|-------|---------|--------|--------|
| 로컬 Python | 낮음 | 즉시 | 낮음 | 낮음 |
| Docker CLI | 낮음 | 중간 | 낮음 | 낮음 |
| Docker Compose | 중간 | 낮음 | 중간 | 중간 |
| HTTP API | 높음 | 낮음 | 높음 | 높음 |

---

## 🔧 설정 및 최적화

### 병렬 워커 수

```bash
# 로컬 CPU 수 확인
python -c "import multiprocessing; print(multiprocessing.cpu_count())"

# 권장값: CPU 수 - 1
gwanbo-ocr convert manifest \
  --workers 7 \  # 8 core - 1
  --manifest pdf_manifest.jsonl \
  --output markdown/
```

### 메모리 관리

```yaml
# docker-compose.yml에서 리소스 제한 설정
services:
  markitdown:
    deploy:
      resources:
        limits:
          cpus: '4'
          memory: 8G
        reservations:
          cpus: '2'
          memory: 4G
```

### 타임아웃 설정

```python
# 대용량 PDF의 경우
import markitdown
import signal

def timeout_handler(signum, frame):
    raise TimeoutError("PDF conversion timeout")

signal.signal(signal.SIGALRM, timeout_handler)
signal.alarm(300)  # 5분 타임아웃

try:
    md = markitdown.MarkItDown()
    result = md.convert("large.pdf")
finally:
    signal.alarm(0)
```

---

## 📋 통합 체크리스트

### 설치 단계
- [ ] Python 3.12+ 설치
- [ ] 시스템 라이브러리 설치 (libmupdf-dev 등)
- [ ] `pip install -e ".[pdf]"` 실행
- [ ] `gwanbo-ocr convert --help` 확인

### Docker 준비 (옵션)
- [ ] Docker 설치 확인
- [ ] `docker --version` 실행
- [ ] Docker Compose 설치 확인
- [ ] 디스크 공간 확인 (최소 5GB)

### 파이프라인 통합
- [ ] Manifest 생성 가능 확인
- [ ] 첫 번째 PDF 변환 테스트
- [ ] 배치 변환 테스트
- [ ] 출력 형식 검증

### 운영 준비
- [ ] 로그 수집 설정
- [ ] 모니터링 설정 (선택)
- [ ] 백업 정책 수립
- [ ] 문서화 완료

---

## 🔐 보안 고려사항

### 1. 파일 검증

```python
# 신뢰할 수 없는 PDF 처리 시
import hashlib
from pathlib import Path

def validate_pdf(pdf_path: Path) -> bool:
    """PDF 파일 무결성 검사"""
    if not pdf_path.suffix.lower() == '.pdf':
        return False

    with open(pdf_path, 'rb') as f:
        header = f.read(4)
        if header != b'%PDF':
            return False

    return True
```

### 2. 리소스 제한

```yaml
# docker-compose.yml
services:
  markitdown:
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 2G  # OOM 방지
```

### 3. 비-루트 실행

```dockerfile
# 모든 Dockerfile에서
RUN useradd -m -u 1000 markitdown
USER markitdown
```

---

## 📚 추가 리소스

| 문서 | 용도 |
|------|------|
| [MARKITDOWN_QUICKSTART.md](MARKITDOWN_QUICKSTART.md) | 빠른 시작 가이드 |
| [docs/markitdown-integration.md](docs/markitdown-integration.md) | 상세 통합 가이드 |
| [CLAUDE.md](CLAUDE.md) | 프로젝트 개요 |
| [AGENTS.md](AGENTS.md) | 에이전트 지침 |

---

## 🆘 트러블슈팅

### 문제: `ImportError: markitdown`

```bash
# 해결
pip install -e ".[pdf]"
```

### 문제: `No module named 'markitdown._pdf_converter_mupdf'`

```bash
# 시스템 라이브러리 확인
apt-get install libmupdf-dev libpdf-dev
pip install --force-reinstall markitdown[pdf]
```

### 문제: Out of Memory

```bash
# 워커 수 감소
gwanbo-ocr convert manifest \
  --workers 1 \  # 병렬도 감소
  --limit 10 \   # 배치 크기 감소
  ...
```

### 문제: Docker 빌드 실패

```bash
# 빌드 로그 확인
docker build --progress=plain -f Dockerfile.markitdown -t markitdown:latest .

# 베이스 이미지 업데이트
docker pull python:3.12-slim-bookworm
```

---

## 📈 향후 개선 계획

- [ ] GPU 가속 지원 (CUDA/ROCm)
- [ ] 캐싱 메커니즘 추가
- [ ] 메트릭 수집 (Prometheus)
- [ ] Kubernetes 배포 지원
- [ ] WebSocket 실시간 스트리밍
- [ ] 다국어 OCR 지원 향상

---

**최종 수정**: 2026-05-13
**버전**: 0.1.0
**상태**: ✅ 프로덕션 준비 완료
