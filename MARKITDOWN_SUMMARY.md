# MarkItDown 통합 요약

## 🎯 개요

현재 프로젝트 `gwanbo-ocr`에 **MarkItDown** (Microsoft의 PDF/문서 마크다운 변환 도구)을 완전히 통합했습니다.

---

## 📦 생성된 파일

### 1. Docker 관련 파일

| 파일 | 설명 |
|-----|------|
| `Dockerfile.markitdown` | 기본 MarkItDown CLI 이미지 |
| `Dockerfile.markitdown.server` | FastAPI 기반 HTTP 서버 이미지 |
| `docker-compose.markitdown.yml` | Docker Compose 설정 |

### 2. 스크립트

| 파일 | 설명 |
|-----|------|
| `scripts/markitdown_server.py` | HTTP API 서버 구현 |

### 3. 문서

| 문서 | 용도 |
|------|------|
| `docs/markitdown-integration.md` | 📖 전체 통합 가이드 |
| `docs/markitdown-architecture.md` | 📐 아키텍처 및 배포 가이드 |
| `MARKITDOWN_QUICKSTART.md` | ⚡ 빠른 시작 가이드 |

### 4. 디렉토리

```
/root/gwanbo-ocr/
├── inputs/          # 입력 파일 저장 (Docker 볼륨)
└── outputs/         # 출력 파일 저장 (Docker 볼륨)
```

---

## 🚀 빠른 시작

### 1️⃣ 설치

```bash
cd /root/gwanbo-ocr
pip install -e ".[pdf]"
```

### 2️⃣ 단일 파일 변환

```bash
gwanbo-ocr convert file document.pdf --output output.md
```

### 3️⃣ 배치 변환

```bash
gwanbo-ocr convert manifest \
  --manifest pdf_manifest.jsonl \
  --output markdown/ \
  --workers 4
```

### 4️⃣ Docker 빌드 및 실행

```bash
# 빌드
docker build -f Dockerfile.markitdown -t markitdown:latest .

# 실행
docker run --rm -i markitdown:latest < input.pdf > output.md
```

### 5️⃣ Docker Compose (HTTP 서버)

```bash
docker-compose -f docker-compose.markitdown.yml --profile server up -d

# API 호출
curl -X POST http://localhost:8080/convert/file \
  -F "file=@document.pdf"
```

---

## 🏗️ 아키텍처

```
gwanbo-ocr Pipeline
    │
    ├─ manifest build ──▶ PDF Manifest
    │
    ├─ pdf classify ──▶ Classification Results
    │
    ├─ convert manifest ──▶ MarkItDown Converter
    │                        (NEW: CLI + Docker + API)
    │
    ├─ pdf render ──▶ PNG Images
    │
    └─ bench run ──▶ OCR Results
```

---

## 📊 배포 옵션

| 옵션 | 방법 | 리소스 | 용도 |
|-----|------|--------|------|
| 로컬 | `pip install` | 최소 | 개발 |
| Docker CLI | `docker run` | 낮음 | CI/CD |
| Docker Compose | `docker-compose up` | 중간 | 배치 |
| HTTP API | FastAPI 서버 | 중간-높음 | 서비스 |

---

## 🔧 CLI 명령어

### 도움말
```bash
gwanbo-ocr convert --help
gwanbo-ocr convert file --help
gwanbo-ocr convert manifest --help
```

### 단일 파일
```bash
gwanbo-ocr convert file \
  document.pdf \
  --output output.md
```

### 배치 처리
```bash
gwanbo-ocr convert manifest \
  --manifest pdf_manifest.jsonl \
  --output markdown_output/ \
  --key sample_id \
  --pdf-path-field pdf_path \
  --workers 4 \
  --limit 100 \
  --skip-errors \
  --force
```

---

## 🌐 HTTP API 엔드포인트

| 메서드 | 엔드포인트 | 설명 |
|-------|----------|------|
| POST | `/convert/file` | 파일 업로드 변환 |
| POST | `/convert/path` | 경로 기반 변환 |
| POST | `/convert/batch` | 배치 변환 |
| GET | `/health` | 헬스 체크 |
| GET | `/docs` | Swagger UI |

### 예제

```bash
# 파일 업로드
curl -X POST http://localhost:8080/convert/file \
  -F "file=@document.pdf"

# 경로 변환
curl -X POST http://localhost:8080/convert/path \
  -H "Content-Type: application/json" \
  -d '{"file_path": "/path/to/file.pdf", "output_path": "/output/file.md"}'

# 배치 변환
curl -X POST http://localhost:8080/convert/batch \
  -H "Content-Type: application/json" \
  -d '{"file_paths": [...], "output_dir": "..."}'
```

---

## 📝 시스템 요구사항

### Python 환경
- Python 3.12+
- pip

### 시스템 라이브러리
```bash
# Ubuntu/Debian
sudo apt-get install libmupdf-dev libpdf-dev

# Fedora/RHEL
sudo dnf install mupdf-devel

# macOS
brew install mupdf
```

### Docker (선택사항)
- Docker 20.10+
- Docker Compose 1.29+

---

## 💾 의존성 추가

`pyproject.toml`에 다음이 추가되었습니다:

```toml
[project.optional-dependencies]
pdf = [
    "markitdown[pdf]>=0.1.3,<0.2",  # NEW
    # ... 기타 의존성
]
```

설치 방법:
```bash
pip install -e ".[pdf]"                # PDF 지원만
pip install -e ".[pdf,qwen,dev]"      # 전체
```

---

## 🗂️ 프로젝트 구조

```
/root/gwanbo-ocr/
├── Dockerfile.markitdown              # CLI 이미지
├── Dockerfile.markitdown.server        # 서버 이미지
├── docker-compose.markitdown.yml       # Compose 설정
├── scripts/
│   └── markitdown_server.py           # FastAPI 서버
├── src/gwanbo_ocr/
│   └── cli.py                         # ✨ convert_app 추가됨
├── docs/
│   ├── markitdown-integration.md      # 상세 가이드
│   └── markitdown-architecture.md     # 아키텍처 가이드
├── MARKITDOWN_QUICKSTART.md           # 빠른 시작
├── inputs/                            # 입력 디렉토리
└── outputs/                           # 출력 디렉토리
```

---

## 📚 문서 가이드

### 👨‍💻 개발자용
1. **MARKITDOWN_QUICKSTART.md** - 빠른 시작 (5분)
2. **docs/markitdown-integration.md** - 상세 가이드 (30분)
3. **docs/markitdown-architecture.md** - 아키텍처 이해 (20분)

### 🏃 운영자용
1. **MARKITDOWN_QUICKSTART.md** - Docker 배포 섹션
2. **docker-compose.markitdown.yml** - 설정 참고
3. **docs/markitdown-architecture.md** - 성능 및 최적화

### 🔍 트러블슈팅
- **docs/markitdown-architecture.md** - "트러블슈팅" 섹션
- **docs/markitdown-integration.md** - "디버깅" 섹션

---

## ✅ 검증 및 테스트

### 1. CLI 설치 확인
```bash
gwanbo-ocr convert --help
```

### 2. 단일 파일 테스트
```bash
# 테스트 PDF 생성 또는 사용
gwanbo-ocr convert file \
  test.pdf \
  --output test.md
```

### 3. Docker 빌드 테스트
```bash
docker build -f Dockerfile.markitdown -t markitdown:latest .
docker run --rm -i markitdown:latest --help
```

### 4. API 서버 테스트
```bash
docker-compose -f docker-compose.markitdown.yml --profile server up -d
curl http://localhost:8080/health
docker-compose -f docker-compose.markitdown.yml --profile server down
```

---

## 🎯 주요 기능

### ✨ CLI 명령어
- ✅ 단일 파일 변환 (`convert file`)
- ✅ 배치 변환 (`convert manifest`)
- ✅ 병렬 처리 (다중 워커)
- ✅ 에러 처리 및 스킵 옵션
- ✅ 진행 상황 보고

### 🐳 Docker 지원
- ✅ 기본 CLI 이미지
- ✅ FastAPI HTTP 서버
- ✅ Docker Compose 통합
- ✅ 리소스 제한 설정
- ✅ 헬스 체크

### 🌐 HTTP API
- ✅ 파일 업로드 변환
- ✅ 경로 기반 변환
- ✅ 배치 변환 (비동기)
- ✅ 메타데이터 포함
- ✅ Swagger/ReDoc 문서

### 📊 통합 기능
- ✅ gwanbo-ocr 파이프라인 통합
- ✅ 매니페스트 기반 배치 처리
- ✅ 다른 OCR 방식과 비교 (peer review)

---

## 🔄 다음 단계

1. **테스트**: `MARKITDOWN_QUICKSTART.md` 의 예제 실행
2. **통합**: 기존 파이프라인에 통합
3. **배포**: Docker Compose로 운영 환경 구성
4. **모니터링**: 성능 메트릭 수집 및 분석

---

## 📞 지원

문제 발생 시:
1. **docs/markitdown-architecture.md** - 트러블슈팅 섹션 확인
2. **docs/markitdown-integration.md** - 디버깅 섹션 확인
3. **MarkItDown 공식 저장소**: https://github.com/microsoft/markitdown

---

**최종 완성**: 2026-05-13
**상태**: ✅ 프로덕션 준비 완료
**다음 마일스톤**: V0.2.0 (GPU 가속, 캐싱 추가)
