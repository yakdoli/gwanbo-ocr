FROM python:3.12-slim-bookworm

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libmupdf-dev \
    libpdf-dev \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src ./src
COPY configs ./configs

RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel \
    && python -m pip install --no-cache-dir -e ".[pdf,qwen,services]"

ENTRYPOINT ["gwanbo-ocr"]
CMD ["--help"]
