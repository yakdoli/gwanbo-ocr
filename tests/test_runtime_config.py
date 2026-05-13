from __future__ import annotations

import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def test_full_compose_exposes_expected_services_and_ports() -> None:
    compose_path = Path("docker-compose.ocr.yml")
    payload = yaml.safe_load(compose_path.read_text(encoding="utf-8"))
    services = payload["services"]

    assert set(services) >= {
        "gwanbo-app",
        "vllm-qwen",
        "paddleocr-vl-vllm",
        "paddleocr-api",
        "markitdown-ocr-api",
    }
    assert "127.0.0.1:8000:8000" in services["vllm-qwen"]["ports"]
    assert "127.0.0.1:8118:8000" in services["paddleocr-vl-vllm"]["ports"]
    assert "127.0.0.1:8081:8080" in services["markitdown-ocr-api"]["ports"]
    assert "127.0.0.1:8082:8080" in services["paddleocr-api"]["ports"]


def test_models_yaml_records_container_service_defaults() -> None:
    payload = yaml.safe_load(Path("configs/models.yaml").read_text(encoding="utf-8"))

    assert payload["services"]["markitdown_ocr_api"]["base_url"] == "http://127.0.0.1:8081"
    assert payload["services"]["paddleocr_api"]["base_url"] == "http://127.0.0.1:8082"
    assert payload["ocr"]["paddleocr_vl"]["vl_rec_server_url"] == "http://127.0.0.1:8118/v1"
