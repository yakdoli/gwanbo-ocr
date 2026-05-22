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


def test_markitdown_compose_uses_modern_compose_schema() -> None:
    payload = yaml.safe_load(Path("docker-compose.markitdown.yml").read_text(encoding="utf-8"))

    assert "version" not in payload
    assert "services" in payload
    assert "127.0.0.1:8081:8080" in payload["services"]["markitdown-server"]["ports"]


def test_models_yaml_records_container_service_defaults() -> None:
    payload = yaml.safe_load(Path("configs/models.yaml").read_text(encoding="utf-8"))

    assert payload["services"]["markitdown_ocr_api"]["base_url"] == "http://127.0.0.1:8081"
    assert payload["services"]["paddleocr_api"]["base_url"] == "http://127.0.0.1:8082"
    assert payload["ocr"]["paddleocr_vl"]["vl_rec_server_url"] == "http://127.0.0.1:8118/v1"
    lighton = payload["vision_language_models"]["lightonocr2_1b_lemonade_vllm"]
    assert lighton["base_url"] == "http://127.0.0.1:13305/api/v1"
    assert lighton["model"] == "user.LightOnOCR-2-1B-vLLM"
    assert lighton["temperature"] == 0.2
    assert lighton["top_p"] == 0.9
    assert lighton["lemonade"]["recipe"] == "vllm"
    assert lighton["lemonade"]["backend"] == "rocm"
    assert lighton["lemonade"]["ctx_size"] == 16384
    assert "--max-num-batched-tokens" in lighton["lemonade"]["vllm_args"]
    assert "16384" in lighton["lemonade"]["vllm_args"]
