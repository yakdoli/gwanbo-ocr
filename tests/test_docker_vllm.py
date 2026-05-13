from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gwanbo_ocr.docker_vllm import (
    VllmDockerConfig,
    build_vllm_docker_command,
    build_vllm_docker_command_string,
    command_to_string,
    mi300x_vllm_config,
)


class TestVllmDockerConfig:
    def test_base_url_uses_configured_port(self) -> None:
        cfg = VllmDockerConfig(model="test-model", port=9000)
        assert cfg.base_url == "http://localhost:9000/v1"

    def test_default_base_url(self) -> None:
        cfg = VllmDockerConfig(model="m")
        assert cfg.base_url == "http://localhost:8000/v1"


class TestBuildVllmDockerCommand:
    def _cmd(self, **kwargs: Any) -> list[str]:
        return build_vllm_docker_command(model="test-model", **kwargs)

    def test_starts_with_docker_run(self) -> None:
        cmd = self._cmd()
        assert cmd[:2] == ["docker", "run"]

    def test_includes_image_name(self) -> None:
        cmd = self._cmd()
        assert "vllm/vllm-openai-rocm:latest" in cmd

    def test_includes_model_flag(self) -> None:
        cmd = self._cmd()
        idx = cmd.index("--model")
        assert cmd[idx + 1] == "test-model"

    def test_network_host_added_by_default(self) -> None:
        cmd = self._cmd()
        assert "--network=host" in cmd

    def test_detach_flag_added_by_default(self) -> None:
        cmd = self._cmd()
        assert "-d" in cmd

    def test_container_name_included(self) -> None:
        cmd = build_vllm_docker_command(model="m", container_name="my-vllm")
        assert "--name" in cmd
        assert cmd[cmd.index("--name") + 1] == "my-vllm"

    def test_enforce_eager_flag(self) -> None:
        cmd = self._cmd(enforce_eager=True)
        assert "--enforce-eager" in cmd

    def test_enforce_eager_not_present_by_default(self) -> None:
        cmd = self._cmd(enforce_eager=False)
        assert "--enforce-eager" not in cmd

    def test_disable_custom_all_reduce_flag(self) -> None:
        cmd = self._cmd(disable_custom_all_reduce=True)
        assert "--disable-custom-all-reduce" in cmd

    def test_tensor_parallel_size_flag(self) -> None:
        cmd = self._cmd(tensor_parallel_size=4)
        idx = cmd.index("--tensor-parallel-size")
        assert cmd[idx + 1] == "4"

    def test_max_model_len_flag(self) -> None:
        cmd = self._cmd(max_model_len=131072)
        idx = cmd.index("--max-model-len")
        assert cmd[idx + 1] == "131072"

    def test_gpu_memory_utilization_flag(self) -> None:
        cmd = self._cmd(gpu_memory_utilization=0.95)
        idx = cmd.index("--gpu-memory-utilization")
        assert cmd[idx + 1] == "0.95"

    def test_raises_without_model_and_no_config(self) -> None:
        with pytest.raises(TypeError, match="model"):
            build_vllm_docker_command()

    def test_config_object_overrides_applied(self) -> None:
        cfg = VllmDockerConfig(model="base-model")
        cmd = build_vllm_docker_command(cfg, model="override-model")
        idx = cmd.index("--model")
        assert cmd[idx + 1] == "override-model"

    def test_ipc_host_included(self) -> None:
        cmd = self._cmd(ipc="host")
        assert "--ipc" in cmd
        assert cmd[cmd.index("--ipc") + 1] == "host"

    def test_rocm_devices_included(self) -> None:
        cmd = self._cmd()
        assert "--device" in cmd
        flat = " ".join(cmd)
        assert "/dev/kfd" in flat

    def test_trust_remote_code_flag(self) -> None:
        cmd = self._cmd(trust_remote_code=True)
        assert "--trust-remote-code" in cmd


class TestCommandToString:
    def test_shell_escapes_spaces(self) -> None:
        import shlex

        cmd = ["docker", "run", "--name", "my container"]
        result = command_to_string(cmd)
        # shlex.join quotes the token; round-tripping must recover the original list
        assert shlex.split(result) == cmd

    def test_simple_command_roundtrips(self) -> None:
        cmd = ["docker", "run", "-d", "image:tag"]
        result = command_to_string(cmd)
        assert "docker" in result
        assert "image:tag" in result


class TestMi300xVllmConfig:
    def test_returns_vllm_docker_config(self) -> None:
        cfg = mi300x_vllm_config("my-model")
        assert isinstance(cfg, VllmDockerConfig)
        assert cfg.model == "my-model"

    def test_overrides_applied(self) -> None:
        cfg = mi300x_vllm_config("m", port=9999)
        assert cfg.port == 9999


class TestBuildVllmDockerCommandString:
    def test_returns_string(self) -> None:
        result = build_vllm_docker_command_string(model="m")
        assert isinstance(result, str)
        assert "docker" in result
        assert "run" in result
