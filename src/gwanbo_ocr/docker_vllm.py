"""Docker command configuration for running vLLM on MI300X/ROCm hosts."""

from __future__ import annotations

import shlex
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

ROCM_VLLM_IMAGE = "vllm/vllm-openai-rocm:latest"
DEFAULT_VLLM_PORT = 8000
DEFAULT_VLLM_BASE_URL = f"http://localhost:{DEFAULT_VLLM_PORT}/v1"


@dataclass(frozen=True)
class VllmDockerConfig:
    """Configuration for an OpenAI-compatible vLLM Docker server."""

    model: str
    image: str = ROCM_VLLM_IMAGE
    container_name: str | None = "gwanbo-vllm"
    detach: bool = True
    network_host: bool = True
    host: str = "0.0.0.0"
    port: int = DEFAULT_VLLM_PORT
    container_port: int = DEFAULT_VLLM_PORT
    tensor_parallel_size: int | None = None
    gpu_memory_utilization: float | None = 0.90
    max_model_len: int | None = None
    max_num_seqs: int | None = None
    dtype: str | None = "auto"
    quantization: str | None = None
    tokenizer: str | None = None
    hf_config_path: str | None = None
    served_model_name: str | None = None
    trust_remote_code: bool = False
    enforce_eager: bool = False
    disable_custom_all_reduce: bool = False
    hf_cache_dir: str | Path | None = "~/.cache/huggingface"
    container_hf_cache_dir: str = "/root/.cache/huggingface"
    devices: Sequence[str] = ("/dev/kfd", "/dev/dri")
    group_add: Sequence[str] = ("video",)
    cap_add: Sequence[str] = ("SYS_PTRACE",)
    security_opt: Sequence[str] = ("seccomp=unconfined",)
    ipc: str | None = "host"
    shm_size: str | None = None
    env: Mapping[str, str] = field(default_factory=dict)
    pass_env: Sequence[str] = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN")
    volumes: Sequence[tuple[str | Path, str]] = field(default_factory=tuple)
    extra_docker_args: Sequence[str] = field(default_factory=tuple)
    extra_vllm_args: Sequence[str] = field(default_factory=tuple)

    @property
    def base_url(self) -> str:
        """OpenAI-compatible base URL exposed on the host."""

        return f"http://localhost:{self.port}/v1"


def mi300x_vllm_config(model: str, **overrides: Any) -> VllmDockerConfig:
    """Return a vLLM Docker config with MI300X/ROCm defaults."""

    return VllmDockerConfig(model=model, **overrides)


def build_vllm_docker_command(
    config: VllmDockerConfig | None = None,
    **overrides: Any,
) -> list[str]:
    """Build a ``docker run`` command for the configured vLLM server."""

    if config is None:
        if "model" not in overrides:
            raise TypeError("model is required when config is not provided")
        config = VllmDockerConfig(**overrides)  # type: ignore[arg-type]
    elif overrides:
        config = replace(config, **overrides)

    command: list[str] = ["docker", "run"]
    if config.detach:
        command.append("-d")
    else:
        command.extend(["--rm", "-it"])

    if config.container_name:
        command.extend(["--name", config.container_name])

    for device in config.devices:
        command.extend(["--device", device])

    for group in config.group_add:
        command.extend(["--group-add", group])

    for capability in config.cap_add:
        command.extend(["--cap-add", capability])

    for option in config.security_opt:
        command.extend(["--security-opt", option])

    if config.network_host:
        command.append("--network=host")
    elif config.port != config.container_port or config.port:
        command.extend(["-p", f"{config.port}:{config.container_port}"])

    if config.ipc:
        command.extend(["--ipc", config.ipc])

    if config.shm_size:
        command.extend(["--shm-size", config.shm_size])

    for name in config.pass_env:
        command.extend(["--env", name])

    for name, value in config.env.items():
        command.extend(["--env", f"{name}={value}"])

    if config.hf_cache_dir is not None:
        host_cache = str(Path(config.hf_cache_dir).expanduser())
        command.extend(["-v", f"{host_cache}:{config.container_hf_cache_dir}"])

    for host_path, container_path in config.volumes:
        command.extend(["-v", f"{Path(host_path).expanduser()}:{container_path}"])

    command.extend(config.extra_docker_args)
    command.append(config.image)
    command.extend(_build_vllm_server_args(config))
    return command


def _build_vllm_server_args(config: VllmDockerConfig) -> list[str]:
    args: list[str] = [
        "--host",
        config.host,
        "--port",
        str(config.container_port),
        "--model",
        config.model,
    ]

    if config.tensor_parallel_size is not None:
        args.extend(["--tensor-parallel-size", str(config.tensor_parallel_size)])

    if config.dtype:
        args.extend(["--dtype", config.dtype])

    if config.quantization:
        args.extend(["--quantization", config.quantization])

    if config.tokenizer:
        args.extend(["--tokenizer", config.tokenizer])

    if config.hf_config_path:
        args.extend(["--hf-config-path", config.hf_config_path])

    if config.gpu_memory_utilization is not None:
        args.extend(["--gpu-memory-utilization", str(config.gpu_memory_utilization)])

    if config.max_model_len is not None:
        args.extend(["--max-model-len", str(config.max_model_len)])

    if config.max_num_seqs is not None:
        args.extend(["--max-num-seqs", str(config.max_num_seqs)])

    if config.served_model_name:
        args.extend(["--served-model-name", config.served_model_name])

    if config.trust_remote_code:
        args.append("--trust-remote-code")

    if config.enforce_eager:
        args.append("--enforce-eager")

    if config.disable_custom_all_reduce:
        args.append("--disable-custom-all-reduce")

    args.extend(config.extra_vllm_args)
    return args


def command_to_string(command: Sequence[str]) -> str:
    """Return a shell-escaped command string for display or scripts."""

    return shlex.join(command)


def build_vllm_docker_command_string(
    config: VllmDockerConfig | None = None,
    **overrides: object,
) -> str:
    """Build a shell-escaped ``docker run`` command string."""

    return command_to_string(build_vllm_docker_command(config, **overrides))


# Friendly aliases for likely call sites.
build_docker_run_command = build_vllm_docker_command
build_docker_run_command_string = build_vllm_docker_command_string
docker_command = build_vllm_docker_command
docker_command_string = build_vllm_docker_command_string
