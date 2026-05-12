#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: scripts/clean_venv_vllm_residue.sh [--apply] [--venv PATH]

Dry-run cleanup helper for removing local vLLM/Torch/Transformers residue from
a Python virtual environment. Pass --apply to uninstall matching distributions
and delete leftover site-packages directories.

Options:
  --apply       Perform deletions. Without this flag, the script only reports.
  --venv PATH   Virtual environment path. Defaults to $VIRTUAL_ENV or .venv.
  -h, --help    Show this help.
USAGE
}

apply=0
venv="${VIRTUAL_ENV:-.venv}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --apply)
      apply=1
      shift
      ;;
    --venv)
      if [[ $# -lt 2 ]]; then
        echo "error: --venv requires a path" >&2
        exit 2
      fi
      venv="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "error: unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ ! -f "$venv/pyvenv.cfg" ]]; then
  echo "error: '$venv' does not look like a Python virtual environment" >&2
  exit 1
fi

python_bin="$venv/bin/python"
if [[ ! -x "$python_bin" ]]; then
  echo "error: '$python_bin' is not executable" >&2
  exit 1
fi

mode="dry run"
if [[ "$apply" -eq 1 ]]; then
  mode="apply"
fi

echo "Mode: $mode"
echo "Virtual environment: $venv"

mapfile -t installed < <("$python_bin" - <<'PY'
from importlib import metadata

targets = {
    "accelerate",
    "amdsmi",
    "bitsandbytes",
    "compressed-tensors",
    "cuda-bindings",
    "cuda-pathfinder",
    "cuda-python",
    "cuda-tile",
    "cuda-toolkit",
    "depyf",
    "fastsafetensors",
    "flash-attn",
    "flashinfer-cubin",
    "flashinfer-python",
    "gguf",
    "llguidance",
    "lm-format-enforcer",
    "mistral-common",
    "model-hosting-container-standards",
    "openai-harmony",
    "outlines-core",
    "quack-kernels",
    "safetensors",
    "sentencepiece",
    "tilelang",
    "tokenizers",
    "torch-c-dlpack-ext",
    "torch",
    "torchaudio",
    "torchvision",
    "transformers",
    "triton",
    "vllm",
    "xgrammar",
    "xformers",
}

for dist in metadata.distributions():
    name = dist.metadata.get("Name", "")
    normalized = name.lower().replace("_", "-")
    if normalized in targets or normalized.startswith("nvidia-"):
        print(name)
PY
)

if [[ "${#installed[@]}" -gt 0 ]]; then
  echo "Matching installed distributions:"
  printf '  %s\n' "${installed[@]}"

  if [[ "$apply" -eq 1 ]]; then
    "$python_bin" -m pip uninstall -y "${installed[@]}"
  else
    echo "Would run: $python_bin -m pip uninstall -y ${installed[*]}"
  fi
else
  echo "No matching installed distributions found."
fi

mapfile -t site_dirs < <("$python_bin" - <<'PY'
import site
import sys
from pathlib import Path

prefix = Path(sys.prefix).resolve()
for raw_path in site.getsitepackages():
    path = Path(raw_path).resolve()
    if path.is_dir() and (path == prefix or prefix in path.parents):
        print(path)
PY
)

patterns=(
  "accelerate"
  "accelerate-*.dist-info"
  "amdsmi"
  "amdsmi-*.dist-info"
  "bitsandbytes"
  "bitsandbytes-*.dist-info"
  "compressed_tensors"
  "compressed_tensors-*.dist-info"
  "cuda"
  "cuda_*.dist-info"
  "depyf"
  "depyf-*.dist-info"
  "fastsafetensors"
  "fastsafetensors-*.dist-info"
  "flashinfer"
  "flashinfer_*.dist-info"
  "flash_attn"
  "flash_attn-*.dist-info"
  "gguf"
  "gguf-*.dist-info"
  "llguidance"
  "llguidance-*.dist-info"
  "lm_format_enforcer"
  "lm_format_enforcer-*.dist-info"
  "mistral_common"
  "mistral_common-*.dist-info"
  "model_hosting_container_standards"
  "model_hosting_container_standards-*.dist-info"
  "nvidia"
  "nvidia_*.dist-info"
  "openai_harmony"
  "openai_harmony-*.dist-info"
  "outlines_core"
  "outlines_core-*.dist-info"
  "quack_kernels"
  "quack_kernels-*.dist-info"
  "safetensors"
  "safetensors-*.dist-info"
  "sentencepiece"
  "sentencepiece-*.dist-info"
  "tilelang"
  "tilelang-*.dist-info"
  "tokenizers"
  "tokenizers-*.dist-info"
  "torch_c_dlpack_ext"
  "torch_c_dlpack_ext-*.dist-info"
  "torch"
  "torch-*.dist-info"
  "torchaudio"
  "torchaudio-*.dist-info"
  "torchvision"
  "torchvision-*.dist-info"
  "transformers"
  "transformers-*.dist-info"
  "triton"
  "triton-*.dist-info"
  "vllm"
  "vllm-*.dist-info"
  "xgrammar"
  "xgrammar-*.dist-info"
  "xformers"
  "xformers-*.dist-info"
)

declare -A residues=()
for site_dir in "${site_dirs[@]}"; do
  for pattern in "${patterns[@]}"; do
    while IFS= read -r -d '' path; do
      residues["$path"]=1
    done < <(find "$site_dir" -mindepth 1 -maxdepth 1 -name "$pattern" -print0)
  done
done

if [[ "${#residues[@]}" -gt 0 ]]; then
  echo "Matching filesystem residue:"
  for path in "${!residues[@]}"; do
    echo "  $path"
  done

  if [[ "$apply" -eq 1 ]]; then
    for path in "${!residues[@]}"; do
      rm -rf -- "$path"
    done
  else
    echo "Would remove the paths listed above."
  fi
else
  echo "No matching filesystem residue found."
fi
