#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
INSTALL_TORCH_STACK="${INSTALL_TORCH_STACK:-0}"
INSTALL_OSX="${INSTALL_OSX:-0}"
TMP_FILES=()

cleanup() {
  if ((${#TMP_FILES[@]})); then
    rm -f "${TMP_FILES[@]}"
  fi
}
trap cleanup EXIT

log() {
  echo
  echo "==> $*"
}

die() {
  echo "ERROR: $*" >&2
  exit 1
}

require_path() {
  local path="$1"
  [[ -e "${path}" ]] || die "Missing ${path}"
}

run_pip_install() {
  local install_args=("$@")
  echo "+ ${PYTHON_BIN} -m pip install ${install_args[*]}"
  "${PYTHON_BIN}" -m pip install "${install_args[@]}"
}

filtered_requirements_without_torch_stack() {
  local source_file="$1"
  local filtered_file

  filtered_file="$(mktemp)"
  TMP_FILES+=("${filtered_file}")
  sed -E '/^[[:space:]]*(numpy|torch|torchvision|torchaudio|xformers)([<=>~! ].*)?$/Id' \
    "${source_file}" >"${filtered_file}"
  echo "${filtered_file}"
}

check_required_layout() {
  require_path "${ROOT_DIR}/requirements.txt"
  require_path "${ROOT_DIR}/Grounded-Segment-Anything/requirements.txt"
  require_path "${ROOT_DIR}/Grounded-Segment-Anything/segment_anything/setup.py"
  require_path "${ROOT_DIR}/Grounded-Segment-Anything/GroundingDINO/setup.py"
  require_path "${ROOT_DIR}/hamer/setup.py"
  require_path "${ROOT_DIR}/hamer/third-party/ViTPose/setup.py"
  require_path "${ROOT_DIR}/recognize-anything/requirements.txt"
  require_path "${ROOT_DIR}/recognize-anything/setup.py"
  require_path "${ROOT_DIR}/Video-Depth-Anything/requirements.txt"
  require_path "${ROOT_DIR}/Video-Depth-Anything/run.py"
}

print_environment() {
  log "Using Python environment"
  "${PYTHON_BIN}" --version
  "${PYTHON_BIN}" -m pip --version
  "${PYTHON_BIN}" -c 'import torch; print(f"torch {torch.__version__}, cuda={torch.version.cuda}, available={torch.cuda.is_available()}")' \
    || echo "torch is not installed yet. For CUDA, install the matching PyTorch wheel before running this script."
}

check_required_layout
print_environment

log "[1/5] Installing EgoLoc base dependencies"
run_pip_install -r "${ROOT_DIR}/requirements.txt"

log "[2/5] Installing Grounded-Segment-Anything dependencies"
run_pip_install -r "${ROOT_DIR}/Grounded-Segment-Anything/requirements.txt"
run_pip_install -e "${ROOT_DIR}/Grounded-Segment-Anything/segment_anything"
run_pip_install --no-build-isolation -e "${ROOT_DIR}/Grounded-Segment-Anything/GroundingDINO"

if [[ "${INSTALL_OSX}" == "1" ]]; then
  require_path "${ROOT_DIR}/Grounded-Segment-Anything/grounded-sam-osx/install.sh"
  log "Installing optional Grounded-SAM OSX module"
  (cd "${ROOT_DIR}/Grounded-Segment-Anything/grounded-sam-osx" && bash install.sh)
fi

log "[3/5] Installing HaMeR dependencies"
run_pip_install -e "${ROOT_DIR}/hamer[all]"
run_pip_install -v -e "${ROOT_DIR}/hamer/third-party/ViTPose"

log "[4/5] Installing recognize-anything dependencies"
run_pip_install -r "${ROOT_DIR}/recognize-anything/requirements.txt"
run_pip_install -e "${ROOT_DIR}/recognize-anything"

log "[5/5] Installing Video-Depth-Anything dependencies"
if [[ "${INSTALL_TORCH_STACK}" == "1" ]]; then
  echo "INSTALL_TORCH_STACK=1: installing VDA's pinned numpy/torch/torchvision/xformers versions."
  run_pip_install -r "${ROOT_DIR}/Video-Depth-Anything/requirements.txt"
else
  echo "Preserving the current numpy/PyTorch/CUDA stack."
  echo "Set INSTALL_TORCH_STACK=1 to install VDA's pinned numpy/torch/torchvision/xformers versions."
  vda_requirements="$(filtered_requirements_without_torch_stack "${ROOT_DIR}/Video-Depth-Anything/requirements.txt")"
  run_pip_install -r "${vda_requirements}"
fi

# EgoLoc's 3D pipeline imports open3d directly; it is not listed by VDA.
run_pip_install open3d

echo "All requested project dependencies have been installed."
