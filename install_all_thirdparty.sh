#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH="${ROOT_DIR}/install_all_thirdparty.sh"
PYTHON_BIN="${PYTHON_BIN:-python}"
INSTALL_OSX="${INSTALL_OSX:-0}"
BOOTSTRAP_CONDA="${BOOTSTRAP_CONDA:-1}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-egoloc}"
CONDA_PYTHON_VERSION="${CONDA_PYTHON_VERSION:-3.10}"
PIP_CACHE_DIR="${PIP_CACHE_DIR:-/tmp/egoloc-pip-cache}"
VERBOSE_INSTALL="${VERBOSE_INSTALL:-1}"
PIP_PROGRESS_BAR="${PIP_PROGRESS_BAR:-raw}"
TMP_FILES=()
MAIN_REQUIREMENTS="${ROOT_DIR}/requirements.txt"
export PIP_CACHE_DIR
export PIP_PROGRESS_BAR

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

conda_env_exists() {
  conda env list | awk '{print $1}' | grep -Fxq "${CONDA_ENV_NAME}"
}

ensure_conda_pip() {
  log "Ensuring env-local pip exists in '${CONDA_ENV_NAME}'"
  conda run --no-capture-output -n "${CONDA_ENV_NAME}" env PYTHONNOUSERSITE=1 python -m ensurepip --upgrade
  conda run --no-capture-output -n "${CONDA_ENV_NAME}" env PYTHONNOUSERSITE=1 python -m pip install 'setuptools<81' wheel
}

ensure_conda_environment() {
  if [[ "${BOOTSTRAP_CONDA}" != "1" ]]; then
    export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"
    return
  fi

  if [[ "${CONDA_BOOTSTRAPPED:-0}" == "1" || "${CONDA_DEFAULT_ENV:-}" == "${CONDA_ENV_NAME}" ]]; then
    PYTHON_BIN="${PYTHON_BIN:-python}"
    export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"
    return
  fi

  command -v conda >/dev/null 2>&1 || die "conda was not found. Install Conda/Mamba or rerun with BOOTSTRAP_CONDA=0 inside your own Python environment."

  if ! conda_env_exists; then
    log "Creating conda env '${CONDA_ENV_NAME}' with Python ${CONDA_PYTHON_VERSION}"
    conda create -n "${CONDA_ENV_NAME}" "python=${CONDA_PYTHON_VERSION}" pip -y
  else
    log "Using existing conda env '${CONDA_ENV_NAME}'"
  fi

  ensure_conda_pip

  log "Re-running installer inside conda env '${CONDA_ENV_NAME}'"
  CONDA_BOOTSTRAPPED=1 PYTHON_BIN=python PYTHONNOUSERSITE=1 conda run --no-capture-output -n "${CONDA_ENV_NAME}" bash "${SCRIPT_PATH}" "$@"
  exit $?
}

run_pip_install() {
  local install_args=("$@")
  local pip_args=()

  if [[ "${VERBOSE_INSTALL}" == "1" ]]; then
    pip_args+=("-vv")
  fi

  pip_args+=("--progress-bar" "${PIP_PROGRESS_BAR}")

  echo "+ ${PYTHON_BIN} -m pip install ${pip_args[*]} ${install_args[*]}"
  "${PYTHON_BIN}" -m pip install "${pip_args[@]}" "${install_args[@]}"
}

filtered_requirements() {
  local source_file="$1"
  local mode="$2"
  local filtered_file

  filtered_file="$(mktemp)"
  TMP_FILES+=("${filtered_file}")

  case "${mode}" in
    torch)
      sed -n -E '/^[[:space:]]*(torch|torchvision|torchaudio)([<=>~! ].*)?$/Ip' \
        "${source_file}" >"${filtered_file}"
      ;;
    regular)
      sed -E \
        -e '/^[[:space:]]*(torch|torchvision|torchaudio)([<=>~! ].*)?$/Id' \
        -e '/^[[:space:]]*mmcv([<=>~! ].*)?$/Id' \
        -e '/^[[:space:]]*(-e|--editable)[[:space:]]+/Id' \
        -e '/@[[:space:]]*git\+/Id' \
        "${source_file}" >"${filtered_file}"
      ;;
    source-builds)
      sed -n -E \
        -e '/^[[:space:]]*mmcv([<=>~! ].*)?$/Ip' \
        -e '/^[[:space:]]*(-e|--editable)[[:space:]]+/Ip' \
        -e '/@[[:space:]]*git\+/Ip' \
        "${source_file}" >"${filtered_file}"
      ;;
    *)
      die "Unknown requirements filter mode: ${mode}"
      ;;
  esac

  echo "${filtered_file}"
}

check_required_layout() {
  require_path "${MAIN_REQUIREMENTS}"
  require_path "${ROOT_DIR}/Grounded-Segment-Anything/requirements.txt"
  require_path "${ROOT_DIR}/Grounded-Segment-Anything/segment_anything/setup.py"
  require_path "${ROOT_DIR}/Grounded-Segment-Anything/GroundingDINO/setup.py"
  require_path "${ROOT_DIR}/hamer/setup.py"
  require_path "${ROOT_DIR}/recognize-anything/requirements.txt"
  require_path "${ROOT_DIR}/recognize-anything/setup.py"
  require_path "${ROOT_DIR}/Video-Depth-Anything/requirements.txt"
  require_path "${ROOT_DIR}/Video-Depth-Anything/run.py"
}

ensure_hamer_submodules() {
  if [[ ! -e "${ROOT_DIR}/hamer/third-party/ViTPose/setup.py" ]]; then
    log "Initializing HaMeR submodules"
    git -C "${ROOT_DIR}/hamer" submodule update --init --recursive third-party/ViTPose
  fi

  require_path "${ROOT_DIR}/hamer/third-party/ViTPose/setup.py"
}

print_environment() {
  log "Using Python environment"
  "${PYTHON_BIN}" --version
  "${PYTHON_BIN}" -m pip --version
  "${PYTHON_BIN}" -c 'import torch; print(f"torch {torch.__version__}, cuda={torch.version.cuda}, available={torch.cuda.is_available()}")' \
    || echo "torch is not installed yet. Installing the pinned PyTorch stack before source builds."
}

ensure_conda_environment "$@"
check_required_layout
ensure_hamer_submodules
print_environment

torch_requirements="$(filtered_requirements "${MAIN_REQUIREMENTS}" torch)"
regular_requirements="$(filtered_requirements "${MAIN_REQUIREMENTS}" regular)"
source_build_requirements="$(filtered_requirements "${MAIN_REQUIREMENTS}" source-builds)"

log "[1/4] Installing PyTorch stack from requirements.txt"
run_pip_install -r "${torch_requirements}"

log "[2/4] Installing regular dependencies from requirements.txt"
run_pip_install -r "${regular_requirements}"

log "[3/4] Installing source/editable packages from requirements.txt"
run_pip_install --no-build-isolation -r "${source_build_requirements}"

if [[ "${INSTALL_OSX}" == "1" ]]; then
  require_path "${ROOT_DIR}/Grounded-Segment-Anything/grounded-sam-osx/install.sh"
  log "Installing optional Grounded-SAM OSX module"
  (cd "${ROOT_DIR}/Grounded-Segment-Anything/grounded-sam-osx" && bash install.sh)
fi

log "[4/4] Done"
echo "All requested project dependencies have been installed."
