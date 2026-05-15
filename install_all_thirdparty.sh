#!/usr/bin/env bash

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

ENV_NAME="${CONDA_ENV_NAME:-egoloc}"
PYTHON_VERSION="${CONDA_PYTHON_VERSION:-3.10}"

echo "==> Create / use conda env: $ENV_NAME"

if command -v conda >/dev/null 2>&1; then
    if ! conda env list | awk '{print $1}' | grep -Fxq "$ENV_NAME"; then
        conda create -n "$ENV_NAME" python="$PYTHON_VERSION" pip -y
    fi

    eval "$(conda shell.bash hook)"
    conda activate "$ENV_NAME"
else
    echo "conda not found, using current python environment"
fi

echo "==> Python:"
python --version
pip --version

echo "==> Upgrade basic tools"
pip install -U pip setuptools wheel

echo "==> Install main requirements"
pip install -r requirements.txt || true

echo "==> Install Grounded-Segment-Anything"
pip install -r Grounded-Segment-Anything/requirements.txt || true
pip install -e Grounded-Segment-Anything/segment_anything || true
pip install --no-build-isolation -e Grounded-Segment-Anything/GroundingDINO || true

echo "==> Install HaMeR"
cd "$ROOT_DIR/hamer"
git submodule update --init --recursive || true
pip install --no-build-isolation -e . || true

if [ -d third-party/ViTPose ]; then
    pip install --no-build-isolation -e third-party/ViTPose || true
fi

echo "==> Install recognize-anything"
cd "$ROOT_DIR/recognize-anything"
pip install -r requirements.txt || true
pip install -e . || true

echo "==> Install Video-Depth-Anything"
cd "$ROOT_DIR/Video-Depth-Anything"
pip install -r requirements.txt || true

pip install transformers==4.30.2
pip install --force-reinstall torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu121
pip install --force-reinstall numpy==1.26.4

python -m pip install --no-cache-dir xformers==0.0.27.post2  --index-url https://download.pytorch.org/whl/cu121
echo "==> Done"