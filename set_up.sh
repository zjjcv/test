#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# setup_conda.sh
# - 在可写数据盘（默认 /root/autodl-tmp）上创建 conda 环境
# - 将 conda pkgs 缓存、pip 缓存、临时编译目录等永久/当前会话指向数据盘
# - 从当前工程目录的 requirements.txt 安装依赖
#
# 用法：
#   bash setup_conda.sh
#   bash setup_conda.sh --python 3.12
#   bash setup_conda.sh --name myenv
#   bash setup_conda.sh --requirements requirements.txt
#   bash setup_conda.sh --base /root/autodl-tmp
#   bash setup_conda.sh --index-url https://pypi.tuna.tsinghua.edu.cn/simple
#
# 说明：
# - 默认环境名为当前目录名
# - 默认 python 版本 3.12
# - 默认 base 盘为 /root/autodl-tmp（你当前机器可写的大盘）
# - 通过 -p 创建环境，确保环境本体不落在 overlay(系统盘)
# - pip 使用清华源（可改）
# ============================================================

PYTHON_VER="3.12"
ENV_NAME=""
REQ_FILE="requirements.txt"
PIP_INDEX_URL="https://pypi.tuna.tsinghua.edu.cn/simple"
BASE_DIR="/root/autodl-tmp"
PERSIST=1  # 1: 写入 ~/.bashrc 永久生效；0: 仅当前会话

while [[ $# -gt 0 ]]; do
  case "$1" in
    --python)
      PYTHON_VER="${2:-}"; shift 2 ;;
    --name|--env)
      ENV_NAME="${2:-}"; shift 2 ;;
    --requirements|--req)
      REQ_FILE="${2:-}"; shift 2 ;;
    --index-url)
      PIP_INDEX_URL="${2:-}"; shift 2 ;;
    --base)
      BASE_DIR="${2:-}"; shift 2 ;;
    --no-persist)
      PERSIST=0; shift 1 ;;
    -h|--help)
      cat <<EOF
Usage:
  bash setup_conda.sh [--python 3.12] [--name myenv] [--requirements requirements.txt]
                      [--base /root/autodl-tmp] [--index-url URL] [--no-persist]
EOF
      exit 0
      ;;
    *)
      echo "Unknown argument: $1"
      exit 1
      ;;
  esac
done

PROJECT_DIR="$(pwd)"
PROJECT_BASENAME="$(basename "${PROJECT_DIR}")"
REQ_PATH="${PROJECT_DIR}/${REQ_FILE}"

if [[ -z "${ENV_NAME}" ]]; then
  ENV_NAME="${PROJECT_BASENAME}"
fi

# --- sanity checks ---
if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: conda not found in PATH. Please install Miniconda/Anaconda or source conda.sh."
  exit 1
fi

if [[ ! -f "${REQ_PATH}" ]]; then
  echo "ERROR: requirements file not found: ${REQ_PATH}"
  exit 1
fi

# 检查 BASE_DIR 是否可写
mkdir -p "${BASE_DIR}" 2>/dev/null || true
if ! ( touch "${BASE_DIR}/.wtest" 2>/dev/null && rm -f "${BASE_DIR}/.wtest" ); then
  echo "ERROR: base dir is not writable: ${BASE_DIR}"
  echo "Hint: choose another writable disk with: df -h && touch <path>/.wtest"
  exit 1
fi

# --- paths on data disk ---
CACHE_ROOT="${BASE_DIR}/.cache"
TMP_ROOT="${BASE_DIR}/.tmp"
PIP_CACHE="${CACHE_ROOT}/pip"
HF_CACHE="${CACHE_ROOT}/huggingface"
TORCH_CACHE="${CACHE_ROOT}/torch"
TRITON_CACHE="${CACHE_ROOT}/triton"
NUMBA_CACHE="${CACHE_ROOT}/numba"
WANDB_DIR_PATH="${BASE_DIR}/wandb"
TORCH_EXT_DIR="${BASE_DIR}/torch_extensions"

CONDA_ENVS_DIR="${BASE_DIR}/conda/envs"
CONDA_PKGS_DIR="${BASE_DIR}/conda/pkgs"
ENV_PREFIX="${CONDA_ENVS_DIR}/${ENV_NAME}"

echo "============================================================"
echo "[1/8] Project:         ${PROJECT_DIR}"
echo "[2/8] Conda env name:  ${ENV_NAME}"
echo "[3/8] Python version:  ${PYTHON_VER}"
echo "[4/8] Requirements:    ${REQ_PATH}"
echo "[5/8] Pip index:       ${PIP_INDEX_URL}"
echo "[6/8] Base dir:        ${BASE_DIR}"
echo "[7/8] Env prefix:      ${ENV_PREFIX}"
echo "============================================================"

# --- create dirs ---
mkdir -p \
  "${TMP_ROOT}" \
  "${PIP_CACHE}" \
  "${HF_CACHE}" \
  "${TORCH_CACHE}" \
  "${TRITON_CACHE}" \
  "${NUMBA_CACHE}" \
  "${WANDB_DIR_PATH}" \
  "${TORCH_EXT_DIR}" \
  "${CONDA_ENVS_DIR}" \
  "${CONDA_PKGS_DIR}"

# --- make conda activate usable in script ---
CONDA_BASE="$(conda info --base)"
# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"

# --- set runtime env vars (current shell) ---
export AUTODL_BASE="${BASE_DIR}"
export TMPDIR="${TMP_ROOT}"
export XDG_CACHE_HOME="${CACHE_ROOT}"
export PIP_CACHE_DIR="${PIP_CACHE}"

export HF_HOME="${HF_CACHE}"
export TRANSFORMERS_CACHE="${HF_CACHE}"
export HF_DATASETS_CACHE="${HF_CACHE}"
export HUGGINGFACE_HUB_CACHE="${HF_CACHE}"

export TORCH_HOME="${TORCH_CACHE}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXT_DIR}"
export TRITON_CACHE_DIR="${TRITON_CACHE}"
export NUMBA_CACHE_DIR="${NUMBA_CACHE}"
export WANDB_DIR="${WANDB_DIR_PATH}"

# --- persist to ~/.bashrc (optional) ---
if [[ "${PERSIST}" == "1" ]]; then
  MARK_BEGIN="# === AUTODL_PERSIST_BEGIN ==="
  MARK_END="# === AUTODL_PERSIST_END ==="

  if ! grep -q "${MARK_BEGIN}" ~/.bashrc 2>/dev/null; then
    cat >> ~/.bashrc <<EOF

${MARK_BEGIN}
export AUTODL_BASE=${BASE_DIR}
export TMPDIR=\$AUTODL_BASE/.tmp
export XDG_CACHE_HOME=\$AUTODL_BASE/.cache
export PIP_CACHE_DIR=\$AUTODL_BASE/.cache/pip

export HF_HOME=\$AUTODL_BASE/.cache/huggingface
export TRANSFORMERS_CACHE=\$AUTODL_BASE/.cache/huggingface
export HF_DATASETS_CACHE=\$AUTODL_BASE/.cache/huggingface
export HUGGINGFACE_HUB_CACHE=\$AUTODL_BASE/.cache/huggingface

export TORCH_HOME=\$AUTODL_BASE/.cache/torch
export TORCH_EXTENSIONS_DIR=\$AUTODL_BASE/torch_extensions
export TRITON_CACHE_DIR=\$AUTODL_BASE/.cache/triton
export NUMBA_CACHE_DIR=\$AUTODL_BASE/.cache/numba

export WANDB_DIR=\$AUTODL_BASE/wandb
${MARK_END}
EOF
    echo "[8/8] Persisted cache/env defaults into ~/.bashrc"
  else
    echo "[8/8] ~/.bashrc already has AUTODL_PERSIST block (skip)"
  fi
else
  echo "[8/8] --no-persist set: will NOT write ~/.bashrc"
fi

# --- configure pip cache dir permanently (pip config) ---
pip config set global.cache-dir "${PIP_CACHE}" >/dev/null 2>&1 || true

# --- configure conda pkgs cache dir to data disk (global conda config) ---
# 注：这会写 ~/.condarc。若你不希望改全局，可注释掉这一段，然后继续使用 -p 方式创建 env。
conda config --remove-key pkgs_dirs 2>/dev/null || true
conda config --add pkgs_dirs "${CONDA_PKGS_DIR}" >/dev/null

# 同时把 envs_dirs 指到数据盘（便于 conda env list 展示）
conda config --remove-key envs_dirs 2>/dev/null || true
conda config --add envs_dirs "${CONDA_ENVS_DIR}" >/dev/null
conda config --add envs_dirs "${CONDA_BASE}/envs" >/dev/null || true

# --- create env on data disk if not exists ---
if [[ -d "${ENV_PREFIX}" ]]; then
  echo "Conda env already exists at: ${ENV_PREFIX} (skip create)"
else
  echo "Creating conda env at: ${ENV_PREFIX}"
  conda create -y -p "${ENV_PREFIX}" "python=${PYTHON_VER}" pip
fi

# --- activate env ---
conda activate "${ENV_PREFIX}"

# --- upgrade pip toolchain ---
echo "Upgrading pip tools..."
python -m pip install --upgrade pip setuptools wheel -i "${PIP_INDEX_URL}"

# --- install requirements ---
echo "Installing dependencies from ${REQ_FILE} ..."
python -m pip install -r "${REQ_PATH}" -i "${PIP_INDEX_URL}"

echo "============================================================"
echo "Done."
echo "Activate later:"
echo "  conda activate ${ENV_PREFIX}"
echo "Key dirs:"
echo "  TMPDIR=${TMPDIR}"
echo "  PIP_CACHE_DIR=${PIP_CACHE_DIR}"
echo "  HF_HOME=${HF_HOME}"
echo "  TORCH_HOME=${TORCH_HOME}"
echo "============================================================"
