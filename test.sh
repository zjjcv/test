CANDIDATES=(/data /workspace /mnt /autodl-tmp /root/autodl-tmp /autodl-pub /autodl-pub/data)

for p in "${CANDIDATES[@]}"; do
  if [ -d "$p" ]; then
    echo "== $p =="
    df -h "$p" | tail -n 1
    if ( touch "$p/.wtest" 2>/dev/null && rm -f "$p/.wtest" ); then
      echo "WRITE: YES"
    else
      echo "WRITE: NO"
    fi
    echo
  fi
done



BASE=/root/autodl-tmp

# 1) 创建目录
mkdir -p \
  "$BASE/.tmp" \
  "$BASE/.cache" \
  "$BASE/.cache/pip" \
  "$BASE/.cache/huggingface" \
  "$BASE/.cache/torch" \
  "$BASE/.cache/triton" \
  "$BASE/.cache/numba" \
  "$BASE/wandb" \
  "$BASE/torch_extensions" \
  "$BASE/conda/envs" \
  "$BASE/conda/pkgs"

# 2) 写入 ~/.bashrc（带标记，避免重复写）
MARK_BEGIN="# === AUTODL_PERSIST_BEGIN ==="
MARK_END="# === AUTODL_PERSIST_END ==="

grep -q "$MARK_BEGIN" ~/.bashrc 2>/dev/null || cat >> ~/.bashrc <<EOF
$MARK_BEGIN
# Base writable disk for caches/envs
export AUTODL_BASE=$BASE

# Temp/build dirs (avoid writing to overlay /tmp)
export TMPDIR=\$AUTODL_BASE/.tmp

# Generic cache root (many libs respect this)
export XDG_CACHE_HOME=\$AUTODL_BASE/.cache

# pip
export PIP_CACHE_DIR=\$AUTODL_BASE/.cache/pip

# HuggingFace
export HF_HOME=\$AUTODL_BASE/.cache/huggingface
export TRANSFORMERS_CACHE=\$AUTODL_BASE/.cache/huggingface
export HF_DATASETS_CACHE=\$AUTODL_BASE/.cache/huggingface
export HUGGINGFACE_HUB_CACHE=\$AUTODL_BASE/.cache/huggingface

# PyTorch
export TORCH_HOME=\$AUTODL_BASE/.cache/torch
export TORCH_EXTENSIONS_DIR=\$AUTODL_BASE/torch_extensions

# Triton / Numba (常见会写系统盘的缓存)
export TRITON_CACHE_DIR=\$AUTODL_BASE/.cache/triton
export NUMBA_CACHE_DIR=\$AUTODL_BASE/.cache/numba

# W&B
export WANDB_DIR=\$AUTODL_BASE/wandb
$MARK_END
EOF

# 3) 让当前会话立刻生效
source ~/.bashrc

# 如果 conda 存在才执行（避免无 conda 报错）
if command -v conda >/dev/null 2>&1; then
  # 清掉旧的 envs_dirs/pkgs_dirs 配置（若存在）
  conda config --remove-key envs_dirs 2>/dev/null || true
  conda config --remove-key pkgs_dirs 2>/dev/null || true

  # 先加入数据盘目录（作为默认优先项）
  conda config --add envs_dirs /root/autodl-tmp/conda/envs
  conda config --add pkgs_dirs /root/autodl-tmp/conda/pkgs

  # 再保留 conda base 的默认目录作为备选（避免某些工具假设存在）
  CONDA_BASE="$(conda info --base)"
  conda config --add envs_dirs "$CONDA_BASE/envs" || true
  conda config --add pkgs_dirs "$CONDA_BASE/pkgs" || true
fi


