#!/bin/bash

# =============================================================================
# Grokking Experiments Runner
# 顺序执行所有实验脚本
# =============================================================================

set -e  # 遇到错误立即退出

# 进入工作目录
cd /root/autodl-tmp/test

echo "========================================================================"
echo "Starting Grokking Experiments"
echo "========================================================================"
echo "Start time: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# =============================================================================
# 1. 运行原始 Transformer 模型 (2层4头128维)
# =============================================================================
echo "========================================================================"
echo "Step 1: Running Original Transformer (2L4H128D)"
echo "========================================================================"
echo "Tasks: 9 modular arithmetic tasks"
echo "Config: 54 experiments (9 tasks × 2 WD × 3 seeds)"
echo "Mode: Single thread"
echo "Started at: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# python src/grokking_multiseed.py

echo ""
echo "✓ Original Transformer completed at: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# =============================================================================
# 2. 运行 GPT-2 Small 模型 (12层12头768维)
# =============================================================================
echo "========================================================================"
echo "Step 2: Running GPT-2 Small (12L12H768D)"
echo "========================================================================"
echo "Tasks: 9 modular arithmetic tasks"
echo "Config: 54 experiments (9 tasks × 2 WD × 3 seeds)"
echo "Mode: 4 parallel workers"
echo "Checkpoints: x-y task at steps 100, 1000, 10000, 100000"
echo "Started at: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

python src/gpt2_grokking.py

echo ""
echo "✓ GPT-2 Small completed at: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# =============================================================================
# 完成
# =============================================================================
echo "========================================================================"
echo "All Experiments Completed!"
echo "========================================================================"
echo "End time: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""
echo "Results saved to:"
echo "  - Original Transformer: /root/autodl-tmp/test/results/data/"
echo "  - GPT-2 Small: /root/autodl-tmp/test/results/data_gpt2_small/"
echo "  - GPT-2 Checkpoints: /root/autodl-tmp/test/results/checkpoints_gpt2_small/"
echo ""
echo "Next steps:"
echo "  1. Run visualization notebook: plot_multiseed_results.ipynb"
echo "  2. Analyze checkpoints for grokking dynamics"
echo "  3. Compare architectures (Original vs GPT-2)"
echo ""

