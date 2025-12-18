# 1. 屏蔽 P2P (防止 4090 多卡死锁)
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

# 2. 设置缓存路径 (防止炸系统盘)
export HF_HOME="/data/zjj/.cache/huggingface"

# # 3. 使用 torchrun 启动 8 卡并行
# torchrun --nproc_per_node=8 train_world_model.py \
#     --precision bf16 \
#     --use_flash_attn \
#     --compile