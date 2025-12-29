import math
import os
import sys
import csv
import numpy as np
import argparse
from datetime import datetime
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
import torch.multiprocessing as mp
from tqdm import tqdm

# 动态导入上一级目录的 regularization.py
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from regularization import get_method_config, SAM, ASAM, SAMPa, SSAM, CPR

# =============================================================================
# 0. 基础组件 setup
# =============================================================================

def setup_torch_runtime(args):
    if args.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    try:
        # 禁用 Flash Attention 以确保二阶导或复杂梯度操作的兼容性
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    except Exception:
        pass

class PatchEmbed(nn.Module):
    def __init__(self, img_size=64, patch_size=8, in_chans=1, embed_dim=384):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=6, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class ViTSmall(nn.Module):
    def __init__(self, img_size=64, patch_size=8, in_chans=1, num_classes=10, 
                 embed_dim=384, depth=12, num_heads=6, mlp_ratio=4., drop_rate=0.):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.layers = nn.ModuleList()
        for _ in range(depth):
            self.layers.append(nn.ModuleDict({
                'norm1': nn.LayerNorm(embed_dim),
                'attn': Attention(embed_dim, num_heads, attn_drop=drop_rate, proj_drop=drop_rate),
                'norm2': nn.LayerNorm(embed_dim),
                'mlp': MLP(embed_dim, int(embed_dim * mlp_ratio), drop=drop_rate)
            }))
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        nn.init.trunc_normal_(self.pos_embed, std=.02)
        nn.init.trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        for layer in self.layers:
            x_norm = layer['norm1'](x)
            x = x + layer['attn'](x_norm)
            x_norm = layer['norm2'](x)
            x = x + layer['mlp'](x_norm)
        x = self.norm(x)
        return self.head(x[:, 0])

def get_mnist_data(root='./data', train_split=0.5):
    """Load MNIST and split train set"""
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    full_train = MNIST(root=root, train=True, download=True, transform=transform)
    test_set = MNIST(root=root, train=False, download=True, transform=transform)
    train_size = int(len(full_train) * train_split)
    train_set, _ = torch.utils.data.random_split(full_train, [train_size, len(full_train) - train_size])
    return train_set, test_set

# =============================================================================
# 1. 训练工作进程
# =============================================================================

def train_worker(config):
    method_name, gpu_id, args, output_dir = config
    
    # 绑定 GPU
    device = torch.device(f"cuda:{gpu_id}")
    torch.cuda.set_device(gpu_id)
    setup_torch_runtime(args)
    
    # 固定种子
    seed = 2025
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    print(f"🚀 [Start] {method_name} on GPU {gpu_id}")

    # 数据准备 (MNIST)
    train_set, test_set = get_mnist_data(root=args.data_root, train_split=0.5)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=2048, shuffle=False, num_workers=0)

    # 模型初始化
    model = ViTSmall(img_size=64, patch_size=8, in_chans=1, num_classes=10).to(device)
    
    # 权重放大
    if args.weight_scale > 1.0:
        with torch.no_grad():
            for param in model.parameters():
                param.mul_(args.weight_scale)
    
    # === 获取正则化配置 ===
    # args.device 用于 DeMoss hook 的 tensor 创建
    args.device = device 
    optimizer, loss_fn, hooks = get_method_config(method_name, model, args)
    
    # 识别优化器类型
    is_sam_type = isinstance(optimizer, (SAM, ASAM, SAMPa, SSAM))
    is_llc_noise = (method_name == "LLCNoise")

    metrics = {'steps': [], 'train_acc': [], 'test_acc': [], 'train_loss': []}
    steps = 0
    
    pbar = tqdm(total=args.budget, desc=f"{method_name[:10]}", position=gpu_id)
    
    while steps < args.budget:
        for inputs, targets in train_loader:
            model.train()
            inputs, targets = inputs.to(device), targets.to(device)
            
            # ==========================
            # 训练步逻辑分支
            # ==========================
            
            if is_sam_type:
                # --- SAM Family Logic (2-Step) ---
                
                # Step 1: Compute Gradients & Perturb
                optimizer.zero_grad() # Manual zero_grad
                logits = model(inputs)
                loss = loss_fn(logits, targets)
                loss.backward()
                optimizer.first_step(zero_grad=True) 
                
                # Step 2: Compute Gradients at Perturbed State & Update
                logits_2 = model(inputs)
                loss_2 = loss_fn(logits_2, targets)
                loss_2.backward()
                optimizer.second_step(zero_grad=True)
                
                current_loss = loss.item()

            elif is_llc_noise:
                # --- LLCNoise: Two-Step Flow (需要先计算梯度再加噪声) ---
                
                # Step 1: 正常前向后向，计算梯度用于 LLC 估计
                optimizer.zero_grad(set_to_none=True)
                logits = model(inputs)
                loss = loss_fn(logits, targets)
                loss.backward()
                
                # Step 2: 基于梯度计算 LLC，添加噪声
                hooks.pre_step_noise()
                
                # Step 3: 用带噪声的参数再做前向后向
                optimizer.zero_grad(set_to_none=True)
                logits_noisy = model(inputs)
                loss_noisy = loss_fn(logits_noisy, targets)
                loss_noisy.backward()
                
                # Step 4: 恢复参数
                hooks.post_step_restore()
                
                # Step 5: 更新参数
                optimizer.step()
                current_loss = loss.item()

            else:
                # --- Standard Flow (AdamW, CPR, LogitNorm, SD, DeMoss) ---
                
                optimizer.zero_grad(set_to_none=True)
                
                # Hook: Pre-step Noise (DeMoss)
                if hooks: hooks.pre_step_noise()
                
                logits = model(inputs)
                
                # Loss Calculation (Handles LogitNorm/SD wrappers)
                loss = loss_fn(logits, targets)
                
                # Hook: Add Spectral Penalty (DeMoss)
                if hooks: loss += hooks.spectral_entropy_loss()
                
                loss.backward()
                
                # Hook: Restore Weights (DeMoss)
                if hooks: hooks.post_step_restore()
                
                optimizer.step()
                current_loss = loss.item()
            
            steps += 1
            pbar.update(1)

            # ==========================
            # 评估
            # ==========================
            if steps % 100 == 0:
                model.eval()
                with torch.no_grad():
                    # Quick Train Acc (using current batch)
                    tr_acc = (logits.argmax(-1) == targets).float().mean().item()
                    
                    # Full Test Acc
                    te_acc_list = []
                    for test_inputs, test_targets in test_loader:
                        test_inputs = test_inputs.to(device)
                        out = model(test_inputs)
                        te_acc_list.append((out.argmax(-1) == test_targets.to(device)).float().mean().item())
                    te_acc = np.mean(te_acc_list)
                
                metrics['steps'].append(steps)
                metrics['train_acc'].append(tr_acc)
                metrics['test_acc'].append(te_acc)
                metrics['train_loss'].append(current_loss)
                
                pbar.set_postfix({'Test': f'{te_acc:.3f}', 'Train': f'{tr_acc:.3f}'})
                
                # Grokking Check (Early Stop)
                # 判定标准：训练集完美拟合且测试集高精度
                if te_acc > 0.995 and steps > 1000:
                    if len(metrics['test_acc']) > 5 and min(metrics['test_acc'][-5:]) > 0.99:
                        print(f"\n[{method_name}] Grokked at step {steps}!")
                        pbar.close()
                        return (method_name, metrics)
            
            if steps >= args.budget:
                break
                
    pbar.close()
    return (method_name, metrics)

# =============================================================================
# 2. 绘图与主控
# =============================================================================

def plot_all_results(results_map, save_dir):
    print("\n[Plotting] Generating comprehensive comparison...")
    plt.style.use('seaborn-v0_8-paper')
    
    # 分组颜色映射
    method_groups = {
        "Baseline": ["L2_Only", "SAM", "ASAM", "LogitNorm", "SpectralDecoupling"],
        "SOTA": ["SAMPa", "CPR"],
        "Ours": ["LLCNoise"]
    }
    
    # 颜色生成
    colors = {}
    cmap_base = plt.get_cmap("Blues")
    cmap_sota = plt.get_cmap("Greens")
    
    for i, m in enumerate(method_groups["Baseline"]):
        colors[m] = cmap_base(0.3 + i * 0.12) # 浅蓝到深蓝
    for i, m in enumerate(method_groups["SOTA"]):
        colors[m] = cmap_sota(0.4 + i * 0.2) # 浅绿到深绿
    colors["LLCNoise"] = "red" # 红色突出 Ours

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    for name, data in results_map.items():
        steps = data['steps']
        acc = data['test_acc']
        
        # 简单平滑曲线
        if len(acc) > 5:
            acc_smooth = np.convolve(acc, np.ones(5)/5, mode='same')
        else:
            acc_smooth = acc
            
        c = colors.get(name, 'gray')
        lw = 2.5 if "Ours" in name else 1.5
        alpha = 1.0 if "Ours" in name else 0.8
        
        ax.plot(steps, acc_smooth, label=name, color=c, linewidth=lw, alpha=alpha)
    
    ax.set_title("Grokking Speed Comparison: 8 Methods (MNIST Task)")
    ax.set_xlabel("Steps")
    ax.set_ylabel("Test Accuracy")
    ax.set_xscale("log")
    ax.axhline(0.99, color='black', linestyle=':', alpha=0.3, label="Grokking Threshold")
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "final_comparison_8_methods.png"), dpi=200)
    print(f"✅ Plot saved to {os.path.join(save_dir, 'final_comparison_8_methods.png')}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--tf32", type=int, default=1)
    parser.add_argument("--weight_scale", type=float, default=3.0)
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--results_base", type=str, default="/data/zjj/test/results/final_comparison_mnist")
    args = parser.parse_args()

    os.makedirs(args.results_base, exist_ok=True)
    
    print("="*80)
    print(" 🚀 Final Grokking Showdown: 8 Regularization Methods")
    print(f"    Task: MNIST Classification")
    print(f"    Weight Scale: {args.weight_scale}x")
    print("="*80)

    # 定义 8 个任务
    method_list = [
        "L2_Only",                                         # Pure Baseline
        "SAM", "ASAM", "LogitNorm", "SpectralDecoupling", # Classic
        "SAMPa", "CPR", "LLCNoise"                        # SOTA & Ours
    ]
    
    tasks = []
    available_gpus = torch.cuda.device_count()
    if available_gpus < len(method_list):
        print(f"⚠️ Warning: Only {available_gpus} GPUs found. Some tasks will share GPUs sequentially.")
    
    for i, method in enumerate(method_list):
        gpu_id = i % available_gpus # 循环分配
        tasks.append((method, gpu_id, args, args.results_base))

    # 并行启动
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=min(len(tasks), available_gpus)) as pool:
        results = pool.map(train_worker, tasks)
    
    # 整理结果
    results_map = {res[0]: res[1] for res in results}
    
    # 绘图
    plot_all_results(results_map, args.results_base)

    # 保存数据
    for name, data in results_map.items():
        csv_path = os.path.join(args.results_base, f"{name}_metrics.csv")
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(data.keys())
            writer.writerows(zip(*data.values()))
        print(f"Data saved to {csv_path}")

if __name__ == "__main__":
    main()