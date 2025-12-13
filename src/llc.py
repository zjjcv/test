import math
import os
import csv
import copy
import numpy as np
from argparse import ArgumentParser
from datetime import datetime

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from torch.optim import Optimizer

# ==========================================
# 0. SGLD 优化器 (用于 LLC 估计的采样)
# ==========================================
class SGLD(Optimizer):
    """
    简化的随机梯度朗之完动力学 (Stochastic Gradient Langevin Dynamics) 优化器
    参考文献: "The Quantization Model of Neural Scaling", Lau et al.
    更新公式: w_{t+1} = w_t - eps/2 * grad + N(0, eps)
    这里的 lr 对应 eps。
    """
    def __init__(self, params, lr=1e-4, noise_scale=1.0):
        defaults = dict(lr=lr, noise_scale=noise_scale)
        super(SGLD, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                
                # 1. 漂移项 (Drift): - epsilon/2 * \nabla L
                p.data.add_(d_p, alpha=-0.5 * group['lr'])
                
                # 2. 扩散项 (Diffusion): + N(0, epsilon)
                # 标准差 sigma = noise_scale * sqrt(lr)
                noise_std = torch.sqrt(torch.tensor(group['lr'])) * group['noise_scale']
                noise = torch.randn_like(p.data) * noise_std
                p.data.add_(noise)

        return loss

# ==========================================
# 1. LLC 估算器 (修正了基准计算逻辑)
# ==========================================
class LLCEstimator:
    def __init__(self, model, criterion, dataloader, device):
        self.model = model
        self.criterion = criterion
        self.dataloader = dataloader 
        self.device = device

    def estimate(self, current_weights_dict, num_draws=40, lr=1e-4, epsilon=1.0):
        """
        计算 LLC 的代理指标 (Energy Gap)。
        """
        # 1. 准备采样模型 (深拷贝，不影响主模型)
        sampling_model = copy.deepcopy(self.model)
        sampling_model.load_state_dict(current_weights_dict)
        
        # 2. 计算准确的基准损失 (Base Loss)
        # [重要修正] 必须遍历整个 DataLoader，否则单 Batch 的随机性会淹没 LLC 信号
        base_loss = self._compute_full_loss(sampling_model)

        # 3. 准备 SGLD
        sampling_model.train() # SGLD 需要梯度
        sgld_optim = SGLD(sampling_model.parameters(), lr=lr, noise_scale=epsilon)
        
        loss_trace = []
        iter_dl = iter(self.dataloader)
        
        # 4. 采样循环 (Sampling Loop)
        for _ in range(num_draws):
            try:
                batch = next(iter_dl)
            except StopIteration:
                iter_dl = iter(self.dataloader)
                batch = next(iter_dl)
            
            batch_x = batch[0].to(self.device)
            # 这里的 input 处理需与主循环一致
            inp, target = batch_x[:, :-1], batch_x[:, -1]
            
            sgld_optim.zero_grad()
            logits = sampling_model(inp)
            loss = self.criterion(logits[:, -1, :], target)
            loss.backward()
            sgld_optim.step()
            
            loss_trace.append(loss.item())

        # 5. 计算 LLC 代理值
        # LLC Proxy = 平均采样能量 - 初始基准能量
        avg_sampling_loss = np.mean(loss_trace)
        llc_proxy = avg_sampling_loss - base_loss
        
        # 清理显存
        del sampling_model
        torch.cuda.empty_cache()
        
        return llc_proxy

    def _compute_full_loss(self, model):
        """计算全量数据的平均 Loss，作为稳定的基准点"""
        model.eval()
        total_loss = 0
        total_count = 0
        with torch.no_grad():
            for batch in self.dataloader:
                batch_x = batch[0].to(self.device)
                inp, target = batch_x[:, :-1], batch_x[:, -1]
                logits = model(inp)
                # reduction='sum' 累加，最后除以总数，或者累加 mean * batch_size
                loss = self.criterion(logits[:, -1, :], target) 
                total_loss += loss.item() * batch_x.size(0)
                total_count += batch_x.size(0)
        return total_loss / total_count

# ==========================================
# 2. 模型定义
# ==========================================
class Block(nn.Module):
    def __init__(self, dim, num_heads, dropout=0.0):
        super().__init__()
        self.ln_1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True, dropout=dropout)
        self.ln_2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, attn_mask=None):
        x_norm = self.ln_1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm, attn_mask=attn_mask, need_weights=False)
        x = x + attn_out
        x = x + self.mlp(self.ln_2(x))
        return x

class Decoder(nn.Module):
    def __init__(self, dim=128, num_layers=2, num_heads=4, num_tokens=97, seq_len=5, dropout=0.0):
        super().__init__()
        self.token_embeddings = nn.Embedding(num_tokens, dim)
        self.position_embeddings = nn.Embedding(seq_len, dim)
        self.layers = nn.ModuleList([Block(dim, num_heads, dropout) for _ in range(num_layers)])
        self.ln_f = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_tokens, bias=False)
        
        mask = torch.triu(torch.full((seq_len, seq_len), float('-inf')), diagonal=1)
        self.register_buffer("causal_mask", mask)
        self.register_buffer("pos_ids", torch.arange(seq_len))

    def forward(self, x):
        h = self.token_embeddings(x) + self.position_embeddings(self.pos_ids[:x.size(1)])
        mask = self.causal_mask[:x.size(1), :x.size(1)]
        for layer in self.layers:
            h = layer(h, attn_mask=mask)
        return self.head(self.ln_f(h))

# ==========================================
# 3. 数据生成
# ==========================================
def get_data(p, eq_token, op_token):
    x = torch.arange(p)
    y = torch.arange(1, p)
    x, y = torch.cartesian_prod(x, y).T
    eq = torch.ones_like(x) * eq_token
    op = torch.ones_like(x) * op_token
    result = (x * y) % p 
    data = torch.stack([x, op, y, eq, result]).T
    return data

# ==========================================
# 4. 训练主程序
# ==========================================
def main(args, run_name=None, disable_writer=False):
    # --- 初始化 ---
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if run_name is None:
        run_name = f"grokking_LLC_p{args.p}_wd{args.weight_decay}_seed{args.seed}_{datetime.now().strftime('%m%d_%H%M')}"
    
    os.makedirs("figures", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    writer = None
    if not disable_writer:
        log_dir = os.path.join("runs", run_name)
        writer = SummaryWriter(log_dir)
        print(f"Device: {device} | Log Dir: {log_dir}")
    else:
        print(f"Device: {device} | Seed: {args.seed}")

    # --- 数据 ---
    # Grokking 通常在 50% 左右的数据比例下最明显
    # 如果数据太少(10%)可能很难学会，如果太多(80%)可能不会发生Grokking而是直接学会
    full_data = get_data(args.p, args.p, args.p+1)
    indices = torch.randperm(len(full_data))
    split = int(len(full_data) * 0.5) # 修改为 50% 训练
    train_data = full_data[indices[:split]]
    test_data = full_data[indices[split:]]

    train_loader = DataLoader(TensorDataset(train_data), batch_size=args.batch_size, shuffle=True, pin_memory=True)
    # 这里的 test_loader 不 shuffle，用于计算 LLC 时能遍历数据即可
    # 注意：理论上 LLC 应在 Training Data 上计算（衡量对训练数据的几何拟合程度）
    llc_loader = DataLoader(TensorDataset(train_data), batch_size=args.batch_size, shuffle=False, pin_memory=True)
    test_loader = DataLoader(TensorDataset(test_data), batch_size=len(test_data), shuffle=False)

    # --- 模型 ---
    model = Decoder(dim=128, num_layers=2, num_heads=4, num_tokens=args.p+2, seq_len=5).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.98))
    scaler = torch.cuda.amp.GradScaler()

    # --- LLC 估算器 ---
    # 使用 llc_loader (训练集) 来估算 LLC
    llc_calc = LLCEstimator(model, F.cross_entropy, llc_loader, device)

    # --- 监控 ---
    metrics = {'steps': [], 'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': [], 'llc': []}

    steps = 0
    pbar = tqdm(total=args.budget, desc="Training")
    
    while steps < args.budget:
        for (batch_x,) in train_loader:
            model.train()
            batch_x = batch_x.to(device, non_blocking=True)
            inp, target = batch_x[:, :-1], batch_x[:, -1]

            with torch.amp.autocast(device_type=device.type if 'cuda' in device.type else 'cpu'):
                logits = model(inp)
                loss = F.cross_entropy(logits[:, -1, :], target)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            acc = (logits[:, -1, :].argmax(-1) == target).float().mean().item()
            train_loss_val = loss.item()
            
            steps += 1
            pbar.update(1)

            # --- 监控逻辑 ---
            if steps % 100 == 0:
                # 1. 验证
                model.eval()
                with torch.no_grad():
                    for (test_batch,) in test_loader:
                        test_batch = test_batch.to(device)
                        t_inp, t_target = test_batch[:, :-1], test_batch[:, -1]
                        t_logits = model(t_inp)
                        test_loss_val = F.cross_entropy(t_logits[:, -1, :], t_target).item()
                        test_acc_val = (t_logits[:, -1, :].argmax(-1) == t_target).float().mean().item()
                
                # 2. LLC 估算 (每 500 步一次，减少开销)
                llc_val = 0.0
                if steps % 500 == 0 or steps == args.budget: 
                    # SGLD 参数: 
                    # lr=1e-4 (SGLD步长, 不宜过大否则飞出局部极小值)
                    # num_draws=50 (采样点越多越准)
                    llc_val = llc_calc.estimate(model.state_dict(), num_draws=50, lr=1e-4)
                elif len(metrics['llc']) > 0:
                    llc_val = metrics['llc'][-1] # 沿用旧值

                # 3. 记录
                metrics['steps'].append(steps)
                metrics['train_loss'].append(train_loss_val)
                metrics['train_acc'].append(acc)
                metrics['test_loss'].append(test_loss_val)
                metrics['test_acc'].append(test_acc_val)
                metrics['llc'].append(llc_val)
                
                if writer is not None:
                    writer.add_scalar("LLC", llc_val, steps)
                    writer.add_scalars("Acc", {'Train': acc, 'Test': test_acc_val}, steps)
                
                pbar.set_description(f"S:{steps}|TeAcc:{test_acc_val:.2f}|LLC:{llc_val:.4f}")

            if steps >= args.budget:
                break
    
    pbar.close()
    if writer is not None:
        writer.close()

    # --- 保存CSV ---
    csv_path = os.path.join("logs", f"{run_name}_metrics.csv")
    with open(csv_path, 'w', newline='') as f:
        writer_csv = csv.writer(f)
        writer_csv.writerow(metrics.keys())
        writer_csv.writerows(zip(*metrics.values()))
    
    return metrics


def plot_single_run(metrics, run_name):
    """绘制单次运行的结果"""
    # 过滤掉 LLC 为 0 的点（未计算的步数）
    valid_idx = [i for i, v in enumerate(metrics['llc']) if v != 0]
    v_steps = [metrics['steps'][i] for i in valid_idx]
    v_llc = [metrics['llc'][i] for i in valid_idx]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # 第一张图：Accuracy + LLC (双y轴)
    ax1_llc = ax1.twinx()
    line1 = ax1.plot(metrics['steps'], metrics['train_acc'], label='Train Acc', color='blue', alpha=0.4)
    line2 = ax1.plot(metrics['steps'], metrics['test_acc'], label='Test Acc', color='red', linewidth=2)
    ax1.set_xlabel('Steps')
    ax1.set_ylabel('Accuracy')
    ax1.set_xscale('log')
    ax1.set_title('Accuracy & LLC vs Steps')
    
    line3 = ax1_llc.plot(v_steps, v_llc, label='LLC', color='green', marker='.', linestyle='--', alpha=0.7)
    ax1_llc.set_ylabel('LLC (Complexity)', color='green')
    ax1_llc.tick_params(axis='y', labelcolor='green')
    ax1_llc.set_xscale('log')
    
    lines = line1 + line2 + line3
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='best')
    ax1.grid(True, alpha=0.3)
    
    # 第二张图：Loss + LLC (双y轴)
    ax2_llc = ax2.twinx()
    line4 = ax2.plot(metrics['steps'], metrics['train_loss'], label='Train Loss', color='blue', alpha=0.4)
    line5 = ax2.plot(metrics['steps'], metrics['test_loss'], label='Test Loss', color='red', linewidth=2)
    ax2.set_xlabel('Steps')
    ax2.set_ylabel('Loss')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_title('Loss & LLC vs Steps')
    
    line6 = ax2_llc.plot(v_steps, v_llc, label='LLC', color='green', marker='.', linestyle='--', alpha=0.7)
    ax2_llc.set_ylabel('LLC (Complexity)', color='green')
    ax2_llc.tick_params(axis='y', labelcolor='green')
    ax2_llc.set_xscale('log')
    
    lines = line4 + line5 + line6
    labels = [l.get_label() for l in lines]
    ax2.legend(lines, labels, loc='best')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"figures/{run_name}.png", dpi=150)
    plt.close()
    print(f"Plot saved to figures/{run_name}.png")


def plot_multi_seed(all_metrics, run_name):
    """绘制多种子运行的误差带图表"""
    # 设置科研级别的字体和样式
    plt.rcParams.update({
        'font.size': 11,
        'font.family': 'serif',
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 14,
        'axes.linewidth': 1.2,
        'grid.linewidth': 0.8,
        'lines.linewidth': 2.0,
    })
    
    # 找出所有运行共同的步数
    common_steps = all_metrics[0]['steps']
    
    # 对每个指标计算均值和上下界
    def compute_stats(key):
        data = np.array([m[key] for m in all_metrics])  # shape: (n_seeds, n_steps)
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        lower = np.min(data, axis=0)
        upper = np.max(data, axis=0)
        return mean, lower, upper
    
    train_acc_mean, train_acc_lower, train_acc_upper = compute_stats('train_acc')
    test_acc_mean, test_acc_lower, test_acc_upper = compute_stats('test_acc')
    train_loss_mean, train_loss_lower, train_loss_upper = compute_stats('train_loss')
    test_loss_mean, test_loss_lower, test_loss_upper = compute_stats('test_loss')
    
    # 处理LLC（过滤0值）
    llc_data = []
    for m in all_metrics:
        valid_idx = [i for i, v in enumerate(m['llc']) if v != 0]
        llc_data.append({
            'steps': [m['steps'][i] for i in valid_idx],
            'llc': [m['llc'][i] for i in valid_idx]
        })
    
    # 找到LLC的共同步数（所有种子都有的）
    common_llc_steps = set(llc_data[0]['steps'])
    for ld in llc_data[1:]:
        common_llc_steps &= set(ld['steps'])
    common_llc_steps = sorted(list(common_llc_steps))
    
    # 对齐LLC数据
    llc_values = []
    for ld in llc_data:
        step_to_llc = dict(zip(ld['steps'], ld['llc']))
        llc_values.append([step_to_llc[s] for s in common_llc_steps])
    
    llc_values = np.array(llc_values)
    llc_mean = np.mean(llc_values, axis=0)
    llc_lower = np.min(llc_values, axis=0)
    llc_upper = np.max(llc_values, axis=0)
    
    # 配色方案（科研美感）
    colors = {
        'train': '#1f77b4',  # 蓝色
        'test': '#d62728',   # 红色
        'llc': '#2ca02c',    # 绿色
    }
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # ===== 第一张图：Accuracy + LLC =====
    ax1_llc = ax1.twinx()
    
    # 绘制 Accuracy
    ax1.fill_between(common_steps, train_acc_lower, train_acc_upper, 
                     color=colors['train'], alpha=0.2, linewidth=0)
    ax1.plot(common_steps, train_acc_mean, label='Train Acc', 
             color=colors['train'], linewidth=2, alpha=0.9)
    
    ax1.fill_between(common_steps, test_acc_lower, test_acc_upper, 
                     color=colors['test'], alpha=0.2, linewidth=0)
    ax1.plot(common_steps, test_acc_mean, label='Test Acc', 
             color=colors['test'], linewidth=2.5, alpha=0.95)
    
    ax1.set_xlabel('Training Steps', fontweight='medium')
    ax1.set_ylabel('Accuracy', fontweight='medium')
    ax1.set_xscale('log')
    ax1.set_ylim([-0.05, 1.05])
    ax1.grid(True, alpha=0.25, linestyle='--')
    
    # 绘制 LLC
    ax1_llc.fill_between(common_llc_steps, llc_lower, llc_upper, 
                         color=colors['llc'], alpha=0.15, linewidth=0)
    ax1_llc.plot(common_llc_steps, llc_mean, label='LLC', 
                 color=colors['llc'], linewidth=2, linestyle='--', alpha=0.8)
    ax1_llc.set_ylabel('LLC (Complexity)', color=colors['llc'], fontweight='medium')
    ax1_llc.tick_params(axis='y', labelcolor=colors['llc'])
    ax1_llc.set_xscale('log')
    
    # 图例放在底部横向排列
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_llc.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, 
              loc='upper center', bbox_to_anchor=(0.5, -0.12), 
              ncol=3, frameon=True, fancybox=True, shadow=False, 
              edgecolor='gray', framealpha=0.95)
    
    # ===== 第二张图：Loss + LLC =====
    ax2_llc = ax2.twinx()
    
    # 绘制 Loss
    ax2.fill_between(common_steps, train_loss_lower, train_loss_upper, 
                     color=colors['train'], alpha=0.2, linewidth=0)
    ax2.plot(common_steps, train_loss_mean, label='Train Loss', 
             color=colors['train'], linewidth=2, alpha=0.9)
    
    ax2.fill_between(common_steps, test_loss_lower, test_loss_upper, 
                     color=colors['test'], alpha=0.2, linewidth=0)
    ax2.plot(common_steps, test_loss_mean, label='Test Loss', 
             color=colors['test'], linewidth=2.5, alpha=0.95)
    
    ax2.set_xlabel('Training Steps', fontweight='medium')
    ax2.set_ylabel('Loss', fontweight='medium')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.25, linestyle='--')
    
    # 绘制 LLC
    ax2_llc.fill_between(common_llc_steps, llc_lower, llc_upper, 
                         color=colors['llc'], alpha=0.15, linewidth=0)
    ax2_llc.plot(common_llc_steps, llc_mean, label='LLC', 
                 color=colors['llc'], linewidth=2, linestyle='--', alpha=0.8)
    ax2_llc.set_ylabel('LLC (Complexity)', color=colors['llc'], fontweight='medium')
    ax2_llc.tick_params(axis='y', labelcolor=colors['llc'])
    ax2_llc.set_xscale('log')
    
    # 图例放在底部横向排列
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_llc.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, 
              loc='upper center', bbox_to_anchor=(0.5, -0.12), 
              ncol=3, frameon=True, fancybox=True, shadow=False, 
              edgecolor='gray', framealpha=0.95)
    
    plt.tight_layout(rect=[0, 0.05, 1, 1])  # 为底部图例留出空间
    plt.savefig(f"figures/{run_name}_multiseed.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Multi-seed plot saved to figures/{run_name}_multiseed.png")
    
    # 恢复默认设置
    plt.rcParams.update(plt.rcParamsDefault)


def run_multi_seed(args, seeds=[42, 123, 456]):
    """运行多个种子并收集结果"""
    all_metrics = []
    base_run_name = f"grokking_LLC_p{args.p}_wd{args.weight_decay}_{datetime.now().strftime('%m%d_%H%M')}"
    
    print(f"\n{'='*60}")
    print(f"Running {len(seeds)} experiments with different seeds")
    print(f"{'='*60}\n")
    
    for i, seed in enumerate(seeds):
        print(f"\n[Experiment {i+1}/{len(seeds)}] Seed: {seed}")
        print("-" * 40)
        args.seed = seed
        metrics = main(args, run_name=f"{base_run_name}_seed{seed}", disable_writer=True)
        all_metrics.append(metrics)
        torch.cuda.empty_cache()  # 清理显存
    
    # 绘制多种子误差带图
    print(f"\n{'='*60}")
    print("Generating multi-seed plot with error bands...")
    print(f"{'='*60}\n")
    plot_multi_seed(all_metrics, base_run_name)
    
    return all_metrics

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--p", type=int, default=97)
    parser.add_argument("--budget", type=int, default=30000) 
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0) 
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--multi_seed", action='store_true', 
                        help="Run with multiple seeds and plot error bands")
    parser.add_argument("--seeds", type=int, nargs='+', default=[42, 123, 456],
                        help="List of seeds to use for multi-seed runs")
    args = parser.parse_args()
    
    if args.multi_seed:
        # 多种子运行，绘制误差带
        all_metrics = run_multi_seed(args, seeds=args.seeds)
    else:
        # 单次运行
        metrics = main(args)
        run_name = f"grokking_LLC_p{args.p}_wd{args.weight_decay}_seed{args.seed}_{datetime.now().strftime('%m%d_%H%M')}"
        plot_single_run(metrics, run_name)