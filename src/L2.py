import math
import os
import copy
import numpy as np
import argparse
from datetime import datetime
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from torch.optim import Optimizer
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

# =============================================================================
# 1. 核心组件 (SGLD & LLC Estimator)
# =============================================================================

class SGLD(Optimizer):
    """用于 LLC 估计的 SGLD 优化器"""
    def __init__(self, params, lr=1e-4, noise_scale=1.0):
        defaults = dict(lr=lr, noise_scale=noise_scale)
        super(SGLD, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None: loss = closure()
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None: continue
                d_p = p.grad.data
                p.data.add_(d_p, alpha=-0.5 * group['lr'])
                noise_std = torch.sqrt(torch.tensor(group['lr'])) * group['noise_scale']
                noise = torch.randn_like(p.data) * noise_std
                p.data.add_(noise)
        return loss

class LLCEstimator:
    def __init__(self, model, criterion, dataloader, device):
        self.model = model
        self.criterion = criterion
        self.dataloader = dataloader
        self.device = device

    def estimate(self, current_weights_dict, num_draws=40, lr=1e-4, epsilon=1.0):
        sampling_model = copy.deepcopy(self.model)
        sampling_model.load_state_dict(current_weights_dict)
        
        # 计算基准损失 (全量数据以减少方差)
        base_loss = self._compute_full_loss(sampling_model)
        
        sampling_model.train()
        sgld_optim = SGLD(sampling_model.parameters(), lr=lr, noise_scale=epsilon)
        loss_trace = []
        iter_dl = iter(self.dataloader)
        
        for _ in range(num_draws):
            try:
                batch = next(iter_dl)
            except StopIteration:
                iter_dl = iter(self.dataloader)
                batch = next(iter_dl)
            
            batch_x = batch[0].to(self.device)
            inp, target = batch_x[:, :-1], batch_x[:, -1]
            
            sgld_optim.zero_grad()
            out = sampling_model(inp)
            logits = out[0] if isinstance(out, tuple) else out
            loss = self.criterion(logits[:, -1, :], target)
            loss.backward()
            sgld_optim.step()
            loss_trace.append(loss.item())

        avg_sampling_loss = np.mean(loss_trace)
        llc_proxy = avg_sampling_loss - base_loss
        del sampling_model
        return llc_proxy

    def _compute_full_loss(self, model):
        model.eval()
        total_loss = 0
        total_count = 0
        with torch.no_grad():
            for batch in self.dataloader:
                batch_x = batch[0].to(self.device)
                inp, target = batch_x[:, :-1], batch_x[:, -1]
                out = model(inp)
                logits = out[0] if isinstance(out, tuple) else out
                loss = self.criterion(logits[:, -1, :], target)
                total_loss += loss.item() * batch_x.size(0)
                total_count += batch_x.size(0)
        return total_loss / total_count

# =============================================================================
# 2. 模型定义
# =============================================================================

class Block(nn.Module):
    def __init__(self, dim, num_heads, dropout=0.0):
        super().__init__()
        self.ln_1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True, dropout=dropout)
        self.ln_2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim), nn.Dropout(dropout)
        )
    def forward(self, x, attn_mask=None):
        x = x + self.attn(self.ln_1(x), self.ln_1(x), self.ln_1(x), attn_mask=attn_mask, need_weights=False)[0]
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

# =============================================================================
# 3. 工具函数
# =============================================================================

def get_data_multiplication(p):
    """只生成 x * y (mod p) 数据"""
    x = torch.arange(p)
    y = torch.arange(1, p)
    x, y = torch.cartesian_prod(x, y).T
    eq = torch.ones_like(x) * (p)
    op = torch.ones_like(x) * (p + 1)
    result = (x * y) % p 
    data = torch.stack([x, op, y, eq, result]).T
    return data

def calc_l2_norm(model):
    """计算模型所有参数的 L2 范数"""
    total_norm = 0.0
    for p in model.parameters():
        param_norm = p.data.norm(2)
        total_norm += param_norm.item() ** 2
    return total_norm ** 0.5

def plot_results(metrics, args, save_path):
    steps = metrics['steps']
    
    # 过滤有效的 LLC 点
    llc_raw = np.array(metrics['llc'])
    mask = ~np.isnan(llc_raw)
    v_steps = np.array(steps)[mask]
    v_llc = llc_raw[mask]

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(24, 5))
    
    # 1. Accuracy
    ax1.plot(steps, metrics['train_acc'], label='Train', color='blue', alpha=0.5)
    ax1.plot(steps, metrics['test_acc'], label='Test', color='red', linewidth=2)
    ax1.set_title('Accuracy')
    ax1.set_xlabel('Steps')
    ax1.set_xscale('log')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Loss
    ax2.plot(steps, metrics['train_loss'], label='Train', color='blue', alpha=0.5)
    ax2.plot(steps, metrics['test_loss'], label='Test', color='red')
    ax2.set_title('Loss')
    ax2.set_xlabel('Steps')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3)

    # 3. LLC
    ax3.plot(v_steps, v_llc, label='LLC Estimate', color='green', marker='.')
    ax3.set_title('Complexity (LLC)')
    ax3.set_xlabel('Steps')
    ax3.set_xscale('log')
    ax3.grid(True, alpha=0.3)
    
    # 4. L2 Norm
    ax4.plot(steps, metrics['l2_norm'], label='Weight L2 Norm', color='purple', linewidth=2)
    ax4.set_title('Parameter Norm (L2)')
    ax4.set_xlabel('Steps')
    ax4.set_xscale('log')
    ax4.grid(True, alpha=0.3)

    plt.suptitle(f"Quick Check: x*y (WD={args.weight_decay})")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"\nPlot saved to {save_path}")

# =============================================================================
# 4. 主程序
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--p", type=int, default=97)
    parser.add_argument("--budget", type=int, default=20000)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0) 
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on {device} | Task: x*y | WD: {args.weight_decay}")
    
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    
    # 数据准备
    full_data = get_data_multiplication(args.p)
    indices = torch.randperm(len(full_data))
    split = int(len(full_data) * 0.5)
    train_data = full_data[indices[:split]]
    test_data = full_data[indices[split:]]

    train_loader = DataLoader(TensorDataset(train_data), batch_size=args.batch_size, shuffle=True)
    llc_loader = DataLoader(TensorDataset(train_data), batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(TensorDataset(test_data), batch_size=len(test_data), shuffle=False)

    model = Decoder(dim=128, num_layers=2, num_heads=4, num_tokens=args.p+2, seq_len=5).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.98))
    scaler = torch.cuda.amp.GradScaler()
    llc_estimator = LLCEstimator(model, F.cross_entropy, llc_loader, device)

    metrics = {
        'steps': [], 'train_loss': [], 'train_acc': [], 
        'test_loss': [], 'test_acc': [], 'llc': [], 'l2_norm': []
    }

    steps = 0
    # 初始化进度条
    pbar = tqdm(total=args.budget, desc="QuickCheck")
    
    # 初始化显示指标
    display_metrics = {'TrAcc': '0.00', 'TeAcc': '0.00', 'L2': '0.00', 'LLC': 'N/A'}

    while steps < args.budget:
        for (batch_x,) in train_loader:
            model.train()
            batch_x = batch_x.to(device)
            inp, target = batch_x[:, :-1], batch_x[:, -1]

            with torch.amp.autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                logits = model(inp)
                loss = F.cross_entropy(logits[:, -1, :], target)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            # 实时计算 Train Acc
            with torch.no_grad():
                train_acc = (logits[:, -1, :].argmax(-1) == target).float().mean().item()
            
            steps += 1
            
            # 每10步更新快速指标 (TrAcc, L2)
            if steps % 10 == 0:
                l2_val = calc_l2_norm(model)
                display_metrics['TrAcc'] = f"{train_acc:.3f}"
                display_metrics['L2'] = f"{l2_val:.2f}"
                pbar.set_postfix(display_metrics)
            
            pbar.update(1)

            # 每100步更新慢速指标 (Test, LLC)
            if steps % 100 == 0:
                model.eval()
                with torch.no_grad():
                    test_acc_sum, test_loss_sum, count = 0, 0, 0
                    for (batch_x,) in test_loader:
                        batch_x = batch_x.to(device)
                        t_inp, t_target = batch_x[:, :-1], batch_x[:, -1]
                        t_logits = model(t_inp)
                        t_loss = F.cross_entropy(t_logits[:, -1, :], t_target).item()
                        t_acc = (t_logits[:, -1, :].argmax(-1) == t_target).sum().item()
                        test_loss_sum += t_loss * batch_x.size(0)
                        test_acc_sum += t_acc
                        count += batch_x.size(0)
                    test_acc = test_acc_sum / count
                    test_loss = test_loss_sum / count
                
                # LLC 计算 (每500步)
                llc_val = np.nan
                if steps % 500 == 0 or steps == args.budget:
                    llc_val = llc_estimator.estimate(model.state_dict(), num_draws=25, lr=1e-4)
                elif len(metrics['llc']) > 0:
                    llc_val = metrics['llc'][-1]

                # 记录
                l2_val = calc_l2_norm(model)
                metrics['steps'].append(steps)
                metrics['train_loss'].append(loss.item())
                metrics['train_acc'].append(train_acc)
                metrics['test_loss'].append(test_loss)
                metrics['test_acc'].append(test_acc)
                metrics['llc'].append(llc_val)
                metrics['l2_norm'].append(l2_val)

                # 更新进度条显示
                display_metrics['TeAcc'] = f"{test_acc:.3f}"
                if not np.isnan(llc_val):
                    display_metrics['LLC'] = f"{llc_val:.2f}"
                pbar.set_postfix(display_metrics)

            if steps >= args.budget:
                break
    
    pbar.close()
    
    # 绘图
    timestamp = datetime.now().strftime('%m%d_%H%M')
    os.makedirs(f"quick_check_{timestamp}", exist_ok=True)
    plot_results(metrics, args, f"quick_check_{timestamp}/xy_monitor.png")

if __name__ == "__main__":
    main()