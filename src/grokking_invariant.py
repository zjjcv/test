import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
import math
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import LambdaLR

# 配置
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
print(f"Using device: {DEVICE}")

# ==========================================
# 1. 任务定义: 模减法 (x - y)
# ==========================================

def make_data_modular_subtraction(p=97, train_frac=0.4, seed=42):
    """生成 x - y (mod p) 数据集"""
    torch.manual_seed(seed)
    pairs = torch.cartesian_prod(torch.arange(p), torch.arange(p))
    
    # 划分训练/测试
    n_total = len(pairs)
    n_train = int(n_total * train_frac)
    perm = torch.randperm(n_total)
    
    train_idx = perm[:n_train]
    test_idx = perm[n_train:]
    
    x_train = pairs[train_idx].to(DEVICE)
    y_train = (pairs[train_idx, 0] - pairs[train_idx, 1]) % p
    y_train = y_train.to(DEVICE)
    
    x_test = pairs[test_idx].to(DEVICE)
    y_test = (pairs[test_idx, 0] - pairs[test_idx, 1]) % p
    y_test = y_test.to(DEVICE)
    
    return x_train, y_train, x_test, y_test

# ==========================================
# 2. 模型定义 (Invariant Transformer)
# ==========================================

class NormalizedLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        
    def forward(self, x):
        w_norm = F.normalize(self.weight, p=2, dim=1)
        return F.linear(x, w_norm)

class MLP(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.up = nn.Linear(dim, mult * dim, bias=False)
        self.down = nn.Linear(mult * dim, dim, bias=False)
        self.act = nn.ReLU()

    def forward(self, x):
        return self.down(self.act(self.up(x)))

class Transformer(nn.Module):
    def __init__(self, vocab_size=97, dim=128, n_layers=2, n_head=4, seq_len=2):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, dim)
        self.pos_emb = nn.Embedding(seq_len, dim)
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'attn': nn.MultiheadAttention(dim, n_head, batch_first=True),
                'ln1': nn.LayerNorm(dim),
                'mlp': MLP(dim),
                'ln2': nn.LayerNorm(dim)
            }) for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(dim)
        self.head = NormalizedLinear(dim, vocab_size)
        self.pos_idxs = torch.arange(seq_len).to(DEVICE)

    def forward(self, x):
        h = self.tok_emb(x) + self.pos_emb(self.pos_idxs[:x.size(1)])
        for layer in self.layers:
            normed = layer['ln1'](h)
            attn, _ = layer['attn'](normed, normed, normed)
            h = h + attn
            normed = layer['ln2'](h)
            h = h + layer['mlp'](normed)
        h = self.ln_f(h)
        return self.head(h[:, -1, :])

# ==========================================
# 3. 几何投影与规范化工具 (Estimator Core)
# ==========================================

CANONICAL_NORM = 20.0

def canonicalize_model(model):
    with torch.no_grad():
        for layer in model.layers:
            mlp = layer['mlp']
            w_u, w_d = mlp.up.weight, mlp.down.weight
            nu, nd = w_u.norm(), w_d.norm()
            target = (nu * nd).sqrt() + 1e-8
            w_u.mul_(target / (nu + 1e-8))
            w_d.mul_(target / (nd + 1e-8))
        
        curr_norm = sum((p**2).sum() for p in model.parameters()).sqrt()
        scale = CANONICAL_NORM / (curr_norm + 1e-8)
        for p in model.parameters():
            p.mul_(scale)

def project_tangent(model, vecs_dict):
    with torch.no_grad():
        for layer in model.layers:
            w_u, w_d = layer['mlp'].up.weight, layer['mlp'].down.weight
            v_u, v_d = vecs_dict[w_u], vecs_dict[w_d]
            dot = (v_u * w_u).sum() - (v_d * w_d).sum()
            norm = (w_u**2).sum() + (w_d**2).sum() + 1e-8
            alpha = dot / norm
            v_u.sub_(alpha * w_u)
            v_d.add_(alpha * w_d)

        dot_vt = sum((vecs_dict[p] * p).sum() for p in model.parameters())
        dot_tt = sum((p**2).sum() for p in model.parameters()) + 1e-8
        scale = dot_vt / dot_tt
        for p in model.parameters():
            vecs_dict[p].sub_(scale * p)

# ==========================================
# 4. 双重不变性 LLC 估计器
# ==========================================

def estimate_invariant_llc(model, x_batch, y_batch, steps=100, lr=1e-4, beta=100.0):
    model_run = copy.deepcopy(model)
    model_run.train() 
    canonicalize_model(model_run)
    
    theta0 = {p: p.detach().clone() for p in model_run.parameters()}
    losses = []
    
    for _ in range(steps):
        model_run.zero_grad()
        out = model_run(x_batch)
        loss = F.cross_entropy(out, y_batch)
        loss.backward()
        
        loss_val = loss.item()
        if math.isnan(loss_val): loss_val = 100.0
        losses.append(loss_val)
        
        with torch.no_grad():
            noise_std = math.sqrt(2 * lr / beta)
            grads = {p: p.grad for p in model_run.parameters()}
            noise = {p: torch.randn_like(p) * noise_std for p in model_run.parameters()}
            deltas = {p: (p - theta0[p]) for p in model_run.parameters()}
            
            project_tangent(model_run, grads)
            project_tangent(model_run, noise)
            project_tangent(model_run, deltas)
            
            for p in model_run.parameters():
                update = -lr * (grads[p] + 10.0 * deltas[p]) + noise[p]
                p.add_(update)
            
            canonicalize_model(model_run)

    return beta * len(x_batch) * (np.mean(losses) - losses[0])

# ==========================================
# 5. 训练循环 (支持 Long-Run Grokking)
# ==========================================

def train_and_monitor(wd_value, steps=100000, eval_interval=500):
    print(f"\n=== Starting Training: Weight Decay = {wd_value}, Steps = {steps} ===")
    
    p = 97
    x_train, y_train, x_test, y_test = make_data_modular_subtraction(p=p, train_frac=0.4)
    
    model = Transformer(vocab_size=p, dim=128, n_layers=2, n_head=4).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=wd_value, betas=(0.9, 0.98))
    
    logs = {'steps': [], 'train_acc': [], 'test_acc': [], 'llc': []}
    
    for step in range(1, steps + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(x_train)
        loss = F.cross_entropy(logits, y_train)
        loss.backward()
        optimizer.step()
        
        # 智能采样策略：
        # 1. 早期 (前1000步) 每 100 步采样一次，保证 Log 轴左侧有数据
        # 2. 后期 每 eval_interval 步采样一次
        is_early_log = (step <= 1000 and step % 100 == 0)
        is_normal_log = (step % eval_interval == 0)
        
        if is_early_log or is_normal_log or step == steps:
            model.eval()
            with torch.no_grad():
                train_acc = (logits.argmax(-1) == y_train).float().mean().item()
                test_logits = model(x_test)
                test_acc = (test_logits.argmax(-1) == y_test).float().mean().item()
            
            # 估计 LLC
            llc_val = estimate_invariant_llc(model, x_train, y_train, steps=100)
            
            logs['steps'].append(step)
            logs['train_acc'].append(train_acc)
            logs['test_acc'].append(test_acc)
            logs['llc'].append(llc_val)
            
            print(f"Step {step:6d} | Train: {train_acc:.3f} | Test: {test_acc:.3f} | Inv-LLC: {llc_val:.1f}")
            
            # 提前停止检测 (Grokking 完成后停止)
            if test_acc > 0.995 and step > 5000:
                # 确认稳定了至少 5 个点
                if len(logs['test_acc']) > 5 and min(logs['test_acc'][-5:]) > 0.99:
                    print("Converged (Grokking Complete). Stopping early.")
                    break
                
    return logs

# ==========================================
# 6. 主程序与绘图 (Log-Scale X-Axis)
# ==========================================

def run_comparison():
    # 1. 无正则化 (WD=0.0): 预计需要很长时间 (30k-100k steps) 才能 Grok
    logs_wd0 = train_and_monitor(wd_value=0.0, steps=100000, eval_interval=1000)
    
    # 2. 强结构约束 (WD=1.0): 预计很快 (2k-10k steps) 发生 Grok
    logs_wd1 = train_and_monitor(wd_value=1.0, steps=100000, eval_interval=200)
    
    print("\nPlotting results with Log10 X-axis...")
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), dpi=150)
    
    def plot_single(ax, log, title):
        steps = log['steps']
        
        # 设置 X 轴为 Log 刻度
        ax.set_xscale('log')
        
        ax1 = ax
        ax2 = ax1.twinx()
        
        # Accuracy
        l1 = ax1.plot(steps, log['train_acc'], label='Train Acc', color='gray', linestyle='--', alpha=0.5, linewidth=1.5)
        l2 = ax1.plot(steps, log['test_acc'], label='Test Acc (Grokking)', color='#1f77b4', linewidth=3)
        
        # LLC
        l3 = ax2.plot(steps, log['llc'], label='Invariant LLC', color='#d62728', linewidth=2.5, alpha=0.9)
        
        ax1.set_xlabel('Optimization Steps (Log Scale)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Accuracy', color='#1f77b4', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Geometric Complexity (LLC)', color='#d62728', fontsize=12, fontweight='bold')
        
        ax1.set_ylim(-0.05, 1.05)
        
        # 设置标题
        ax.set_title(title, fontsize=14, pad=12, fontweight='bold')
        
        # 合并图例
        lines = l1 + l2 + l3
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='center right', fontsize=10, frameon=True, framealpha=0.9)
        
        ax1.grid(True, which="both", ls="-", alpha=0.2) # Log grid

    # 左图: 无正则化 (Observe Delayed Grokking)
    plot_single(axes[0], logs_wd0, 'A. No Regularization (WD=0.0)\nDelayed Grokking & Slow Complexity Decay')
    
    # 右图: 强正则化 (Observe Accelerated Grokking)
    plot_single(axes[1], logs_wd1, 'B. Structural Constraint (WD=1.0)\nAccelerated Grokking & Sharp Complexity Collapse')
    
    plt.tight_layout()
    plt.savefig('grokking_dynamics_log_scale.png')
    print("Done! Saved plot to 'grokking_dynamics_log_scale.png'")

if __name__ == "__main__":
    run_comparison()