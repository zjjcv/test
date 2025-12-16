"""
GPT-2 Medium Grokking Experiment with FP16 + Tensor Core Optimization

Performance Optimizations:
- FP16 mixed precision training with torch.amp
- Tensor Core acceleration (requires dimensions divisible by 8)
- TF32 enabled for Ampere+ GPUs (A100/4090)
- cuDNN benchmark mode
- Gradient checkpointing for memory efficiency
- Optimized batch size (128, divisible by 8)
- Model dim (1024, divisible by 8)

Model Architecture:
- GPT-2 Medium: 1024 dim, 24 layers, 16 heads (~345M parameters)
- Single task: x-y only
- Single seed per experiment
"""

import math
import os
import csv
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
# 1. 基础组件
# =============================================================================

class SGLD(Optimizer):
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
            logits = sampling_model(inp)
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
                logits = model(inp)
                loss = self.criterion(logits[:, -1, :], target)
                total_loss += loss.item() * batch_x.size(0)
                total_count += batch_x.size(0)
        return total_loss / total_count

# =============================================================================
# 2. GPT-2 架构
# =============================================================================

class CausalSelfAttention(nn.Module):
    """GPT-2 style causal self-attention"""
    def __init__(self, dim, num_heads, dropout=0.0):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        
        # Q, K, V projections
        self.c_attn = nn.Linear(dim, 3 * dim)
        # Output projection
        self.c_proj = nn.Linear(dim, dim)
        # Dropout
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        B, T, C = x.size()  # batch, sequence, embedding
        
        # Calculate Q, K, V
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.dim, dim=2)
        
        # Reshape for multi-head attention
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # (B, nh, T, hs)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention scores
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        
        # Apply causal mask
        if mask is not None:
            att = att.masked_fill(mask[:T, :T] == float('-inf'), float('-inf'))
        
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        
        # Apply attention to values
        y = att @ v  # (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        
        # Output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

class MLP(nn.Module):
    """GPT-2 style MLP"""
    def __init__(self, dim, dropout=0.0):
        super().__init__()
        self.c_fc = nn.Linear(dim, 4 * dim)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * dim, dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class GPT2Block(nn.Module):
    """GPT-2 Transformer Block with Pre-LN"""
    def __init__(self, dim, num_heads, dropout=0.0):
        super().__init__()
        self.ln_1 = nn.LayerNorm(dim)
        self.attn = CausalSelfAttention(dim, num_heads, dropout)
        self.ln_2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, dropout)
        
    def forward(self, x, mask=None):
        # Pre-LN: LayerNorm before attention
        x = x + self.attn(self.ln_1(x), mask)
        # Pre-LN: LayerNorm before MLP
        x = x + self.mlp(self.ln_2(x))
        return x

class GPT2Decoder(nn.Module):
    """GPT-2 style decoder for modular arithmetic"""
    def __init__(self, dim=128, num_layers=2, num_heads=4, num_tokens=97, seq_len=5, dropout=0.0, use_checkpoint=False):
        super().__init__()
        self.token_embeddings = nn.Embedding(num_tokens, dim)
        self.position_embeddings = nn.Embedding(seq_len, dim)
        self.drop = nn.Dropout(dropout)
        self.use_checkpoint = use_checkpoint
        
        # GPT-2 blocks
        self.blocks = nn.ModuleList([
            GPT2Block(dim, num_heads, dropout) for _ in range(num_layers)
        ])
        
        # Final layer norm
        self.ln_f = nn.LayerNorm(dim)
        # Output head
        self.head = nn.Linear(dim, num_tokens, bias=False)
        
        # Causal mask
        mask = torch.triu(torch.full((seq_len, seq_len), float('-inf')), diagonal=1)
        self.register_buffer("causal_mask", mask)
        self.register_buffer("pos_ids", torch.arange(seq_len))
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
    
    def forward(self, x):
        B, T = x.size()
        
        # Token + position embeddings
        tok_emb = self.token_embeddings(x)
        pos_emb = self.position_embeddings(self.pos_ids[:T])
        h = self.drop(tok_emb + pos_emb)
        
        # Get causal mask
        mask = self.causal_mask[:T, :T]
        
        # Apply transformer blocks with optional gradient checkpointing
        if self.use_checkpoint and self.training:
            for block in self.blocks:
                h = torch.utils.checkpoint.checkpoint(block, h, mask, use_reentrant=False)
        else:
            for block in self.blocks:
                h = block(h, mask)
        
        # Final layer norm and projection
        h = self.ln_f(h)
        logits = self.head(h)
        
        return logits

# =============================================================================
# 3. 工具函数
# =============================================================================

def calc_l2_norm(model):
    """计算模型所有参数的 L2 范数"""
    total_norm = 0.0
    for p in model.parameters():
        param_norm = p.data.norm(2)
        total_norm += param_norm.item() ** 2
    return total_norm ** 0.5

def calc_spectral_entropy(model):
    """计算模型权重矩阵的谱熵"""
    total_entropy = 0.0
    num_matrices = 0
    
    for name, param in model.named_parameters():
        if len(param.shape) >= 2:  # 只对矩阵计算谱熵
            try:
                # 计算奇异值
                with torch.no_grad():
                    s = torch.linalg.svdvals(param.data.float())
                    # 归一化为概率分布
                    s_normalized = s / (s.sum() + 1e-10)
                    # 计算熵
                    entropy = -(s_normalized * torch.log(s_normalized + 1e-10)).sum().item()
                    total_entropy += entropy
                    num_matrices += 1
            except Exception:
                continue
    
    return total_entropy / max(num_matrices, 1)

# =============================================================================
# 4. 数据生成
# =============================================================================

def get_data(p, eq_token, op_token, task_name):
    x = torch.arange(p)
    y = torch.arange(1, p)
    x, y = torch.cartesian_prod(x, y).T
    eq = torch.ones_like(x) * eq_token
    op = torch.ones_like(x) * op_token
    if task_name == 'x+y': result = (x + y) % p
    elif task_name == 'x-y': result = (x - y) % p
    elif task_name == 'x_div_y': 
        res_list = [(xi.item() * pow(yi.item(), p-2, p)) % p for xi, yi in zip(x, y)]
        result = torch.tensor(res_list)
    elif task_name == 'x*y': result = (x * y) % p
    elif task_name == 'x2+y2': result = (x**2 + y**2) % p
    elif task_name == 'x2+xy+y2': result = (x**2 + x*y + y**2) % p
    elif task_name == 'x2+xy+y2+x': result = (x**2 + x*y + y**2 + x) % p
    elif task_name == 'x3+xy': result = (x**3 + x*y) % p
    elif task_name == 'x3+xy2+y': result = (x**3 + x*y**2 + y) % p
    else: raise ValueError(f"Unknown task: {task_name}")
    data = torch.stack([x, op, y, eq, result]).T
    return data

# =============================================================================
# 5. 原子训练任务
# =============================================================================

def run_atomic_experiment(config):
    """
    原子任务：训练单个实验配置
    保存格式：每个种子保存一个CSV文件，每行记录每100步的指标
    """
    task_name, wd, seed, device_id, args, output_dir = config
    device = torch.device(f"cuda:{device_id}")
    
    # 清理显存，避免累积
    torch.cuda.empty_cache()
    
    # 启用 TF32 for Tensor Cores (Ampere+ GPUs)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    # 启用 cuDNN benchmark 模式以优化性能
    torch.backends.cudnn.benchmark = True
    
    # 设置 matmul 精度为 high 以使用 Tensor Cores
    torch.set_float32_matmul_precision('high')
    
    # 再次设置种子确保进程独立性
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    full_data = get_data(args.p, args.p, args.p+1, task_name)
    indices = torch.randperm(len(full_data))
    split = int(len(full_data) * 0.5)
    train_data = full_data[indices[:split]]
    test_data = full_data[indices[split:]]

    # num_workers=0 避免多进程嵌套
    train_loader = DataLoader(TensorDataset(train_data), batch_size=args.batch_size, shuffle=True, num_workers=0)
    llc_loader = DataLoader(TensorDataset(train_data), batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(TensorDataset(test_data), batch_size=len(test_data), shuffle=False, num_workers=0)

    # 定义checkpoint配置 - 只保存关键步数的checkpoint
    save_checkpoints = True
    checkpoint_steps = {100, 1000, 10000, 100000}  # 只保存4个关键步数
    
    # 使用 GPT-2 Medium 架构 (1024 dim, 24 layers, 16 heads)
    model = GPT2Decoder(dim=1024, num_layers=24, num_heads=16, num_tokens=args.p+2, seq_len=5, 
                       dropout=0.1, use_checkpoint=True).to(device)
    
    # 计算参数量
    num_params = sum(p.numel() for p in model.parameters())
    print(f"\n{'='*60}")
    print(f"GPT-2 Medium | Task: {task_name} | WD={wd} | Seed={seed}")
    print(f"Model parameters: {num_params:,} ({num_params/1e6:.2f}M)")
    print(f"FP16: Enabled | Tensor Cores: Enabled | Grad Checkpoint: On")
    print(f"Checkpoints: Save at steps {sorted(checkpoint_steps)}")
    print(f"{'='*60}\n")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=wd, betas=(0.9, 0.98))
    
    # FP16 混合精度训练 - 启用 Tensor Cores
    scaler = torch.amp.GradScaler('cuda', 
                                   init_scale=2.**16,  # 初始缩放因子
                                   growth_factor=2.0,   # 增长因子
                                   backoff_factor=0.5,  # 回退因子
                                   growth_interval=2000) # 增长间隔
    llc_estimator = LLCEstimator(model, F.cross_entropy, llc_loader, device)

    # 创建保存目录 - 固定在/root/autodl-tmp/test/results下，标注为GPT-2-Medium
    task_safe = task_name.replace("/", "_div_").replace("*", "_mul_").replace("+", "_plus_")
    results_base = "/root/autodl-tmp/test/results"
    data_dir = os.path.join(results_base, "data_gpt2_medium", task_safe, f"wd_{wd}")
    os.makedirs(data_dir, exist_ok=True)
    
    # 创建checkpoints目录 - 每100步保存一次
    checkpoint_dir = os.path.join(results_base, "checkpoints_gpt2_medium", task_safe, f"wd_{wd}")
    os.makedirs(checkpoint_dir, exist_ok=True)

    metrics = {'steps': [], 'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': [], 'llc': [], 'l2_norm': [], 'spectral_entropy': []}
    steps = 0
    
    # 创建进度条 (双线程模式，禁用内部进度条避免混乱)
    # pbar = tqdm(total=args.budget, desc=f"GPT-2-Medium {task_name} WD={wd} S={seed}", 
    #             unit="step", ncols=120, leave=False, disable=True)

    while steps < args.budget:
        for (batch_x,) in train_loader:
            model.train()
            batch_x = batch_x.to(device)
            inp, target = batch_x[:, :-1], batch_x[:, -1]

            # 使用 FP16 自动混合精度 + Tensor Core 加速
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                logits = model(inp)
                loss = F.cross_entropy(logits[:, -1, :], target)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            
            # Gradient clipping for stability in large models
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()
            
            steps += 1
            # pbar.update(1)  # 双线程模式下禁用
            
            if steps % 100 == 0:
                # 验证
                acc = (logits[:, -1, :].argmax(-1) == target).float().mean().item()
                loss_val = loss.item()
                
                model.eval()
                with torch.no_grad(), torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                    for (test_batch,) in test_loader:
                        test_batch = test_batch.to(device)
                        t_inp, t_target = test_batch[:, :-1], test_batch[:, -1]
                        t_logits = model(t_inp)
                        test_loss = F.cross_entropy(t_logits[:, -1, :], t_target).item()
                        test_acc = (t_logits[:, -1, :].argmax(-1) == t_target).float().mean().item()
                
                # LLC
                llc_val = llc_estimator.estimate(model.state_dict(), num_draws=20, lr=1e-4)

                # L2 Norm
                l2_val = calc_l2_norm(model)
                
                # Spectral Entropy
                spectral_entropy_val = calc_spectral_entropy(model)

                metrics['steps'].append(steps)
                metrics['train_loss'].append(loss_val)
                metrics['train_acc'].append(acc)
                metrics['test_loss'].append(test_loss)
                metrics['test_acc'].append(test_acc)
                metrics['llc'].append(llc_val)
                metrics['l2_norm'].append(l2_val)
                metrics['spectral_entropy'].append(spectral_entropy_val)
                
                # 保存模型权重（每100步）
                if steps in checkpoint_steps:
                    checkpoint_path = os.path.join(checkpoint_dir, f"seed{seed}_step{steps}.pt")
                    step_data = {
                        'step': steps,
                        'train_loss': loss_val,
                        'train_acc': acc,
                        'test_loss': test_loss,
                        'test_acc': test_acc,
                        'llc': llc_val,
                        'l2_norm': l2_val,
                        'spectral_entropy': spectral_entropy_val
                    }
                    torch.save({
                        'step': steps,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'metrics': step_data,
                        'task': task_name,
                        'weight_decay': wd,
                        'seed': seed
                    }, checkpoint_path)
                
                # # 更新进度条显示关键指标 (双线程模式下禁用)
                # pbar.set_postfix({
                #     'TrAcc': f'{acc:.3f}',
                #     'TeAcc': f'{test_acc:.3f}',
                #     'Loss': f'{loss_val:.3f}',
                #     'LLC': f'{llc_val:.2f}',
                #     'L2': f'{l2_val:.1f}'
                # })

            if steps >= args.budget:
                break
    
    # # 关闭进度条 (双线程模式下禁用)
    # pbar.close()
                
    # 保存数据到统一的results目录
    csv_path = os.path.join(data_dir, f"seed{seed}.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(metrics.keys())
        writer.writerows(zip(*metrics.values()))
    
    # 获取最终指标
    final_train_acc = metrics['train_acc'][-1]
    final_test_acc = metrics['test_acc'][-1]
    
    # 统计保存的checkpoints数量
    saved_ckpts = len(checkpoint_steps.intersection(set(metrics['steps'])))
    
    print(f"✓ Completed: {task_name} | WD={wd} | S={seed} | Train={final_train_acc:.3f} | Test={final_test_acc:.3f} | Ckpts={saved_ckpts}")
    
    # 清理资源，释放显存
    del model, optimizer, scaler, llc_estimator
    torch.cuda.empty_cache()
    
    return (task_name, wd, metrics)

# =============================================================================
# 6. 绘图函数
# =============================================================================

def plot_multiseed_results(results_list, task_name, wd, save_dir):
    steps = results_list[0]['steps']
    
    def get_stats(key):
        data = np.array([r[key] for r in results_list])
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        return mean, std

    tr_acc_m, tr_acc_s = get_stats('train_acc')
    te_acc_m, te_acc_s = get_stats('test_acc')
    tr_loss_m, tr_loss_s = get_stats('train_loss')
    te_loss_m, te_loss_s = get_stats('test_loss')
    l2_m, l2_s = get_stats('l2_norm')
    se_m, se_s = get_stats('spectral_entropy')
    
    llc_raw = np.array([r['llc'] for r in results_list])
    llc_m = np.nanmean(llc_raw, axis=0)
    llc_s = np.nanstd(llc_raw, axis=0)

    fig, axes = plt.subplots(1, 5, figsize=(32, 5))
    ax1, ax2, ax3, ax4, ax5 = axes
    
    # Acc
    ax1.plot(steps, tr_acc_m, color='blue', label='Train')
    ax1.fill_between(steps, tr_acc_m-tr_acc_s, tr_acc_m+tr_acc_s, color='blue', alpha=0.2)
    ax1.plot(steps, te_acc_m, color='red', label='Test')
    ax1.fill_between(steps, te_acc_m-te_acc_s, te_acc_m+te_acc_s, color='red', alpha=0.2)
    ax1.set_title(f'Accuracy (WD={wd})')
    ax1.set_xlabel('Steps')
    ax1.set_xscale('log')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Loss
    ax2.plot(steps, tr_loss_m, color='blue')
    ax2.fill_between(steps, tr_loss_m-tr_loss_s, tr_loss_m+tr_loss_s, color='blue', alpha=0.2)
    ax2.plot(steps, te_loss_m, color='red')
    ax2.fill_between(steps, te_loss_m-te_loss_s, te_loss_m+te_loss_s, color='red', alpha=0.2)
    ax2.set_title('Loss')
    ax2.set_xlabel('Steps')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3)

    # LLC
    valid_mask = ~np.isnan(llc_m)
    v_steps = np.array(steps)[valid_mask]
    v_llc_m = llc_m[valid_mask]
    v_llc_s = llc_s[valid_mask]
    ax3.plot(v_steps, v_llc_m, color='green', label='LLC')
    ax3.fill_between(v_steps, v_llc_m-v_llc_s, v_llc_m+v_llc_s, color='green', alpha=0.2)
    ax3.set_title('Complexity (LLC)')
    ax3.set_xlabel('Steps')
    ax3.set_ylabel('Complexity')
    ax3.set_xscale('log')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # L2 Norm
    ax4.plot(steps, l2_m, color='purple', label='L2 Norm')
    ax4.fill_between(steps, l2_m-l2_s, l2_m+l2_s, color='purple', alpha=0.2)
    ax4.set_title('Parameter Norm (L2)')
    ax4.set_xlabel('Steps')
    ax4.set_ylabel('L2 Norm')
    ax4.set_xscale('log')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # Spectral Entropy
    ax5.plot(steps, se_m, color='orange', label='Spectral Entropy')
    ax5.fill_between(steps, se_m-se_s, se_m+se_s, color='orange', alpha=0.2)
    ax5.set_title('Spectral Entropy')
    ax5.set_xlabel('Steps')
    ax5.set_ylabel('Entropy')
    ax5.set_xscale('log')
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    plt.suptitle(rf"GPT-2-Medium (1024d, 24L, 16H) | Task: {task_name} | WD: {wd} | (Single seed)")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"gpt2_medium_{task_name.replace('/','_')}_wd{wd}.png"), dpi=150)
    plt.close()

# =============================================================================
# 7. 调度器
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--p", type=int, default=97)
    parser.add_argument("--budget", type=int, default=100000) 
    parser.add_argument("--batch_size", type=int, default=128, help="Optimized for GPT-2-Medium with memory efficiency")
    parser.add_argument("--lr", type=float, default=1e-3)
    # 双线程并行执行 (GPT-2 Medium ~8-10GB per model, 4090有24GB显存)
    parser.add_argument("--max_workers", type=int, default=2, help="2 parallel workers to max out RTX 4090")
    args = parser.parse_args()

    seeds = [42]  # 单种子
    weight_decays = [0.0, 1.0]
    tasks = ['x-y']  # 只跑x-y任务
    
    # 使用绝对路径避免子进程工作目录问题
    base_dir = os.path.abspath(f"gpt2_medium_grokking_{datetime.now().strftime('%m%d_%H%M')}")
    os.makedirs(base_dir, exist_ok=True)

    # 1. 构建任务队列
    task_queue = []
    for task in tasks:
        task_safe = task.replace("/", "_div_").replace("*", "_mul_").replace("+", "_plus_")
        for wd in weight_decays:
            out_dir = os.path.join(base_dir, task_safe, f"wd_{wd}")
            os.makedirs(out_dir, exist_ok=True)
            for seed in seeds:
                # 单线程顺序执行，GPU 0
                gpu_id = 0
                config = (task, wd, seed, gpu_id, args, out_dir)
                task_queue.append(config)

    print(f"Total experiments: {len(task_queue)} (GPT-2-Medium Architecture)")
    print(f"Model config: 1024 dim, 24 layers, 16 heads (~345M params)")
    print(f"Task: x-y only | Seeds: {seeds} | WD: {weight_decays}")
    print(f"Execution mode: {args.max_workers}x Parallel (max out RTX 4090)")
    print(f"Memory: ~8-10GB per model × {args.max_workers} = ~{8*args.max_workers}-{10*args.max_workers}GB (24GB available)")
    print(f"Batch size: {args.batch_size} (optimized for memory + Tensor Cores)")
    print(f"Output dir: {base_dir}")
    print(f"\n{'='*70}")
    print("Performance Optimizations Enabled:")
    print("  ✓ FP16 Mixed Precision (torch.amp)")
    print("  ✓ Tensor Core Acceleration (NVIDIA Ampere)")
    print("  ✓ TF32 enabled for matmul")
    print("  ✓ cuDNN benchmark mode")
    print("  ✓ Gradient Checkpointing (memory efficient)")
    print(f"  ✓ {args.max_workers}x Parallel Execution (RTX 4090 fully utilized)")
    print(f"{'='*70}")
    print("Data & Checkpoint Saving:")
    print("  CSV Data: Save every 100 steps (all metrics)")
    print("    → Location: /root/autodl-tmp/test/results/data_gpt2_medium/")
    print("  Model Weights: Save at steps [100, 1000, 10000, 100000] only")
    print("    → Location: /root/autodl-tmp/test/results/checkpoints_gpt2_medium/")
    print("    → Total checkpoints: 4 steps × 2 experiments = 8 files")
    print(f"{'='*70}")
    print("Starting GPT-2-Medium training (x-y task, 2 experiments)...")
    print(f"{'='*70}\n")
    
    results_cache = {} 
    
    try:
        multiprocessing.set_start_method('spawn')
    except RuntimeError:
        pass

    # 双线程并行执行 - 充分利用RTX 4090
    with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        # 提交所有任务
        futures = [executor.submit(run_atomic_experiment, cfg) for cfg in task_queue]
        
        # 使用进度条跟踪完成情况
        for future in tqdm(futures, total=len(futures), desc="Overall Progress", 
                          ncols=100, colour='green'):
            try:
                task_name, wd, metrics = future.result()
                key = (task_name, wd)
                if key not in results_cache: results_cache[key] = []
                results_cache[key].append(metrics)
            except Exception as e:
                print(f"\n!!! Error: {e}")
                import traceback
                traceback.print_exc()

    # 汇总绘图
    print("\nGenerating plots...")
    for (task, wd), metrics_list in results_cache.items():
        if len(metrics_list) >= 1:  # 只要有数据就绘图
            task_safe = task.replace("/", "_div_").replace("*", "_mul_").replace("+", "_plus_")
            save_dir = os.path.join(base_dir, task_safe, f"wd_{wd}")
            plot_multiseed_results(metrics_list, task, wd, save_dir)
            print(f"✓ Plot saved: {task} WD={wd}")
        else:
            print(f"⚠ Skipping plot for {task} WD={wd} (no data)")

    print(f"\n{'='*70}")
    print(f"All Done! GPT-2-Medium training completed.")
    print(f"Results saved to:")
    print(f"  - CSV files: /root/autodl-tmp/test/results/data_gpt2_medium/")
    print(f"  - Checkpoints: /root/autodl-tmp/test/results/checkpoints_gpt2_medium/")
    print(f"  - Plots: {base_dir}")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()

