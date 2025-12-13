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
# 1. 基础组件 (保持不变)
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
# 2. 工具函数
# =============================================================================

def calc_l2_norm(model):
    """计算模型所有参数的 L2 范数"""
    total_norm = 0.0
    for p in model.parameters():
        param_norm = p.data.norm(2)
        total_norm += param_norm.item() ** 2
    return total_norm ** 0.5

# =============================================================================
# 3. 数据生成 (保持不变)
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
# 3. 原子训练任务 (Worker Logic) - 已优化
# =============================================================================

def run_atomic_experiment(config):
    """原子任务：无进度条，只输出 Log"""
    task_name, wd, seed, device_id, args, output_dir = config
    device = torch.device(f"cuda:{device_id}")
    
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

    model = Decoder(dim=128, num_layers=2, num_heads=4, num_tokens=args.p+2, seq_len=5).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=wd, betas=(0.9, 0.98))
    scaler = torch.amp.GradScaler('cuda')  # 修复 FutureWarning
    llc_estimator = LLCEstimator(model, F.cross_entropy, llc_loader, device)

    metrics = {'steps': [], 'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': [], 'llc': [], 'l2_norm': []}
    steps = 0
    
    # [Log] 开始
    # print(f"[START] GPU:{device_id} {task_name} WD={wd} S={seed}")

    while steps < args.budget:
        for (batch_x,) in train_loader:
            model.train()
            batch_x = batch_x.to(device)
            inp, target = batch_x[:, :-1], batch_x[:, -1]

            with torch.amp.autocast(device_type='cuda'):
                logits = model(inp)
                loss = F.cross_entropy(logits[:, -1, :], target)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            steps += 1
            
            if steps % 100 == 0:
                # 验证
                acc = (logits[:, -1, :].argmax(-1) == target).float().mean().item()
                loss_val = loss.item()
                
                model.eval()
                with torch.no_grad():
                    for (test_batch,) in test_loader:
                        test_batch = test_batch.to(device)
                        t_inp, t_target = test_batch[:, :-1], test_batch[:, -1]
                        t_logits = model(t_inp)
                        test_loss = F.cross_entropy(t_logits[:, -1, :], t_target).item()
                        test_acc = (t_logits[:, -1, :].argmax(-1) == t_target).float().mean().item()
                
                # LLC
                llc_val = np.nan
                if steps % 100 == 0 or steps == args.budget:
                    llc_val = llc_estimator.estimate(model.state_dict(), num_draws=20, lr=1e-4)
                elif len(metrics['llc']) > 0:
                    llc_val = metrics['llc'][-1]

                # L2 Norm
                l2_val = calc_l2_norm(model)

                metrics['steps'].append(steps)
                metrics['train_loss'].append(loss_val)
                metrics['train_acc'].append(acc)
                metrics['test_loss'].append(test_loss)
                metrics['test_acc'].append(test_acc)
                metrics['llc'].append(llc_val)
                metrics['l2_norm'].append(l2_val)

            if steps >= args.budget:
                break
                
    # 保存数据
    csv_name = f"seed{seed}.csv"
    with open(os.path.join(output_dir, csv_name), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(metrics.keys())
        writer.writerows(zip(*metrics.values()))
        
    # [Log] 结束
    # print(f"[DONE]  GPU:{device_id} {task_name} WD={wd} S={seed}")
    return (task_name, wd, metrics)

# =============================================================================
# 4. 绘图函数
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
    
    llc_raw = np.array([r['llc'] for r in results_list])
    llc_m = np.nanmean(llc_raw, axis=0)
    llc_s = np.nanstd(llc_raw, axis=0)

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(26, 5))
    
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

    plt.suptitle(rf"Task: {task_name} | WD: {wd} | (Mean $\pm$ Std over 3 seeds)")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{task_name.replace('/','_')}_wd{wd}.png"), dpi=150)
    plt.close()

# =============================================================================
# 5. 调度器 (Max Parallelism)
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--p", type=int, default=97)
    parser.add_argument("--budget", type=int, default=100000) 
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    # 54个并发进程 - 平衡GPU利用率和稳定性
    # 如果遇到 CUDA out of memory，请手动改小这个数字 (例如 32)
    parser.add_argument("--max_workers", type=int, default=54, help="Max parallel processes")
    args = parser.parse_args()

    seeds = [42, 101, 2025]
    weight_decays = [0.0, 1.0]
    tasks = [
        'x+y', 'x-y', 'x*y', 'x_div_y',
        'x2+y2', 'x2+xy+y2', 'x2+xy+y2+x',
        'x3+xy', 'x3+xy2+y'
    ]
    
    # 使用绝对路径避免子进程工作目录问题
    base_dir = os.path.abspath(f"grokking_full_parallel_{datetime.now().strftime('%m%d_%H%M')}")
    os.makedirs(base_dir, exist_ok=True)

    # 1. 构建任务队列
    task_queue = []
    for task in tasks:
        task_safe = task.replace("/", "_div_").replace("*", "_mul_").replace("+", "_plus_")
        for wd in weight_decays:
            out_dir = os.path.join(base_dir, task_safe, f"wd_{wd}")
            os.makedirs(out_dir, exist_ok=True)
            for seed in seeds:
                # 均匀分配 GPU: 0, 1, 0, 1 ...
                gpu_id = len(task_queue) % 2 
                config = (task, wd, seed, gpu_id, args, out_dir)
                task_queue.append(config)

    print(f"Total experiments: {len(task_queue)}")
    print(f"Max workers: {args.max_workers}")
    print(f"Batch size: {args.batch_size}")
    print(f"Output dir: {base_dir}")
    print("Starting parallel training...")

    results_cache = {} 
    
    try:
        multiprocessing.set_start_method('spawn')
    except RuntimeError:
        pass

    # 使用 ProcessPoolExecutor 拉满并发
    with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        futures = [executor.submit(run_atomic_experiment, cfg) for cfg in task_queue]
        
        # 总进度条
        for future in tqdm(futures, total=len(futures), desc="All Experiments"):
            try:
                task_name, wd, metrics = future.result()
                key = (task_name, wd)
                if key not in results_cache: results_cache[key] = []
                results_cache[key].append(metrics)
            except Exception as e:
                print(f"\n!!! Error: {e}")

    # 汇总绘图
    print("\nGenerating plots...")
    for (task, wd), metrics_list in results_cache.items():
        if len(metrics_list) == len(seeds):
            task_safe = task.replace("/", "_div_").replace("*", "_mul_").replace("+", "_plus_")
            save_dir = os.path.join(base_dir, task_safe, f"wd_{wd}")
            plot_multiseed_results(metrics_list, task, wd, save_dir)
        else:
            print(f"Skipping plot for {task} WD={wd} (incomplete data)")

    print(f"\nAll Done! Results in {base_dir}")

if __name__ == "__main__":
    main()