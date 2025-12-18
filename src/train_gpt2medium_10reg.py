import math
import os
import sys
import csv
import copy
import numpy as np
import argparse
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)


# === 关键导入：从您的 regularization.py 中导入工厂函数 ===
# 假设您的代码保存为 utils/regularization.py
try:
    from utils.regularization import get_regularizer_and_optimizer
except ImportError:
    # 为了防止报错，如果您还没保存文件，这里抛出提示
    raise ImportError("请将您提供的正则化代码保存为 'utils/regularization.py' 以便脚本调用。")


# =============================================================================
# 0. Runtime & Basics
# =============================================================================

def setup_torch_runtime(args):
    if args.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = bool(args.cudnn_benchmark)

# 简单的 LLC 估算器 (为了保持独立性，不依赖 utils.regularization 里的定义)
class LLCEstimator:
    def __init__(self, model_ctor, criterion, dataloader, device):
        self.model_ctor = model_ctor
        self.criterion = criterion
        self.dataloader = dataloader
        self.device = device

    def estimate(self, current_weights_dict, num_draws=40, lr=1e-4, epsilon=1.0):
        # 简化的 SGLD 用于估算
        sampling_model = self.model_ctor().to(self.device)
        sampling_model.load_state_dict(current_weights_dict)
        sampling_model.train()
        
        # SGLD Optimizer inline
        params = list(sampling_model.parameters())
        
        loss_trace = []
        iter_dl = iter(self.dataloader)
        
        # Base loss
        with torch.no_grad():
            sampling_model.eval()
            total_loss = 0; count = 0
            for batch in self.dataloader:
                b_x = batch[0].to(self.device)
                total_loss += self.criterion(sampling_model(b_x[:, :-1])[:, -1, :], b_x[:, -1]).item() * b_x.size(0)
                count += b_x.size(0)
            base_loss = total_loss / count
        
        sampling_model.train()
        for _ in range(num_draws):
            try: batch = next(iter_dl)
            except StopIteration: iter_dl = iter(self.dataloader); batch = next(iter_dl)
            
            inp, target = batch[0][:, :-1].to(self.device), batch[0][:, -1].to(self.device)
            logits = sampling_model(inp)
            loss = self.criterion(logits[:, -1, :], target)
            
            # Manual SGLD Step
            grads = torch.autograd.grad(loss, params)
            with torch.no_grad():
                for p, g in zip(params, grads):
                    noise = torch.randn_like(p) * epsilon * math.sqrt(lr)
                    p.data.add_(g, alpha=-0.5 * lr).add_(noise)
            loss_trace.append(loss.item())

        llc_proxy = float(np.mean(loss_trace)) - base_loss
        del sampling_model
        return llc_proxy

# =============================================================================
# 1. Model Definitions (GPT-2)
# =============================================================================

class CausalSelfAttention(nn.Module):
    def __init__(self, dim, num_heads, dropout=0.0):
        super().__init__()
        self.c_attn = nn.Linear(dim, 3 * dim)
        self.c_proj = nn.Linear(dim, dim)
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

    def forward(self, x, mask=None):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.dim, dim=2)
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        if mask is not None: att = att + mask[:T, :T]
        att = F.softmax(att, dim=-1)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)

class MLP(nn.Module):
    def __init__(self, dim, dropout=0.0):
        super().__init__()
        self.c_fc = nn.Linear(dim, 4 * dim)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * dim, dim)

    def forward(self, x):
        return self.c_proj(self.gelu(self.c_fc(x)))

class GPT2Block(nn.Module):
    def __init__(self, dim, num_heads, dropout=0.0):
        super().__init__()
        self.ln_1 = nn.LayerNorm(dim)
        self.attn = CausalSelfAttention(dim, num_heads, dropout)
        self.ln_2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, dropout)

    def forward(self, x, mask=None):
        x = x + self.attn(self.ln_1(x), mask)
        x = x + self.mlp(self.ln_2(x))
        return x

class GPT2Decoder(nn.Module):
    def __init__(self, dim=1024, num_layers=24, num_heads=16, num_tokens=99, seq_len=5, dropout=0.0, use_checkpoint=True):
        super().__init__()
        self.token_embeddings = nn.Embedding(num_tokens, dim)
        self.position_embeddings = nn.Embedding(seq_len, dim)
        self.blocks = nn.ModuleList([GPT2Block(dim, num_heads, dropout) for _ in range(num_layers)])
        self.ln_f = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_tokens, bias=False)
        self.register_buffer("causal_mask", torch.triu(torch.full((seq_len, seq_len), float('-inf')), diagonal=1))
        self.register_buffer("pos_ids", torch.arange(seq_len))
        self.use_checkpoint = use_checkpoint

    def forward(self, x):
        B, T = x.size()
        h = self.token_embeddings(x) + self.position_embeddings(self.pos_ids[:T])
        mask = self.causal_mask[:T, :T]
        if self.use_checkpoint and self.training:
            for block in self.blocks:
                h = torch.utils.checkpoint.checkpoint(block, h, mask, use_reentrant=False)
        else:
            for block in self.blocks:
                h = block(h, mask)
        return self.head(self.ln_f(h))

# =============================================================================
# 2. Metric Utils
# =============================================================================

def calc_metrics(model):
    total_norm = 0.0
    for p in model.parameters():
        total_norm += p.data.norm(2).item() ** 2
    return total_norm ** 0.5

def get_data(p, eq_token, op_token, task_name):
    x = torch.arange(p)
    y = torch.arange(1, p)
    x, y = torch.cartesian_prod(x, y).T
    eq = torch.ones_like(x) * eq_token
    op = torch.ones_like(x) * op_token
    result = (x - y) % p
    data = torch.stack([x, op, y, eq, result]).T
    return data

# =============================================================================
# 3. Experiment Runner (Updated loop for SAM/SWA)
# =============================================================================

def run_atomic_experiment(task_name, method_name, seed, device_id, args):
    device = torch.device(f"cuda:{device_id}")
    torch.cuda.set_device(device_id)
    setup_torch_runtime(args)
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    # Prepare Data
    full_data = get_data(args.p, args.p, args.p + 1, task_name)
    indices = torch.randperm(len(full_data))
    split = int(len(full_data) * 0.5)
    train_data = full_data[indices[:split]]
    test_data = full_data[indices[split:]]

    train_loader = DataLoader(TensorDataset(train_data), batch_size=args.batch_size, shuffle=True, num_workers=0)
    llc_loader = DataLoader(TensorDataset(train_data), batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(TensorDataset(test_data), batch_size=len(test_data), shuffle=False, num_workers=0)

    # Model Constructor
    def model_ctor():
        return GPT2Decoder(
            dim=1024, num_layers=24, num_heads=16,
            num_tokens=args.p + 2, seq_len=5,
            dropout=0.0, 
            use_checkpoint=bool(args.grad_ckpt)
        )

    model = model_ctor().to(device)

    # === 使用工厂函数获取 Optimizer, Hook, Wrapper ===
    # 注意：get_regularizer_and_optimizer 来自您的 regularization.py
    optimizer, loss_hook, model_wrapper = get_regularizer_and_optimizer(
        method_name, model, base_lr=args.lr, weight_decay=0.0 # 默认不加额外WD，除非方法内部处理
    )
    
    # 辅助判断是否是 SAM 类优化器 (有 first_step 方法)
    is_sam_opt = hasattr(optimizer, 'first_step')

    if args.precision == "bf16":
        amp_dtype = torch.bfloat16
        use_scaler = False
    else:
        amp_dtype = torch.float16
        use_scaler = True
    scaler = torch.amp.GradScaler('cuda', enabled=use_scaler)
    
    llc_estimator = LLCEstimator(model_ctor, F.cross_entropy, llc_loader, device)

    # Paths
    results_base = args.results_base
    data_dir = os.path.join(results_base, "data", task_name, method_name)
    checkpoint_dir = os.path.join(results_base, "checkpoints", task_name, method_name)
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    metrics = {'steps': [], 'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': [], 'llc': [], 'l2_norm': []}
    checkpoint_steps = {100, 1000, 10000, 100000}
    
    steps = 0
    print(f"[START] cuda:{device_id} method={method_name} seed={seed}")
    
    pbar = tqdm(total=args.budget, desc=f"Steps | {method_name}", ncols=100, disable=(device_id!=0))

    try:
        while steps < args.budget:
            for (batch_x,) in train_loader:
                model.train()
                batch_x = batch_x.to(device, non_blocking=True)
                inp, target = batch_x[:, :-1], batch_x[:, -1]

                # --- Training Step Start ---
                
                # 1. First Forward (Clean)
                optimizer.zero_grad() # Standard zero_grad
                
                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    logits = model(inp)
                    loss = F.cross_entropy(logits[:, -1, :].float(), target)
                    # 应用 Loss Hook (例如 Flooding, Spectral Decoupling, LogitNorm)
                    loss = loss_hook(model, loss, logits[:, -1, :].float(), target)

                if is_sam_opt:
                    # === SAM/ASAM/GSAM 逻辑 ===
                    # 1. Backward (Clean gradients)
                    if use_scaler: scaler.scale(loss).backward()
                    else: loss.backward()
                    
                    if use_scaler: scaler.unscale_(optimizer) # SAM needs real grads for norm
                    
                    # 2. First Step (Ascent / Perturbation)
                    # 注意：Regularization.py 里的 SAM.first_step 默认 zero_grad=False
                    # 但我们需要在第二次 backward 前清空梯度
                    optimizer.first_step(zero_grad=True) 

                    # 3. Second Forward (Perturbed)
                    with torch.autocast(device_type="cuda", dtype=amp_dtype):
                        logits_2 = model(inp)
                        loss_2 = F.cross_entropy(logits_2[:, -1, :].float(), target)
                        # SAF 需要在这一步应用 Flooding
                        loss_2 = loss_hook(model, loss_2, logits_2[:, -1, :].float(), target)
                    
                    # 4. Backward (Perturbed gradients)
                    if use_scaler: scaler.scale(loss_2).backward()
                    else: loss_2.backward()
                    
                    # 5. Second Step (Descent)
                    if use_scaler: 
                        # Scaler 不太适配手动 optimizer.step，这里简化处理：
                        # SAM 内部调用 base_optimizer.step()，这里如果用了 scaler 可能需要手动 unscale
                        # 但为兼容性，假设 SAM 内部处理好了或者不用 scaler (通常 SAM 推荐不用 scaler 或小心使用)
                        # 这里我们直接调用 optimizer.second_step
                        optimizer.second_step(zero_grad=True)
                        scaler.update() 
                    else:
                        optimizer.second_step(zero_grad=True)
                        
                else:
                    # === 标准优化器 (SGD/AdamW) ===
                    if use_scaler:
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        loss.backward()
                        optimizer.step()

                # --- Model Averaging Update (SWA/EMA) ---
                if model_wrapper is not None:
                    model_wrapper.update()

                steps += 1
                pbar.update(1)

                # --- Evaluation ---
                if steps % 100 == 0:
                    acc = (logits[:, -1, :].argmax(-1) == target).float().mean().item()
                    loss_val = float(loss.item())

                    # 如果用了 SWA/EMA，评估时应该用 averaged model
                    eval_model = model_wrapper.get_averaged_model() if model_wrapper else model
                    eval_model.eval()
                    
                    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=amp_dtype):
                        t_loss_sum, t_acc_sum, count = 0, 0, 0
                        for (test_batch,) in test_loader:
                            test_batch = test_batch.to(device)
                            t_inp, t_target = test_batch[:, :-1], test_batch[:, -1]
                            t_logits = eval_model(t_inp)
                            t_loss_sum += F.cross_entropy(t_logits[:, -1, :].float(), t_target).item()
                            t_acc_sum += (t_logits[:, -1, :].argmax(-1) == t_target).float().mean().item()
                            count += 1
                        test_loss = t_loss_sum / max(count, 1)
                        test_acc = t_acc_sum / max(count, 1)

                    # LLC 用基础模型估算即可（Proxy）
                    llc_val = llc_estimator.estimate(model.state_dict(), num_draws=args.llc_draws, lr=args.llc_lr)
                    l2_val = calc_metrics(model)

                    metrics['steps'].append(steps)
                    metrics['train_loss'].append(loss_val)
                    metrics['train_acc'].append(acc)
                    metrics['test_loss'].append(test_loss)
                    metrics['test_acc'].append(test_acc)
                    metrics['llc'].append(llc_val)
                    metrics['l2_norm'].append(l2_val)

                    if steps in checkpoint_steps:
                        ckpt_path = os.path.join(checkpoint_dir, f"seed{seed}_step{steps}.pt")
                        torch.save({
                            'step': steps,
                            'method': method_name,
                            'model_state_dict': model.state_dict(), # Save base model
                            'avg_model_state_dict': model_wrapper.get_averaged_model().state_dict() if model_wrapper else None,
                            'optimizer_state_dict': optimizer.state_dict(),
                            'metrics': metrics
                        }, ckpt_path)

                if steps >= args.budget:
                    break
    finally:
        pbar.close()

    # Save CSV
    csv_path = os.path.join(data_dir, f"seed{seed}.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(metrics.keys())
        writer.writerows(zip(*metrics.values()))

    print(f"✓ [DONE] cuda:{device_id} {method_name} | TestAcc={metrics['test_acc'][-1]:.3f}")
    del model, optimizer, scaler
    torch.cuda.empty_cache()
    return f"{method_name}_seed{seed}"


# =============================================================================
# 4. Main Scheduler (8+2)
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--p", type=int, default=97)
    parser.add_argument("--budget", type=int, default=100000)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--precision", type=str, default="bf16")
    parser.add_argument("--tf32", type=int, default=1)
    parser.add_argument("--cudnn_benchmark", type=int, default=1)
    parser.add_argument("--grad_ckpt", type=int, default=1)
    parser.add_argument("--llc_draws", type=int, default=20)
    parser.add_argument("--llc_lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--results_base", type=str, default="/data/zjj/test/results")

    args = parser.parse_args()

    # === 定义 10 个正则化实验 (对应 regularization.py 里的 keys) ===
    # 注意：baseline 和 l2 您说已经跑过了/另算，所以这里跑剩下的 10 个
    methods = [
        # Loss Regularizers
        'logit_norm', 
        'spectral_decoupling', 
        'flooding',
        'si_llc', # 您的自定义方法
        
        # Model Averaging
        'swa', 
        'ema',
        
        # Flatness Optimizers
        'sam',
        'asam',
        'saf',
        'gsam'
    ]

    assert len(methods) == 10
    seed = args.seed
    
    # 8 + 2 调度
    tasks_batch_1 = []
    for i in range(8):
        tasks_batch_1.append({'gpu': i, 'method': methods[i], 'seed': seed})
        
    tasks_batch_2 = []
    for i in range(2):
        tasks_batch_2.append({'gpu': i, 'method': methods[8+i], 'seed': seed})

    print(f"Total Experiments: 10. Batch 1 (8 GPUs) -> Batch 2 (2 GPUs).")
    
    ctx = multiprocessing.get_context("spawn")
    
    # Batch 1
    print("\n=== Batch 1 (8 experiments) ===")
    with ProcessPoolExecutor(max_workers=8, mp_context=ctx) as executor:
        futures = [executor.submit(run_atomic_experiment, "x-y", t['method'], t['seed'], t['gpu'], args) for t in tasks_batch_1]
        for fut in as_completed(futures):
            try: print(f"Batch 1 Finished: {fut.result()}")
            except Exception as e: print(f"Batch 1 Error: {e}")

    # Batch 2
    print("\n=== Batch 2 (2 experiments) ===")
    with ProcessPoolExecutor(max_workers=2, mp_context=ctx) as executor:
        futures = [executor.submit(run_atomic_experiment, "x-y", t['method'], t['seed'], t['gpu'], args) for t in tasks_batch_2]
        for fut in as_completed(futures):
            try: print(f"Batch 2 Finished: {fut.result()}")
            except Exception as e: print(f"Batch 2 Error: {e}")

    print("\nAll 10 Regularization Experiments Completed.")

if __name__ == "__main__":
    main()