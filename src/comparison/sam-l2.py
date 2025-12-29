import math
import os
import csv
import numpy as np
import argparse
from datetime import datetime
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import torch.multiprocessing as mp
from tqdm import tqdm

# =============================================================================
# 0. SAM Optimizer Implementation (Standard)
# =============================================================================

class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)  # Climb to the local maximum

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.data = self.state[p]["old_p"]  # Get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # Do the actual "w" update

        if zero_grad: self.zero_grad()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
                    torch.stack([
                        ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
               )
        return norm
    
    def zero_grad(self, set_to_none=False):
        self.base_optimizer.zero_grad(set_to_none=set_to_none)

# =============================================================================
# 1. Model & Data
# =============================================================================

def setup_torch_runtime(args):
    if args.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    try:
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    except Exception:
        pass

class Decoder(nn.Module):
    def __init__(self, dim=128, num_layers=2, num_heads=4, num_tokens=97, seq_len=5):
        super().__init__()
        self.token_embeddings = nn.Embedding(num_tokens, dim)
        self.position_embeddings = nn.Embedding(seq_len, dim)
        layer = nn.TransformerEncoderLayer(d_model=dim, nhead=num_heads, dim_feedforward=4*dim, 
                                         dropout=0.0, activation='relu', batch_first=True, norm_first=True)
        self.layers = nn.TransformerEncoder(layer, num_layers=num_layers, enable_nested_tensor=False)
        self.ln_f = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_tokens, bias=False)
        self.register_buffer("pos_ids", torch.arange(seq_len))

    def forward(self, x):
        h = self.token_embeddings(x) + self.position_embeddings(self.pos_ids[:x.size(1)])
        h = self.layers(h)
        return self.head(self.ln_f(h))

def get_data_xy(p):
    x = torch.arange(p)
    y = torch.arange(p)
    x, y = torch.cartesian_prod(x, y).T
    result = (x - y) % p 
    return torch.stack([x, torch.full_like(x, p), y, torch.full_like(x, p+1), result]).T

# =============================================================================
# 2. Training Worker
# =============================================================================

def train_worker(config):
    method_name, gpu_id, args, output_dir = config
    device = torch.device(f"cuda:{gpu_id}")
    torch.cuda.set_device(gpu_id)
    setup_torch_runtime(args)
    
    seed = 2025
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    print(f"🚀 [Start] {method_name} on GPU {gpu_id}")

    # Data
    full_data = get_data_xy(args.p)
    indices = torch.randperm(len(full_data))
    split = int(len(full_data) * 0.5)
    train_data = full_data[indices[:split]]
    test_data = full_data[indices[split:]]

    train_loader = DataLoader(TensorDataset(train_data), batch_size=args.batch_size, shuffle=True, pin_memory=True)
    test_loader = DataLoader(TensorDataset(test_data), batch_size=2048, shuffle=False)

    model = Decoder(num_tokens=args.p+2).to(device)
    
    # === Optimizer Config ===
    optimizer = None
    
    if method_name == "Baseline_L2":
        # 基线：AdamW, WD=1.0 (Strong L2)
        optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=args.lr, 
            weight_decay=1.0, 
            betas=(0.9, 0.98)
        )
        
    elif method_name == "SAM_NoL2":
        # 实验组：SAM wrapping Adam, WD=0.0 (No L2)
        # 验证 SAM 是否能独自诱导 Grokking
        base_optimizer = torch.optim.Adam
        optimizer = SAM(
            model.parameters(), 
            base_optimizer, 
            rho=0.001, 
            lr=args.lr, 
            weight_decay=0.0, # 关键：完全关闭 L2
            betas=(0.9, 0.98)
        )
        
    else:
        raise ValueError("Unknown method")

    metrics = {'steps': [], 'train_acc': [], 'test_acc': []}
    steps = 0
    
    pbar = tqdm(total=args.budget, desc=f"{method_name}", position=gpu_id)
    
    while steps < args.budget:
        for (batch_x,) in train_loader:
            model.train()
            batch_x = batch_x.to(device)
            inp, target = batch_x[:, :-1], batch_x[:, -1]
            
            # === Training Step Logic ===
            
            if method_name == "Baseline_L2":
                # Standard Training
                optimizer.zero_grad(set_to_none=True)
                logits = model(inp)
                loss = F.cross_entropy(logits[:, -1, :], target)
                loss.backward()
                optimizer.step()
                
            elif method_name == "SAM_NoL2":
                # SAM Training (Two passes)
                
                # 1. First Step: Compute Gradient & Perturb
                # enable_math_sdp is safer for double backward/complex flows, though SAM is first order
                optimizer.zero_grad() # Use SAM's zero_grad
                logits = model(inp)
                loss = F.cross_entropy(logits[:, -1, :], target)
                loss.backward()
                optimizer.first_step(zero_grad=True) # Perturb weights
                
                # 2. Second Step: Compute Gradient at Perturbed State & Update
                logits_2 = model(inp)
                loss_2 = F.cross_entropy(logits_2[:, -1, :], target)
                loss_2.backward()
                optimizer.second_step(zero_grad=True) # Un-perturb and update
            
            steps += 1
            pbar.update(1)

            # === Evaluation ===
            if steps % 100 == 0:
                model.eval()
                with torch.no_grad():
                    tr_acc = (logits[:, -1, :].argmax(-1) == target).float().mean().item()
                    
                    te_acc_list = []
                    for (tb,) in test_loader:
                        tb = tb.to(device)
                        out = model(tb[:, :-1])
                        te_acc_list.append((out[:, -1, :].argmax(-1) == tb[:, -1]).float().mean().item())
                    te_acc = np.mean(te_acc_list)
                
                metrics['steps'].append(steps)
                metrics['train_acc'].append(tr_acc)
                metrics['test_acc'].append(te_acc)
                
                pbar.set_postfix({'Test': f'{te_acc:.3f}', 'Train': f'{tr_acc:.3f}'})
                
                # Grokking Check
                if te_acc > 0.995 and steps > 2000:
                    if len(metrics['test_acc']) > 5 and min(metrics['test_acc'][-5:]) > 0.99:
                        print(f"\n[{method_name}] Grokked at step {steps}!")
                        pbar.close()
                        return (method_name, metrics)
            
            if steps >= args.budget:
                break
                
    pbar.close()
    return (method_name, metrics)

# =============================================================================
# 3. Plotting
# =============================================================================

def plot_comparison(results_map, save_dir):
    print("\n[Plotting] Generating comparison curves...")
    plt.style.use('seaborn-v0_8-paper')
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    colors = {"Baseline_L2": "gray", "SAM_NoL2": "green"}
    labels = {"Baseline_L2": "Baseline (AdamW, WD=1.0)", "SAM_NoL2": "SAM (Adam, WD=0.0)"}
    
    for name, data in results_map.items():
        steps = data['steps']
        ax.plot(steps, data['train_acc'], linestyle='--', alpha=0.5, color=colors[name], label=f"{labels[name]} Train")
        ax.plot(steps, data['test_acc'], linestyle='-', linewidth=2, color=colors[name], label=f"{labels[name]} Test")
    
    ax.set_title("Can SAM Trigger Grokking without Weight Decay?")
    ax.set_xlabel("Steps")
    ax.set_ylabel("Accuracy")
    ax.set_xscale("log")
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = os.path.join(save_dir, "sam_vs_l2.png")
    plt.savefig(output_path, dpi=150)
    print(f"✅ Plot saved to {output_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--p", type=int, default=97)
    parser.add_argument("--budget", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--tf32", type=int, default=1)
    parser.add_argument("--results_base", type=str, default="/data/zjj/test/results/sam_test")
    args = parser.parse_args()

    os.makedirs(args.results_base, exist_ok=True)
    
    print("="*80)
    print(" 🚀 Test: SAM vs Weight Decay (Is SAM alone enough?)")
    print(f"    Task: Modular Subtraction (x - y) mod {args.p}")
    print("    GPU 0: Baseline (AdamW, WD=1.0)")
    print("    GPU 1: SAM (Adam, WD=0.0, rho=0.05)")
    print("="*80)

    tasks = [
        ("Baseline_L2", 0, args, args.results_base),
        ("SAM_NoL2",    1, args, args.results_base)
    ]

    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=2) as pool:
        results = pool.map(train_worker, tasks)
    
    results_map = {res[0]: res[1] for res in results}
    plot_comparison(results_map, args.results_base)

    for name, data in results_map.items():
        csv_path = os.path.join(args.results_base, f"{name}_metrics.csv")
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(data.keys())
            writer.writerows(zip(*data.values()))
        print(f"Data saved to {csv_path}")

if __name__ == "__main__":
    main()