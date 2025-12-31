import os
import sys
import csv
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm import tqdm
import copy
import argparse
import multiprocessing

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

# Try to import Decoder and get_data
try:
    from checkpoint_sp_l2_get import Decoder, get_data
except ImportError:
    # Fallback: Define them here if import fails (simplified version based on context)
    print("Import failed, using local definitions")
    
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
            q = self.ln_1(x)
            x = x + self.attn(q, q, q, attn_mask=attn_mask, need_weights=False)[0]
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
            x = self.token_embeddings(x) + self.position_embeddings(self.pos_ids)
            for layer in self.layers:
                x = layer(x, attn_mask=self.causal_mask)
            x = self.ln_f(x)
            return self.head(x)

    def get_data(task, batch_size=256):
        # Dummy implementation for fallback - ideally should not be reached
        # You might need to adjust paths if this is used
        pass

# =============================================================================
# Scaling Functions
# =============================================================================

def scale_model(model, alpha, beta):
    """
    Apply alpha (residual) and beta (FFN) scaling to the model in-place.
    
    Alpha scaling:
    - Scale Embedding output by alpha.
    - Scale Attention output projection by alpha.
    - Scale MLP output projection by alpha.
    - (LayerNorms are invariant, so input to next layer is effectively scaled by alpha)
    
    Beta scaling:
    - Scale MLP fc1 weights by beta, bias by beta.
    - Scale MLP fc2 weights by 1/beta.
    """
    with torch.no_grad():
        # Alpha Scaling
        # 1. Embeddings
        model.token_embeddings.weight.mul_(alpha)
        model.position_embeddings.weight.mul_(alpha)
        
        # 2. Layers
        for layer in model.layers:
            # Scale Attention Output Projection
            # nn.MultiheadAttention output projection is 'out_proj'
            layer.attn.out_proj.weight.mul_(alpha)
            if layer.attn.out_proj.bias is not None:
                layer.attn.out_proj.bias.mul_(alpha)
            
            # Scale MLP Output Projection (second linear layer)
            # mlp is [Linear, GELU, Linear, Dropout]
            # The last Linear is mlp[2]
            layer.mlp[2].weight.mul_(alpha)
            if layer.mlp[2].bias is not None:
                layer.mlp[2].bias.mul_(alpha)
                
        # Beta Scaling (FFN)
        for layer in model.layers:
            # mlp[0] is fc1, mlp[2] is fc2
            
            # Scale fc1 by beta
            layer.mlp[0].weight.mul_(beta)
            if layer.mlp[0].bias is not None:
                layer.mlp[0].bias.mul_(beta)
            
            # Scale fc2 by 1/beta (weights only, bias is output)
            # Wait, if we scale fc2 weights by 1/beta, the output is scaled by 1/beta.
            # Combined with fc1 scaled by beta:
            # y = (1/beta) * W2 * sigma(beta * W1 * x + beta * b1) + b2
            # If sigma is homogeneous (ReLU): y = (1/beta) * beta * W2 * sigma(W1 * x + b1) + b2 = Original.
            # If sigma is GELU: Approximate invariance.
            
            # Note: We already scaled mlp[2] (fc2) by alpha above.
            # So now we multiply by 1/beta.
            layer.mlp[2].weight.mul_(1.0 / beta)
            # We do NOT scale fc2 bias by 1/beta usually, as it's added after the multiplication.
            # But if we want strict invariance of the *term* W2*h, yes.
            # However, the bias is a separate term.
            # Let's stick to scaling weights.

# =============================================================================
# SGLD Optimizers
# =============================================================================

class SGLD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-4, noise_std=1e-4):
        defaults = dict(lr=lr, noise_std=noise_std)
        super().__init__(params, defaults)

    def step(self):
        for group in self.param_groups:
            lr = group['lr']
            noise_std = group['noise_std']
            for p in group['params']:
                if p.grad is None: continue
                d_p = p.grad.data
                
                # Langevin noise
                noise = torch.randn_like(p.data) * noise_std
                
                # Update: theta = theta - lr/2 * grad + sqrt(lr) * noise
                # Here we assume lr in config is epsilon.
                # Standard SGLD: d_theta = - (epsilon/2) * grad + N(0, epsilon)
                # So noise_std should be sqrt(lr).
                # But usually we parameterize with lr and noise scale explicitly or implicitly.
                # Let's follow: delta = -0.5 * lr * grad + noise
                
                p.data.add_(d_p, alpha=-0.5 * lr)
                p.data.add_(noise)

class pSGLD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-4, alpha=0.99, eps=1e-8, noise_std=1e-4):
        defaults = dict(lr=lr, alpha=alpha, eps=eps, noise_std=noise_std)
        super().__init__(params, defaults)

    def step(self):
        for group in self.param_groups:
            lr = group['lr']
            alpha = group['alpha']
            eps = group['eps']
            noise_std = group['noise_std']
            
            for p in group['params']:
                if p.grad is None: continue
                grad = p.grad.data
                state = self.state[p]
                
                if 'square_avg' not in state:
                    state['square_avg'] = torch.zeros_like(p.data)
                
                square_avg = state['square_avg']
                square_avg.mul_(alpha).addcmul_(grad, grad, value=1 - alpha)
                
                avg = square_avg.sqrt().add_(eps)
                
                # Preconditioner M = 1/avg
                # Update: theta = theta - lr/2 * M * grad + noise * sqrt(M)
                
                # 1. Gradient term: - 0.5 * lr * grad / avg
                p.data.addcdiv_(grad, avg, value=-0.5 * lr)
                
                # 2. Noise term: N(0, lr) * 1/sqrt(avg)
                # noise_std is passed as sqrt(lr) usually
                noise = torch.randn_like(p.data) * noise_std
                p.data.addcdiv_(noise, avg.sqrt())

class RSGLD(torch.optim.Optimizer):
    # Simplified Riemannian SGLD using diagonal Empirical Fisher (similar to pSGLD but different averaging?)
    # For this experiment, we'll use a running average of squared gradients similar to pSGLD but maybe different hyperparameters
    # or just treat it as pSGLD with different name if no specific formula provided.
    # Let's implement it identical to pSGLD for now but allow different config.
    def __init__(self, params, lr=1e-4, alpha=0.99, eps=1e-8, noise_std=1e-4):
        defaults = dict(lr=lr, alpha=alpha, eps=eps, noise_std=noise_std)
        super().__init__(params, defaults)

    def step(self):
        for group in self.param_groups:
            lr = group['lr']
            alpha = group['alpha']
            eps = group['eps']
            noise_std = group['noise_std']
            
            for p in group['params']:
                if p.grad is None: continue
                grad = p.grad.data
                state = self.state[p]
                
                if 'square_avg' not in state:
                    state['square_avg'] = torch.zeros_like(p.data)
                
                square_avg = state['square_avg']
                square_avg.mul_(alpha).addcmul_(grad, grad, value=1 - alpha)
                
                # Fisher approximation G
                G = square_avg.add(eps)
                
                # Update: theta = theta - lr/2 * G^-1 * grad + N(0, lr * G^-1)
                
                p.data.addcdiv_(grad, G, value=-0.5 * lr)
                
                noise = torch.randn_like(p.data) * noise_std
                p.data.addcdiv_(noise, G.sqrt())

class SISGLD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-4, eps=1e-8, noise_std=1e-4):
        defaults = dict(lr=lr, eps=eps, noise_std=noise_std)
        super().__init__(params, defaults)

    def step(self):
        for group in self.param_groups:
            lr = group['lr']
            eps = group['eps']
            noise_std = group['noise_std']
            
            for p in group['params']:
                if p.grad is None: continue
                grad = p.grad.data
                
                # Calculate scalar preconditioner G = ||W||^2 + eps
                norm_sq = p.data.norm(2).pow(2)
                G = norm_sq + eps
                
                # Update: theta = theta - lr/2 * (1/G) * grad + N(0, lr/G)
                
                p.data.add_(grad, alpha=-0.5 * lr / G)
                
                noise = torch.randn_like(p.data) * noise_std
                p.data.add_(noise, alpha=1.0 / G.sqrt())

# =============================================================================
# Experiment Runner
# =============================================================================

def get_data_loader(task, batch_size=1024):
    # Map task name to what get_data expects
    task_map = {
        'x_plus_y': 'x+y',
        'x_minus_y': 'x-y',
        'x_mul_y': 'x*y',
        'x_div_y': 'x_div_y'
    }
    real_task_name = task_map.get(task, task)
    
    # p=97, eq_token=97, op_token=98
    data = get_data(97, 97, 98, real_task_name)
    X = data[:, :-1]
    y = data[:, -1]
    dataset = TensorDataset(X, y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def run_experiment(task, sampler_name, device_id, output_dir):
    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
    print(f"Starting {task} - {sampler_name} on {device}")
    
    # 1. Load Data
    train_loader = get_data_loader(task, batch_size=1024)
    
    # 2. Load Model Checkpoint
    # Path: /data/zjj/test/results/checkpoint_transformer_2_4_128/{task}/wd_1.0/seed42_step100000.pt
    # Fix folder name for x_minus_y -> x-y
    folder_name = task
    if task == 'x_minus_y':
        folder_name = 'x-y'
        
    ckpt_path = f"/data/zjj/test/results/checkpoint_transformer_2_4_128/{folder_name}/wd_1.0/seed42_step100000.pt"
    if not os.path.exists(ckpt_path):
        print(f"Checkpoint not found: {ckpt_path}")
        return

    # 3. Grid Search
    alphas = np.logspace(-1, 1, 30)
    betas = np.logspace(-1, 1, 30)
    
    results = []
    
    # Pre-load model to memory to avoid reloading
    # Checkpoint was trained with p=97, so num_tokens = p + 2 = 99
    base_model = Decoder(num_tokens=99).to(device)
    checkpoint = torch.load(ckpt_path, map_location=device)
    
    # Extract model_state_dict if present
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
        
    # Handle state dict key mismatch if any (e.g. 'module.')
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    base_model.load_state_dict(state_dict)
    base_model.eval()
    
    # Iterate grid
    # To speed up, we can loop alpha/beta
    # Total 900 points.
    
    for alpha in tqdm(alphas, desc=f"{task}-{sampler_name}-Alpha"):
        for beta in betas:
            # Clone model for this run
            model = copy.deepcopy(base_model)
            
            # Apply Scaling
            scale_model(model, alpha, beta)
            
            # Setup Optimizer
            lr = 1e-5 # Small LR for local exploration
            noise_std = math.sqrt(lr)
            
            if sampler_name == 'SGLD':
                optimizer = SGLD(model.parameters(), lr=lr, noise_std=noise_std)
            elif sampler_name == 'pSGLD':
                optimizer = pSGLD(model.parameters(), lr=lr, noise_std=noise_std)
            elif sampler_name == 'RSGLD':
                optimizer = RSGLD(model.parameters(), lr=lr, noise_std=noise_std)
            elif sampler_name == 'SISGLD':
                optimizer = SISGLD(model.parameters(), lr=lr, noise_std=noise_std)
            
            # Run SGLD
            model.train()
            losses = []
            
            # Run for a few steps (e.g. 20 steps) to estimate local energy
            # We use the first batch repeatedly or a few batches
            iterator = iter(train_loader)
            
            for _ in range(20):
                try:
                    inputs, targets = next(iterator)
                except StopIteration:
                    iterator = iter(train_loader)
                    inputs, targets = next(iterator)
                
                inputs, targets = inputs.to(device), targets.to(device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                # Loss: CrossEntropy
                # outputs: [B, T, V], targets: [B]
                # We only care about the last token prediction (the result)
                loss = F.cross_entropy(outputs[:, -1, :], targets)
                loss.backward()
                optimizer.step()
                
                losses.append(loss.item())
            
            avg_loss = np.mean(losses)
            results.append([alpha, beta, avg_loss])
            
            del model, optimizer
            torch.cuda.empty_cache()

    # Save Results
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, f"{task}_{sampler_name}.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['alpha', 'beta', 'llc'])
        writer.writerows(results)
    
    print(f"Saved {csv_path}")

def main():
    tasks = ['x_plus_y', 'x_minus_y', 'x_mul_y', 'x_div_y']
    samplers = ['SGLD', 'pSGLD', 'RSGLD', 'SISGLD']
    
    output_dir = '/data/zjj/test/results/Data/EPR-SGLD'
    
    # Create a list of jobs
    jobs = []
    gpu_count = torch.cuda.device_count()
    
    idx = 0
    for task in tasks:
        for sampler in samplers:
            device_id = idx % gpu_count
            p = multiprocessing.Process(target=run_experiment, args=(task, sampler, device_id, output_dir))
            jobs.append(p)
            idx += 1
    
    # Run jobs in chunks of GPU count to avoid OOM if models are large
    # Or just run all if memory permits (GPT2 medium is ~300M params, 24GB VRAM can hold many)
    # But we have 16 jobs. 8 GPUs. 2 jobs per GPU is fine.
    
    for p in jobs:
        p.start()
        
    for p in jobs:
        p.join()

if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)
    main()
