import os
import sys
import glob
import csv
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import multiprocessing
from tqdm import tqdm
import re

# Add src to path to import model and data utils
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))
try:
    from checkpoint_sp_l2_get import Decoder, get_data
except ImportError:
    # Fallback definition if import fails
    print("Warning: Could not import from checkpoint_sp_l2_get.py, using local definitions.")
    
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
            h = self.token_embeddings(x) + self.position_embeddings(self.pos_ids[:x.size(1)])
            mask = self.causal_mask[:x.size(1), :x.size(1)]
            for layer in self.layers:
                h = layer(h, attn_mask=mask)
            return self.head(self.ln_f(h))

    def get_data(p, eq_token, op_token, task_name):
        x = torch.arange(p)
        y = torch.arange(1, p)
        x, y = torch.cartesian_prod(x, y).T
        eq = torch.ones_like(x) * eq_token
        op = torch.ones_like(x) * op_token

        if task_name == 'x+y':
            result = (x + y) % p
        elif task_name == 'x-y':
            result = (x - y) % p
        elif task_name == 'x_div_y':
            res_list = [(xi.item() * pow(yi.item(), p - 2, p)) % p for xi, yi in zip(x, y)]
            result = torch.tensor(res_list)
        elif task_name == 'x*y':
            result = (x * y) % p
        else:
            raise ValueError(f"Unknown task: {task_name}")

        data = torch.stack([x, op, y, eq, result]).T
        return data

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def calc_input_gradient_agop(model, data_loader, device):
    """
    Calculate Input-Gradient AGOP (Average Gradient Outer Product).
    We compute the gradients of the loss w.r.t. the input embeddings (after token embedding layer).
    AGOP = || 1/N * sum(g_i * g_i^T) ||_F
    """
    model.eval()
    
    # Accumulate the covariance matrix (running sum)
    cov_matrix = None
    total_samples = 0
    
    for batch in data_loader:
        inputs, targets = batch[0][:, :-1].to(device), batch[0][:, -1].to(device)
        
        # 1. Get Embeddings
        # We manually call the embedding part of the model to get the tensor we want gradients for
        x = inputs
        token_emb = model.token_embeddings(x)
        pos_emb = model.position_embeddings(model.pos_ids[:x.size(1)])
        
        # We want gradients w.r.t. this combined embedding
        embeddings = token_emb + pos_emb
        embeddings.retain_grad()
        
        # 2. Continue Forward
        h = embeddings
        mask = model.causal_mask[:x.size(1), :x.size(1)]
        for layer in model.layers:
            h = layer(h, attn_mask=mask)
        logits = model.head(model.ln_f(h))
        
        loss = F.cross_entropy(logits[:, -1, :], targets)
        
        # 3. Backward
        model.zero_grad()
        loss.backward()
        
        # 4. Collect Gradients
        grads = embeddings.grad # (B, Seq, Dim)
        if grads is None:
            continue
            
        B, S, D = grads.shape
        grads_flat = grads.view(B, -1) # (B, S*D)
        
        # Update uncentered covariance: sum(g * g^T)
        # Using batch matrix multiplication: (B, D_flat, 1) @ (B, 1, D_flat) -> (B, D_flat, D_flat)
        # Then sum over B
        # To save memory, we can do:
        batch_cov = torch.matmul(grads_flat.unsqueeze(2), grads_flat.unsqueeze(1)).sum(dim=0)
        
        if cov_matrix is None:
            cov_matrix = batch_cov
        else:
            cov_matrix += batch_cov
            
        total_samples += B
        
    if cov_matrix is None:
        return 0.0
        
    agop_matrix = cov_matrix / total_samples
    return torch.norm(agop_matrix, p='fro').item()

def calc_cycle_bias(model):
    """
    Calculate Cycle Bias (Circulant Score).
    Measures how much the weight matrices are diagonal in the Fourier basis.
    Score = (Energy on Diagonal of F*W*F_inv) / (Total Energy)
    """
    total_score = 0.0
    count = 0
    
    target_layers = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # Check if square
            if module.weight.shape[0] == module.weight.shape[1]:
                target_layers.append(module.weight)
            
    # Also manually check MultiheadAttention parameters
    for layer in model.layers:
        # In PyTorch MultiheadAttention, in_proj_weight is (3*dim, dim)
        if layer.attn.in_proj_weight is not None:
            q, k, v = layer.attn.in_proj_weight.chunk(3, dim=0)
            target_layers.extend([q, k, v])
        if layer.attn.out_proj.weight is not None:
            target_layers.append(layer.attn.out_proj.weight)
            
    if not target_layers:
        return 0.0
        
    for W in target_layers:
        W = W.detach().float()
        if W.shape[0] != W.shape[1]:
            continue
            
        n = W.shape[0]
        # Create DFT matrix F
        F_matrix = torch.fft.fft(torch.eye(n, device=W.device))
        F_inv_matrix = torch.fft.ifft(torch.eye(n, device=W.device))
        
        # W_hat = F @ W.complex() @ F_inv_matrix
        # Fix: Use type casting instead of .complex() which might not exist on Tensor
        W_complex = W.type(torch.complex64)
        W_hat = torch.matmul(torch.matmul(F_matrix, W_complex), F_inv_matrix)
        
        # Energy
        energy = torch.abs(W_hat) ** 2
        total_energy = torch.sum(energy)
        
        # Diagonal energy
        diag_energy = torch.sum(torch.diagonal(energy))
        
        score = diag_energy / (total_energy + 1e-10)
        total_score += score.item()
        count += 1
        
    return total_score / max(count, 1)

def process_checkpoint(args):
    ckpt_path, task_name, wd, seed, device_id, p = args
    
    device = torch.device(f"cuda:{device_id}")
    
    # Load Checkpoint
    try:
        checkpoint = torch.load(ckpt_path, map_location=device)
    except Exception as e:
        print(f"Error loading {ckpt_path}: {e}")
        return None

    step = checkpoint.get('step', 0)
    
    # Setup Model
    num_tokens = p + 2
    seq_len = 5
    model = Decoder(dim=128, num_layers=2, num_heads=4, num_tokens=num_tokens, seq_len=seq_len)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Setup Data
    setup_seed(seed) # Important to get same split
    full_data = get_data(p, p, p + 1, task_name)
    indices = torch.randperm(len(full_data))
    split = int(len(full_data) * 0.5)
    train_data = full_data[indices[:split]]
    test_data = full_data[indices[split:]]
    
    train_loader = DataLoader(TensorDataset(train_data), batch_size=512, shuffle=False)
    test_loader = DataLoader(TensorDataset(test_data), batch_size=512, shuffle=False)
    
    # 1. Calculate Metrics (Loss/Acc)
    def eval_model(loader):
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        with torch.no_grad():
            for batch in loader:
                inputs, targets = batch[0][:, :-1].to(device), batch[0][:, -1].to(device)
                logits = model(inputs)
                loss = F.cross_entropy(logits[:, -1, :], targets, reduction='sum')
                preds = logits[:, -1, :].argmax(dim=-1)
                total_loss += loss.item()
                total_correct += (preds == targets).sum().item()
                total_samples += inputs.size(0)
        return total_loss / total_samples, total_correct / total_samples

    train_loss, train_acc = eval_model(train_loader)
    test_loss, test_acc = eval_model(test_loader)
    
    # 2. Calculate AGOP
    agop_val = calc_input_gradient_agop(model, train_loader, device)
    
    # 3. Calculate Cycle Bias
    cycle_bias_val = calc_cycle_bias(model)
    
    return {
        'step': step,
        'train_loss': train_loss,
        'train_acc': train_acc,
        'test_loss': test_loss,
        'test_acc': test_acc,
        'agop': agop_val,
        'cycle_bias': cycle_bias_val
    }

def worker(gpu_id, task_queue, result_queue):
    torch.cuda.set_device(gpu_id)
    while True:
        task = task_queue.get()
        if task is None:
            break
        
        # Unpack task tuple and inject the correct gpu_id
        # task structure: (ckpt_path, task_name, wd, seed, placeholder_device_id, p)
        ckpt_path, task_name, wd, seed, _, p = task
        
        # Create new args tuple with correct gpu_id
        args = (ckpt_path, task_name, wd, seed, gpu_id, p)
        
        try:
            res = process_checkpoint(args)
            result_queue.put((ckpt_path, res)) # Path as key
        except Exception as e:
            # import traceback
            # traceback.print_exc()
            print(f"Worker Error on {ckpt_path}: {e}")
            result_queue.put((ckpt_path, None))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", type=str, default="results/checkpoint_transformer_2_4_128")
    parser.add_argument("--output_dir", type=str, default="results/Data/AGOP_CycleBias")
    parser.add_argument("--p", type=int, default=97)
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    tasks = ['x_plus_y', 'x-y', 'x_mul_y', 'x_div_y']
    task_map = {
        'x_plus_y': 'x+y',
        'x-y': 'x-y',
        'x_mul_y': 'x*y',
        'x_div_y': 'x_div_y'
    }
    
    all_checkpoints = []
    
    print("Scanning checkpoints...")
    for task_safe in tasks:
        task_path = os.path.join(args.checkpoint_dir, task_safe)
        if not os.path.exists(task_path):
            continue
            
        for wd_dir in os.listdir(task_path):
            wd_path = os.path.join(task_path, wd_dir)
            if not os.path.isdir(wd_path):
                continue
                
            wd_val = float(wd_dir.replace("wd_", ""))
            pt_files = glob.glob(os.path.join(wd_path, "*.pt"))
            
            files_by_seed = {}
            for f in pt_files:
                fname = os.path.basename(f)
                match = re.match(r"seed(\d+)_step(\d+)\.pt", fname)
                if match:
                    seed = int(match.group(1))
                    step = int(match.group(2))
                    if seed not in files_by_seed:
                        files_by_seed[seed] = []
                    files_by_seed[seed].append((step, f))
            
            for seed, files in files_by_seed.items():
                files.sort(key=lambda x: x[0])
                csv_name = f"{task_safe}_wd_{wd_val}_seed_{seed}.csv"
                csv_path = os.path.join(args.output_dir, csv_name)
                
                done_steps = set()
                if os.path.exists(csv_path):
                    with open(csv_path, 'r') as f:
                        reader = csv.reader(f)
                        next(reader, None)
                        for row in reader:
                            if row:
                                done_steps.add(int(row[0]))
                
                for step, fpath in files:
                    if step not in done_steps:
                        all_checkpoints.append({
                            'path': fpath,
                            'task_name': task_map[task_safe],
                            'wd': wd_val,
                            'seed': seed,
                            'p': args.p,
                            'csv_path': csv_path,
                            'device_id': -1 # Placeholder
                        })

    print(f"Found {len(all_checkpoints)} checkpoints to process.")
    
    if not all_checkpoints:
        return

    num_gpus = torch.cuda.device_count()
    ctx = multiprocessing.get_context("spawn")
    task_queue = ctx.Queue()
    result_queue = ctx.Queue()
    
    workers = []
    for i in range(num_gpus):
        p = ctx.Process(target=worker, args=(i, task_queue, result_queue))
        p.start()
        workers.append(p)
        
    for item in all_checkpoints:
        task_queue.put((item['path'], item['task_name'], item['wd'], item['seed'], -1, item['p']))
        
    for _ in range(num_gpus):
        task_queue.put(None)
        
    pbar = tqdm(total=len(all_checkpoints))
    
    finished_count = 0
    while finished_count < len(all_checkpoints):
        path, res = result_queue.get()
        finished_count += 1
        pbar.update(1)
        
        if res is None:
            continue
            
        parent = os.path.dirname(path)
        wd_dir = os.path.basename(parent)
        task_dir = os.path.basename(os.path.dirname(parent))
        fname = os.path.basename(path)
        match = re.match(r"seed(\d+)_step(\d+)\.pt", fname)
        seed = int(match.group(1))
        wd_val = float(wd_dir.replace("wd_", ""))
        
        csv_name = f"{task_dir}_wd_{wd_val}_seed_{seed}.csv"
        csv_path = os.path.join(args.output_dir, csv_name)
        
        file_exists = os.path.exists(csv_path)
        
        with open(csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(['step', 'train_loss', 'train_acc', 'test_loss', 'test_acc', 'agop', 'cycle_bias'])
            
            writer.writerow([
                res['step'],
                f"{res['train_loss']:.6f}",
                f"{res['train_acc']:.6f}",
                f"{res['test_loss']:.6f}",
                f"{res['test_acc']:.6f}",
                f"{res['agop']:.6f}",
                f"{res['cycle_bias']:.6f}"
            ])
            
    pbar.close()
    for p in workers:
        p.join()
        
    print("Sorting CSVs...")
    for csv_file in glob.glob(os.path.join(args.output_dir, "*.csv")):
        rows = []
        with open(csv_file, 'r') as f:
            reader = csv.reader(f)
            try:
                header = next(reader)
                rows = list(reader)
            except StopIteration:
                continue
        
        rows.sort(key=lambda x: int(x[0]))
        
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(rows)

if __name__ == "__main__":
    main()
