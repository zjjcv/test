import os
import glob
import csv
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pandas as pd
from tqdm import tqdm
import sys
import copy
import multiprocessing

# Add src to path to import model and data utils
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))
try:
    from checkpoint_sp_l2_get import Decoder, get_data
except ImportError:
    # Fallback if running from a different directory structure
    sys.path.append(os.path.join(os.path.dirname(__file__), '../../src'))
    from checkpoint_sp_l2_get import Decoder, get_data

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SGLD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, num_data=1, preconditioner=None, alpha=0.9, eps=1e-8):
        defaults = dict(lr=lr, num_data=num_data, alpha=alpha, eps=eps)
        super(SGLD, self).__init__(params, defaults)
        self.preconditioner = preconditioner
        self.state = dict()
        for group in self.param_groups:
            for p in group['params']:
                self.state[p] = dict(momentum=torch.zeros_like(p.data))

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            num_data = group['num_data']
            alpha = group['alpha']
            eps = group['eps']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                
                # Add noise
                noise = torch.randn_like(p.data)
                
                if self.preconditioner == 'euclidean':
                    p.data.add_(d_p, alpha=-0.5 * lr)
                    p.data.add_(noise, alpha=np.sqrt(lr))
                    
                elif self.preconditioner == 'psgld':
                    state = self.state[p]
                    momentum = state['momentum']
                    momentum.mul_(alpha).addcmul_(d_p, d_p, value=1 - alpha)
                    G_sqrt = momentum.add(eps).sqrt()
                    p.data.addcdiv_(d_p, G_sqrt, value=-0.5 * lr)
                    p.data.addcdiv_(noise, G_sqrt.sqrt(), value=np.sqrt(lr))
                    
                elif self.preconditioner == 'riemannian':
                    state = self.state[p]
                    momentum = state['momentum']
                    momentum.mul_(alpha).addcmul_(d_p, d_p, value=1 - alpha)
                    G = momentum.add(eps)
                    G_inv = 1.0 / G
                    p.data.addcmul_(d_p, G_inv, value=-0.5 * lr)
                    p.data.addcmul_(noise, G.sqrt().reciprocal(), value=np.sqrt(lr))

        return loss

def estimate_llc(model, data_loader, criterion, sampler_type, num_steps=40, burn_in=20, lr=1e-5, device='cuda'):
    """
    Estimate LLC using SGLD sampling.
    LLC = mean(n * (L(theta) - L(theta*)))
    """
    # 1. Calculate L(theta*) - Loss at the mode (starting point)
    model.eval()
    total_loss_star = 0.0
    total_samples = 0
    
    with torch.no_grad():
        for batch in data_loader:
            inputs, targets = batch[0][:, :-1].to(device), batch[0][:, -1].to(device)
            logits = model(inputs)
            loss = criterion(logits[:, -1, :], targets)
            total_loss_star += loss.item() * inputs.size(0)
            total_samples += inputs.size(0)
    
    avg_loss_star = total_loss_star / total_samples
    n = total_samples
    
    # 2. Run SGLD chain
    model.train()
    if sampler_type == 'euclidean':
        optimizer = SGLD(model.parameters(), lr=lr, preconditioner='euclidean')
    elif sampler_type == 'psgld':
        optimizer = SGLD(model.parameters(), lr=lr, preconditioner='psgld', alpha=0.99)
    elif sampler_type == 'riemannian':
        optimizer = SGLD(model.parameters(), lr=lr, preconditioner='riemannian', alpha=0.99)
    else:
        raise ValueError(f"Unknown sampler: {sampler_type}")
        
    llc_samples = []
    iterator = iter(data_loader)
    
    for step in range(num_steps):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(data_loader)
            batch = next(iterator)
            
        inputs, targets = batch[0][:, :-1].to(device), batch[0][:, -1].to(device)
        
        optimizer.zero_grad()
        logits = model(inputs)
        loss = criterion(logits[:, -1, :], targets)
        (loss * n).backward()
        optimizer.step()
        
        if step >= burn_in:
            with torch.no_grad():
                current_loss = loss.item()
                llc_val = n * (current_loss - avg_loss_star)
                llc_samples.append(llc_val)
                
    return np.mean(llc_samples)

def get_task_name_from_dir(dir_name):
    if dir_name == 'x_plus_y': return 'x+y'
    if dir_name == 'x_minus_y': return 'x-y'
    if dir_name == 'x-y': return 'x-y'
    if dir_name == 'x_div_y': return 'x_div_y'
    if dir_name == 'x_mul_y': return 'x*y'
    return dir_name

def worker_process(gpu_id, task_configs, args):
    # task_configs is a list of dicts: {'task_name':, 'wd':, 'seed':, 'checkpoints': [paths]}
    device = torch.device(f"cuda:{gpu_id}")
    print(f"Worker on GPU {gpu_id} started. Processing {len(task_configs)} experiments.")
    
    criterion = nn.CrossEntropyLoss()
    
    for config in task_configs:
        task_name = config['task_name']
        wd = config['wd']
        seed = config['seed']
        ckpts = config['checkpoints']
        
        print(f"GPU {gpu_id}: Processing {task_name} WD={wd} Seed={seed} ({len(ckpts)} ckpts)")
        
        # Load Data
        full_data = get_data(args.p, args.p, args.p + 1, task_name)
        
        # Recreate Data Split
        torch.manual_seed(seed)
        indices = torch.randperm(len(full_data))
        split = int(len(full_data) * 0.5)
        train_data = full_data[indices[:split]]
        test_data = full_data[indices[split:]]
        
        train_loader = DataLoader(TensorDataset(train_data), batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(TensorDataset(test_data), batch_size=len(test_data), shuffle=False)
        
        # Initialize Model
        num_tokens = args.p + 2
        seq_len = 5
        model = Decoder(dim=128, num_layers=2, num_heads=4, num_tokens=num_tokens, seq_len=seq_len).to(device)
        
        # Output CSV
        task_safe = task_name.replace("/", "_div_").replace("*", "_mul_").replace("+", "_plus_")
        csv_filename = f"{task_safe}_wd{wd}_seed{seed}_llc.csv"
        csv_path = os.path.join(args.output_dir, csv_filename)
        
        if not os.path.exists(csv_path):
            with open(csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['seed', 'step', 'train_loss', 'train_acc', 'test_loss', 'test_acc', 'llc_euclidean', 'llc_psgld', 'llc_riemannian'])
        
        # Sort checkpoints by step
        ckpts.sort(key=lambda x: x['step'])
        
        for ckpt_info in tqdm(ckpts, desc=f"GPU{gpu_id}-{task_safe}-S{seed}", position=gpu_id):
            ckpt_path = ckpt_info['path']
            step = ckpt_info['step']
            
            try:
                checkpoint = torch.load(ckpt_path, map_location=device)
                model.load_state_dict(checkpoint['model_state_dict'])
            except Exception as e:
                print(f"Error loading {ckpt_path}: {e}")
                continue
            
            # Metrics
            model.eval()
            with torch.no_grad():
                # Train
                train_loss = 0.0; train_correct = 0; train_total = 0
                for batch in train_loader:
                    inputs, targets = batch[0][:, :-1].to(device), batch[0][:, -1].to(device)
                    logits = model(inputs)
                    loss = criterion(logits[:, -1, :], targets)
                    train_loss += loss.item() * inputs.size(0)
                    preds = logits[:, -1, :].argmax(dim=-1)
                    train_correct += (preds == targets).sum().item()
                    train_total += inputs.size(0)
                train_loss /= max(train_total, 1)
                train_acc = train_correct / max(train_total, 1)
                
                # Test
                test_loss = 0.0; test_correct = 0; test_total = 0
                for batch in test_loader:
                    inputs, targets = batch[0][:, :-1].to(device), batch[0][:, -1].to(device)
                    logits = model(inputs)
                    loss = criterion(logits[:, -1, :], targets)
                    test_loss += loss.item() * inputs.size(0)
                    preds = logits[:, -1, :].argmax(dim=-1)
                    test_correct += (preds == targets).sum().item()
                    test_total += inputs.size(0)
                test_loss /= max(test_total, 1)
                test_acc = test_correct / max(test_total, 1)
            
            # LLC
            original_state = copy.deepcopy(model.state_dict())
            results = {
                'seed': seed, 'step': step,
                'train_loss': train_loss, 'train_acc': train_acc,
                'test_loss': test_loss, 'test_acc': test_acc
            }
            
            for sampler in ['euclidean', 'psgld', 'riemannian']:
                model.load_state_dict(original_state)
                llc = estimate_llc(model, train_loader, criterion, sampler, 
                                 num_steps=args.num_steps, burn_in=args.burn_in, lr=args.lr, device=device)
                results[f'llc_{sampler}'] = llc
            
            with open(csv_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    results['seed'], results['step'], 
                    results['train_loss'], results['train_acc'],
                    results['test_loss'], results['test_acc'],
                    results['llc_euclidean'], results['llc_psgld'], results['llc_riemannian']
                ])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_root", type=str, default="/data/zjj/test/results/checkpoint_transformer_2_4_128")
    parser.add_argument("--output_dir", type=str, default="/data/zjj/test/results/Data/LLC")
    parser.add_argument("--p", type=int, default=97)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-6, help="Learning rate for SGLD")
    parser.add_argument("--num_steps", type=int, default=40, help="Total SGLD steps per checkpoint")
    parser.add_argument("--burn_in", type=int, default=20, help="Burn-in steps")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Scan for all checkpoints
    print(f"Scanning {args.checkpoint_root}...")
    # Structure: root/task_dir/wd_dir/seed_step.pt
    
    experiments = {} # Key: (task_name, wd, seed), Value: list of ckpt paths
    
    # Walk through directories
    for task_dir in os.listdir(args.checkpoint_root):
        task_path = os.path.join(args.checkpoint_root, task_dir)
        if not os.path.isdir(task_path): continue
        
        task_name = get_task_name_from_dir(task_dir)
        
        for wd_dir in os.listdir(task_path):
            wd_path = os.path.join(task_path, wd_dir)
            if not os.path.isdir(wd_path): continue
            
            # Parse WD
            try:
                wd = float(wd_dir.replace("wd_", ""))
            except:
                continue
                
            # Find pt files
            for f in glob.glob(os.path.join(wd_path, "*.pt")):
                name = os.path.basename(f)
                try:
                    parts = name.replace(".pt", "").split("_")
                    seed_part = [p for p in parts if p.startswith("seed")][0]
                    step_part = [p for p in parts if p.startswith("step")][0]
                    seed = int(seed_part.replace("seed", ""))
                    step = int(step_part.replace("step", ""))
                    
                    key = (task_name, wd, seed)
                    if key not in experiments:
                        experiments[key] = []
                    experiments[key].append({'path': f, 'step': step})
                except:
                    continue

    print(f"Found {len(experiments)} unique experiments (Task/WD/Seed combinations).")
    
    # 2. Prepare tasks for workers
    # We want to run 16 processes.
    # If we have exactly 16 experiments, each process gets 1.
    
    experiment_keys = list(experiments.keys())
    experiment_keys.sort()
    
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0: num_gpus = 1
    
    # We want 16 processes total, distributed over 8 GPUs (2 per GPU)
    num_processes = 16
    
    process_tasks = [[] for _ in range(num_processes)]
    for i, key in enumerate(experiment_keys):
        task_name, wd, seed = key
        ckpts = experiments[key]
        process_id = i % num_processes
        process_tasks[process_id].append({
            'task_name': task_name,
            'wd': wd,
            'seed': seed,
            'checkpoints': ckpts
        })
        
    # 3. Launch Processes
    ctx = multiprocessing.get_context('spawn')
    processes = []
    
    print(f"Launching {num_processes} processes across {num_gpus} GPUs...")
    
    for i in range(num_processes):
        if not process_tasks[i]: continue
        
        gpu_id = i % num_gpus
        p = ctx.Process(target=worker_process, args=(gpu_id, process_tasks[i], args))
        p.start()
        processes.append(p)
        
    for p in processes:
        p.join()
        
    print("All done.")

if __name__ == "__main__":
    main()
