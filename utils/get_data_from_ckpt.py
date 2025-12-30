import os
import csv
import glob
import argparse
import re
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import multiprocessing

# =============================================================================
# 1. 模型与组件 (保持一致)
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

    # 还原任务名映射逻辑
    if 'plus' in task_name or '+' in task_name:
        result = (x + y) % p
    elif 'minus' in task_name or '-' in task_name:
        result = (x - y) % p
    elif 'div' in task_name:
        res_list = [(xi.item() * pow(yi.item(), p - 2, p)) % p for xi, yi in zip(x, y)]
        result = torch.tensor(res_list)
    elif 'mul' in task_name or '*' in task_name:
        result = (x * y) % p
    else:
        # 尝试直接解析
        if task_name == 'x+y': result = (x + y) % p
        elif task_name == 'x-y': result = (x - y) % p
        elif task_name == 'x*y': result = (x * y) % p
        else: raise ValueError(f"Unknown task: {task_name}")

    data = torch.stack([x, op, y, eq, result]).T
    return data

# =============================================================================
# 2. 指标计算函数
# =============================================================================

def calc_l2_norm(model):
    total_norm = 0.0
    for p in model.parameters():
        param_norm = p.data.norm(2)
        total_norm += param_norm.item() ** 2
    return total_norm ** 0.5

def calc_spectral_entropy(model):
    total_entropy = 0.0
    num_matrices = 0
    for _, param in model.named_parameters():
        if len(param.shape) >= 2:
            try:
                s = torch.linalg.svdvals(param.data.float())
                s_normalized = s / (s.sum() + 1e-10)
                entropy = -(s_normalized * torch.log(s_normalized + 1e-10)).sum().item()
                total_entropy += entropy
                num_matrices += 1
            except: continue
    return total_entropy / max(num_matrices, 1)

def calc_attention_spectral_entropy(model):
    total_entropy = 0.0
    num_matrices = 0
    for name, param in model.named_parameters():
        if 'attn' in name and len(param.shape) >= 2:
            try:
                s = torch.linalg.svdvals(param.data.float())
                s_normalized = s / (s.sum() + 1e-10)
                entropy = -(s_normalized * torch.log(s_normalized + 1e-10)).sum().item()
                total_entropy += entropy
                num_matrices += 1
            except: continue
    return total_entropy / max(num_matrices, 1)

def calc_embedding_spectral_entropy(model):
    try:
        emb_weight = model.token_embeddings.weight.data.float()
        s = torch.linalg.svdvals(emb_weight)
        s_normalized = s / (s.sum() + 1e-10)
        return -(s_normalized * torch.log(s_normalized + 1e-10)).sum().item()
    except: return 0.0

def evaluate_dataset(model, loader, device):
    """计算整个数据集的 Loss 和 Acc"""
    model.eval()
    total_loss = 0.0
    total_correct = 0.0
    total_samples = 0
    
    with torch.no_grad():
        for (batch,) in loader:
            batch = batch.to(device)
            inp, target = batch[:, :-1], batch[:, -1]
            logits = model(inp)
            loss = F.cross_entropy(logits[:, -1, :].float(), target, reduction='sum')
            
            preds = logits[:, -1, :].argmax(-1)
            correct = (preds == target).float().sum()
            
            total_loss += loss.item()
            total_correct += correct.item()
            total_samples += batch.size(0)
            
    return total_loss / total_samples, total_correct / total_samples

# =============================================================================
# 3. 单个实验回测逻辑
# =============================================================================

def run_recollection_job(config):
    task_safe, wd, seed, pt_files, gpu_id, args = config
    
    device = torch.device(f"cuda:{gpu_id}")
    torch.cuda.set_device(gpu_id)
    
    # --- 1. 复现数据划分 (关键) ---
    torch.manual_seed(seed)
    
    p = 97
    full_data = get_data(p, p, p + 1, task_safe)
    indices = torch.randperm(len(full_data)) # 这里的 indices 应该和训练时一模一样
    split = int(len(full_data) * 0.5)
    
    train_data = full_data[indices[:split]]
    test_data = full_data[indices[split:]]
    
    # 增大 Batch Size 加快评估速度
    eval_bs = 2048 
    train_loader = DataLoader(TensorDataset(train_data), batch_size=eval_bs, shuffle=False)
    test_loader = DataLoader(TensorDataset(test_data), batch_size=eval_bs, shuffle=False)
    
    # --- 2. 初始化模型 ---
    num_tokens = p + 2
    model = Decoder(dim=128, num_layers=2, num_heads=4, num_tokens=num_tokens, seq_len=5).to(device)
    
    # 按步数排序文件
    def get_step(path):
        match = re.search(r'step(\d+)', path)
        return int(match.group(1)) if match else 0
        
    pt_files.sort(key=get_step)
    
    # --- [修改点] 设置新的输出路径 ---
    output_dir = "/data/zjj/test/results/Data/4_modular"
    os.makedirs(output_dir, exist_ok=True)
    
    csv_filename = f"{task_safe}_wd_{wd}_seed_{seed}.csv"
    csv_path = os.path.join(output_dir, csv_filename)
    
    # 如果文件已存在且很大，可能想跳过？这里为了安全默认覆盖或追加。
    # 为简单起见，这里直接覆盖写入
    
    results = []
    print(f"[START] GPU:{gpu_id} Recalling {task_safe} WD={wd} S={seed} ({len(pt_files)} checkpoints)")
    
    for pt_file in pt_files:
        step = get_step(pt_file)
        
        # 加载权重
        try:
            checkpoint = torch.load(pt_file, map_location=device)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
        except Exception as e:
            print(f"Error loading {pt_file}: {e}")
            continue
            
        # 计算指标
        train_loss, train_acc = evaluate_dataset(model, train_loader, device)
        test_loss, test_acc = evaluate_dataset(model, test_loader, device)
        
        l2_val = calc_l2_norm(model)
        spec_ent = calc_spectral_entropy(model)
        attn_ent = calc_attention_spectral_entropy(model)
        emb_ent = calc_embedding_spectral_entropy(model)
        
        results.append([step, train_loss, train_acc, test_loss, test_acc, 
                        l2_val, spec_ent, attn_ent, emb_ent])
        
    # 写入 CSV
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['steps', 'train_loss', 'train_acc', 'test_loss', 'test_acc', 
                         'l2_norm', 'spectral_entropy', 'attention_spectral_entropy', 'embedding_spectral_entropy'])
        writer.writerows(results)
        
    print(f"[DONE] Saved to {csv_path}")
    return True

# =============================================================================
# 4. 多进程调度逻辑
# =============================================================================

def worker_func(gpu_id, queue, args):
    while True:
        task = queue.get()
        if task is None:
            break
        try:
            run_recollection_job((*task, gpu_id, args))
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Job failed: {e}")

def main():
    parser = argparse.ArgumentParser()
    # 默认权重输入路径，请根据实际情况修改
    parser.add_argument("--checkpoint_root", type=str, default="results/checkpoint_transformer_2_4_128", 
                        help="Root directory containing task subfolders")
    parser.add_argument("--jobs_per_gpu", type=int, default=4, help="Concurrent jobs per GPU")
    args = parser.parse_args()
    
    # --- 1. 扫描所有 Checkpoint ---
    print(f"Scanning checkpoints in {args.checkpoint_root}...")
    
    # 存储结构: experiments[(task, wd, seed)] = [file_path1, file_path2, ...]
    experiments = {}
    
    # 递归查找所有 .pt 文件
    all_pts = glob.glob(os.path.join(args.checkpoint_root, "**", "*.pt"), recursive=True)
    
    for pt in all_pts:
        try:
            filename = os.path.basename(pt)
            
            # 解析 Seed
            seed_match = re.search(r'seed(\d+)', filename)
            if not seed_match: continue
            seed = int(seed_match.group(1))
            
            # 解析 WD 和 Task
            wd = None
            task_name = None
            
            curr = os.path.dirname(pt)
            # 向上寻找包含 'wd_' 的目录
            while len(curr) > len(os.path.abspath(args.checkpoint_root)) or (not os.path.isabs(args.checkpoint_root) and len(curr) > len(args.checkpoint_root)):
                dirname = os.path.basename(curr)
                if dirname.startswith("wd_"):
                    wd = float(dirname.replace("wd_", ""))
                    task_name = os.path.basename(os.path.dirname(curr))
                    break
                curr = os.path.dirname(curr)
                # 防止死循环，到达根目录停止
                if curr == os.path.dirname(curr): 
                    break
            
            if wd is None or task_name is None:
                # 尝试另一种常见的目录结构解析
                continue
                
            key = (task_name, wd, seed)
            if key not in experiments:
                experiments[key] = []
            experiments[key].append(pt)
            
        except Exception as e:
            print(f"Skipping {pt}: {e}")
            continue

    print(f"Found {len(experiments)} unique experiments.")
    
    # --- 2. 多进程处理 ---
    ctx = multiprocessing.get_context("spawn")
    queue = ctx.Queue()
    
    # 填充任务队列
    for key, files in experiments.items():
        task_name, wd, seed = key
        queue.put((task_name, wd, seed, files))
        
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        print("Error: No GPUs found!")
        return

    total_workers = num_gpus * args.jobs_per_gpu
    
    for _ in range(total_workers):
        queue.put(None)
        
    procs = []
    for i in range(total_workers):
        gpu_id = i % num_gpus
        p = ctx.Process(target=worker_func, args=(gpu_id, queue, args))
        p.start()
        procs.append(p)
        
    for p in procs:
        p.join()
        
    print(f"All recollection jobs finished. Data saved to /data/zjj/test/results/Data/4_modular")

if __name__ == "__main__":
    main()