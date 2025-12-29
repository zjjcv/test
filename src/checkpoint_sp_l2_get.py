import math
import os
import csv
import numpy as np
import argparse
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

import multiprocessing


# =============================================================================
# 0. 速度/稳定性开关（每个进程都会调用）
# =============================================================================

def setup_torch_runtime(args):
    # TF32（对 Ada 4090 非常关键）
    if args.tf32:
        try:
            torch.set_float32_matmul_precision(args.matmul_precision)  # 'high'/'medium'
        except Exception:
            pass
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    else:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    # Flash/Memory-efficient SDP（不依赖 flash-attn，但你设备暂时不想用就默认关闭）
    if args.enable_flash_sdp:
        try:
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
            torch.backends.cuda.enable_math_sdp(False)
        except Exception:
            pass
    else:
        # 更稳：明确关掉 flash/mem-efficient，走 math SDP（兼容性最好）
        try:
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(False)
            torch.backends.cuda.enable_math_sdp(True)
        except Exception:
            pass

    # CPU 线程数（每个训练进程的 PyTorch intra-op 线程；DataLoader workers 是另一个维度）
    if args.cpu_threads is not None and args.cpu_threads > 0:
        torch.set_num_threads(args.cpu_threads)


# =============================================================================
# 1. 模型架构
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


# =============================================================================
# 2. 工具函数
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
                with torch.no_grad():
                    s = torch.linalg.svdvals(param.data.float())
                    s_normalized = s / (s.sum() + 1e-10)
                    entropy = -(s_normalized * torch.log(s_normalized + 1e-10)).sum().item()
                    total_entropy += entropy
                    num_matrices += 1
            except Exception:
                continue
    return total_entropy / max(num_matrices, 1)


def calc_attention_spectral_entropy(model):
    total_entropy = 0.0
    num_matrices = 0
    for name, param in model.named_parameters():
        if 'attn' in name and len(param.shape) >= 2:
            try:
                with torch.no_grad():
                    s = torch.linalg.svdvals(param.data.float())
                    s_normalized = s / (s.sum() + 1e-10)
                    entropy = -(s_normalized * torch.log(s_normalized + 1e-10)).sum().item()
                    total_entropy += entropy
                    num_matrices += 1
            except Exception:
                continue
    return total_entropy / max(num_matrices, 1)


def calc_embedding_spectral_entropy(model):
    try:
        with torch.no_grad():
            emb_weight = model.token_embeddings.weight.data.float()
            s = torch.linalg.svdvals(emb_weight)
            s_normalized = s / (s.sum() + 1e-10)
            entropy = -(s_normalized * torch.log(s_normalized + 1e-10)).sum().item()
            return entropy
    except Exception:
        return 0.0


# =============================================================================
# 3. 数据生成
# =============================================================================

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


# =============================================================================
# 4. 原子训练任务 (单个实验)
# =============================================================================

def _make_dataloader(dataset, args, shuffle):
    num_workers = int(args.dataloader_workers)
    pin_memory = bool(args.pin_memory)

    kwargs = dict(
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    if num_workers > 0:
        kwargs.update(
            prefetch_factor=int(args.prefetch_factor),
            persistent_workers=bool(args.persistent_workers),
        )
    return DataLoader(dataset, **kwargs)


def run_atomic_experiment(config):
    task_name, wd, seed, device_id, args, output_dir = config
    device = torch.device(f"cuda:{device_id}")
    torch.cuda.set_device(device_id)

    setup_torch_runtime(args)
    torch.cuda.empty_cache()

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    full_data = get_data(args.p, args.p, args.p + 1, task_name)
    indices = torch.randperm(len(full_data))
    split = int(len(full_data) * 0.5)
    train_data = full_data[indices[:split]]
    test_data = full_data[indices[split:]]

    train_loader = _make_dataloader(TensorDataset(train_data), args, shuffle=True)
    test_loader = DataLoader(TensorDataset(test_data), batch_size=len(test_data), shuffle=False, num_workers=0)

    num_tokens = args.p + 2
    seq_len = 5

    def model_ctor():
        return Decoder(dim=128, num_layers=2, num_heads=4, num_tokens=num_tokens, seq_len=seq_len)

    model = model_ctor().to(device)

    if args.use_compile:
        try:
            model = torch.compile(model, mode=args.compile_mode)
        except Exception as e:
            print(f"[WARN] torch.compile failed on GPU:{device_id} ({e}), fallback to eager.")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=wd, betas=(0.9, 0.98))

    if args.precision == "bf16":
        amp_dtype = torch.bfloat16
        use_amp = True
        use_scaler = False
    elif args.precision == "fp16":
        amp_dtype = torch.float16
        use_amp = True
        use_scaler = True
    else:
        amp_dtype = None
        use_amp = False
        use_scaler = False

    scaler = torch.amp.GradScaler('cuda', enabled=use_scaler)

    task_safe = task_name.replace("/", "_div_").replace("*", "_mul_").replace("+", "_plus_")
    
    # 修改保存路径为results/checkpoint_transformer_2_4_128
    checkpoint_dir = os.path.join("results", "checkpoint_transformer_2_4_128", task_safe, f"wd_{wd}")
    data_dir = os.path.join("results", "data")
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)

    # 每100步保存一次模型权重
    checkpoint_steps = set(range(100, args.budget + 1, 100))

    # 创建CSV文件用于保存所有指标
    csv_path = os.path.join(data_dir, f"{task_safe}_wd_{wd}_seed_{seed}.csv")
    
    # 初始化CSV文件，写入表头
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['steps', 'train_loss', 'train_acc', 'test_loss', 'test_acc', 
                       'l2_norm', 'spectral_entropy', 'attention_spectral_entropy', 'embedding_spectral_entropy'])

    steps = 0
    print(f"[START] GPU:{device_id} {task_name} WD={wd} S={seed}")

    while steps < args.budget:
        for (batch_x,) in train_loader:
            model.train()
            batch_x = batch_x.to(device, non_blocking=True)
            inp, target = batch_x[:, :-1], batch_x[:, -1]

            optimizer.zero_grad(set_to_none=True)

            if use_amp:
                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    logits = model(inp)
                    loss = F.cross_entropy(logits[:, -1, :].float(), target)
            else:
                logits = model(inp)
                loss = F.cross_entropy(logits[:, -1, :].float(), target)

            if use_scaler:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            steps += 1

            if steps % 100 == 0:
                # 计算训练损失和准确率
                train_loss_val = float(loss.item())
                train_acc_val = (logits[:, -1, :].argmax(-1) == target).float().mean().item()
                
                # 计算测试损失和准确率
                model.eval()
                with torch.no_grad():
                    for (test_batch,) in test_loader:
                        test_batch = test_batch.to(device, non_blocking=True)
                        t_inp, t_target = test_batch[:, :-1], test_batch[:, -1]
                        if use_amp:
                            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                                t_logits = model(t_inp)
                                test_loss_val = F.cross_entropy(t_logits[:, -1, :].float(), t_target).item()
                        else:
                            t_logits = model(t_inp)
                            test_loss_val = F.cross_entropy(t_logits[:, -1, :].float(), t_target).item()
                        test_acc_val = (t_logits[:, -1, :].argmax(-1) == t_target).float().mean().item()
                
                # 计算谱熵和L2范数
                l2_val = calc_l2_norm(model)
                spectral_entropy_val = calc_spectral_entropy(model)
                attention_entropy_val = calc_attention_spectral_entropy(model)
                embedding_entropy_val = calc_embedding_spectral_entropy(model)

                # 保存所有指标到CSV文件
                with open(csv_path, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([steps, train_loss_val, train_acc_val, test_loss_val, test_acc_val,
                                   l2_val, spectral_entropy_val, attention_entropy_val, embedding_entropy_val])

                # 每100步保存一次模型权重
                checkpoint_path = os.path.join(checkpoint_dir, f"seed{seed}_step{steps}.pt")
                torch.save({
                    'step': steps,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'task': task_name,
                    'weight_decay': wd,
                    'seed': seed
                }, checkpoint_path)

            if steps >= args.budget:
                break

    print(f"✓ [DONE] GPU:{device_id} {task_name} WD={wd} S={seed}")

    del model, optimizer, scaler
    torch.cuda.empty_cache()
    return (task_name, wd, seed)


# =============================================================================
# 5. GPU worker
# =============================================================================

def gpu_task_worker(worker_id, gpu_id, task_queue, result_queue, args):
    """
    每个 worker 固定绑定到 gpu_id；从全局 task_queue 抢任务直到遇到 None。
    """
    try:
        torch.cuda.set_device(gpu_id)
        setup_torch_runtime(args)

        while True:
            item = task_queue.get()
            if item is None:
                break

            task_name, wd, seed, out_dir = item
            try:
                res = run_atomic_experiment((task_name, wd, seed, gpu_id, args, out_dir))
                result_queue.put(("ok", res))
            except Exception as e:
                import traceback
                result_queue.put(("err", (task_name, wd, seed, str(e), traceback.format_exc(), gpu_id, worker_id)))

    except Exception as e:
        import traceback
        result_queue.put(("fatal", (gpu_id, worker_id, str(e), traceback.format_exc())))


# =============================================================================
# 6. 主程序：8 卡 + 每卡并发多个实验进程（一次性跑完 8 个实验）
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--p", type=int, default=97)
    parser.add_argument("--budget", type=int, default=100000)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)

    # 速度相关
    parser.add_argument("--precision", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--tf32", type=int, default=1)
    parser.add_argument("--matmul_precision", type=str, default="high", choices=["high", "medium"])
    parser.add_argument("--enable_flash_sdp", type=int, default=0)  # 你机器暂时不装 flash-attn：默认关

    parser.add_argument("--use_compile", type=int, default=0)
    parser.add_argument("--compile_mode", type=str, default="max-autotune",
                        choices=["default", "reduce-overhead", "max-autotune"])

    # DataLoader（每个训练进程内部的多线程）
    parser.add_argument("--dataloader_workers", type=int, default=8)
    parser.add_argument("--pin_memory", type=int, default=1)
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument("--persistent_workers", type=int, default=1)

    # CPU intra-op threads
    parser.add_argument("--cpu_threads", type=int, default=1)

    # 关键：每张 GPU 同时跑多少个实验（多进程并发）
    parser.add_argument("--jobs_per_gpu", type=int, default=1,
                        help="Concurrent experiment processes per GPU (increase to 2/3 if memory allows)")

    args = parser.parse_args()

    # 任务定义：修改为四个任务（加减乘除），每个任务两个种子
    seeds = [42, 101]
    weight_decays = [0.0, 1.0]
    tasks = [
        'x+y', 'x-y', 'x*y', 'x_div_y'
    ]

    num_gpus = torch.cuda.device_count()
    assert num_gpus >= 8, f"Need at least 8 GPUs, but found {num_gpus}."

    base_dir = os.path.abspath(f"grokking_checkpoint_{datetime.now().strftime('%m%d_%H%M')}")
    os.makedirs(base_dir, exist_ok=True)

    # 生成全部 8 个实验任务（4个任务 × 2个权重衰减）
    all_tasks = []
    for task in tasks:
        task_safe = task.replace("/", "_div_").replace("*", "_mul_").replace("+", "_plus_")
        for wd in weight_decays:
            out_dir = os.path.join(base_dir, task_safe, f"wd_{wd}")
            os.makedirs(out_dir, exist_ok=True)
            for seed in seeds:
                all_tasks.append((task, wd, seed, out_dir))

    total_jobs = len(all_tasks)

    print(f"Total experiments: {total_jobs}")
    print(f"GPUs: {num_gpus}")
    print(f"jobs_per_gpu: {args.jobs_per_gpu}  -> total GPU workers: {num_gpus * args.jobs_per_gpu}")
    print(f"Precision: {args.precision} | TF32: {bool(args.tf32)} | Flash-SDP: {bool(args.enable_flash_sdp)} | compile: {bool(args.use_compile)}")
    print(f"DataLoader workers per TRAIN process: {args.dataloader_workers}")
    print(f"Checkpoint save path: results/checkpoint_transformer_2_4_128/")
    print(f"Metrics data save path: results/data/")

    # spawn，避免 CUDA+fork 坑
    ctx = multiprocessing.get_context("spawn")
    task_queue = ctx.Queue()
    result_queue = ctx.Queue()

    for t in all_tasks:
        task_queue.put(t)

    total_workers = num_gpus * int(args.jobs_per_gpu)

    # 结束信号：每个 worker 一个 None
    for _ in range(total_workers):
        task_queue.put(None)

    # 启动 worker：worker_id 映射到 gpu_id（轮转）
    procs = []
    for worker_id in range(total_workers):
        gpu_id = worker_id % num_gpus
        p = ctx.Process(
            target=gpu_task_worker,
            args=(worker_id, gpu_id, task_queue, result_queue, args),
            daemon=False
        )
        p.start()
        procs.append(p)

    finished = 0

    pbar = tqdm(total=total_jobs, desc="Overall Progress", ncols=100)

    while finished < total_jobs:
        status, payload = result_queue.get()

        if status == "ok":
            finished += 1
            pbar.update(1)

        elif status == "err":
            task_name, wd, seed, err_str, tb, gpu_id, worker_id = payload
            finished += 1
            pbar.update(1)
            print(f"\n!!! Error | GPU:{gpu_id} worker:{worker_id} | {task_name} WD={wd} S={seed}: {err_str}\n{tb}")

        elif status == "fatal":
            gpu_id, worker_id, err_str, tb = payload
            print(f"\n!!! FATAL | GPU:{gpu_id} worker:{worker_id}: {err_str}\n{tb}")
            break

    pbar.close()

    # 等待所有 worker 退出
    for p in procs:
        p.join()

    print(f"\nAll Done! Checkpoints saved in: results/checkpoint_transformer_2_4_128/")
    print(f"Metrics data saved in: results/data/")


if __name__ == "__main__":
    main()