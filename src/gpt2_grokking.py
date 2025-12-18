import math
import os
import csv
import copy
import numpy as np
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Optimizer
from tqdm import tqdm

from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing


# =============================================================================
# 0. Runtime 设置（每个进程都会调用）
# =============================================================================

def setup_torch_runtime(args):
    # TF32（Ada 4090 有效）
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

    # cuDNN benchmark（对固定 shape 有利）
    torch.backends.cudnn.benchmark = bool(args.cudnn_benchmark)


# =============================================================================
# 1. 基础组件
# =============================================================================

class SGLD(Optimizer):
    def __init__(self, params, lr=1e-4, noise_scale=1.0):
        defaults = dict(lr=lr, noise_scale=noise_scale)
        super(SGLD, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            noise_scale = group['noise_scale']
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                p.data.add_(d_p, alpha=-0.5 * lr)

                noise_std = torch.sqrt(torch.tensor(lr, device=p.data.device)) * noise_scale
                noise = torch.randn_like(p.data) * noise_std
                p.data.add_(noise)

        return loss


class LLCEstimator:
    """
    修正点：
    - 不再 deepcopy(可能与 torch.compile 冲突、且非常慢)
    - 改为 model_ctor() 新建同构模型 + load_state_dict
    """
    def __init__(self, model_ctor, criterion, dataloader, device):
        self.model_ctor = model_ctor
        self.criterion = criterion
        self.dataloader = dataloader
        self.device = device

    def estimate(self, current_weights_dict, num_draws=40, lr=1e-4, epsilon=1.0):
        sampling_model = self.model_ctor().to(self.device)
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

            batch_x = batch[0].to(self.device, non_blocking=True)
            inp, target = batch_x[:, :-1], batch_x[:, -1]

            sgld_optim.zero_grad(set_to_none=True)
            logits = sampling_model(inp)
            loss = self.criterion(logits[:, -1, :], target)
            loss.backward()
            sgld_optim.step()
            loss_trace.append(loss.item())

        avg_sampling_loss = float(np.mean(loss_trace))
        llc_proxy = avg_sampling_loss - base_loss

        del sampling_model
        torch.cuda.empty_cache()
        return llc_proxy

    def _compute_full_loss(self, model):
        model.eval()
        total_loss = 0.0
        total_count = 0
        with torch.no_grad():
            for batch in self.dataloader:
                batch_x = batch[0].to(self.device, non_blocking=True)
                inp, target = batch_x[:, :-1], batch_x[:, -1]
                logits = model(inp)
                loss = self.criterion(logits[:, -1, :], target)
                total_loss += loss.item() * batch_x.size(0)
                total_count += batch_x.size(0)
        return total_loss / max(total_count, 1)


# =============================================================================
# 2. GPT-2 Medium 架构（保持你原来的实现，且 dropout 强制为 0）
# =============================================================================

class CausalSelfAttention(nn.Module):
    def __init__(self, dim, num_heads, dropout=0.0):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads

        self.c_attn = nn.Linear(dim, 3 * dim)
        self.c_proj = nn.Linear(dim, dim)

        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.dim, dim=2)

        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

        if mask is not None:
            # mask: [T, T] with -inf above diagonal
            att = att + mask[:T, :T]

        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
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
    # 注意：dropout 默认值改为 0.0（避免任何“默认污染”）
    def __init__(self, dim=1024, num_layers=24, num_heads=16, num_tokens=99, seq_len=5, dropout=0.0, use_checkpoint=True):
        super().__init__()
        self.token_embeddings = nn.Embedding(num_tokens, dim)
        self.position_embeddings = nn.Embedding(seq_len, dim)
        self.drop = nn.Dropout(dropout)
        self.use_checkpoint = use_checkpoint

        self.blocks = nn.ModuleList([GPT2Block(dim, num_heads, dropout) for _ in range(num_layers)])
        self.ln_f = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_tokens, bias=False)

        mask = torch.triu(torch.full((seq_len, seq_len), float('-inf')), diagonal=1)
        self.register_buffer("causal_mask", mask)
        self.register_buffer("pos_ids", torch.arange(seq_len))

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
        tok_emb = self.token_embeddings(x)
        pos_emb = self.position_embeddings(self.pos_ids[:T])
        h = self.drop(tok_emb + pos_emb)

        mask = self.causal_mask[:T, :T]

        if self.use_checkpoint and self.training:
            for block in self.blocks:
                h = torch.utils.checkpoint.checkpoint(block, h, mask, use_reentrant=False)
        else:
            for block in self.blocks:
                h = block(h, mask)

        h = self.ln_f(h)
        logits = self.head(h)
        return logits


# =============================================================================
# 3. 工具函数
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


# =============================================================================
# 4. 数据生成（只跑 x-y 也保持通用）
# =============================================================================

def get_data(p, eq_token, op_token, task_name):
    x = torch.arange(p)
    y = torch.arange(1, p)
    x, y = torch.cartesian_prod(x, y).T
    eq = torch.ones_like(x) * eq_token
    op = torch.ones_like(x) * op_token

    if task_name == 'x-y':
        result = (x - y) % p
    else:
        raise ValueError(f"Unknown task: {task_name}")

    data = torch.stack([x, op, y, eq, result]).T
    return data


# =============================================================================
# 5. 原子训练任务（保存逻辑按“上一个代码”）
# =============================================================================

def run_atomic_experiment(task_name, wd, seed, device_id, args):
    device = torch.device(f"cuda:{device_id}")
    torch.cuda.set_device(device_id)
    setup_torch_runtime(args)

    # 严格禁止 dropout 非 0
    if abs(args.dropout) > 1e-12:
        raise ValueError("Dropout is forbidden by experiment setting. Please set --dropout 0.0")

    torch.cuda.empty_cache()
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    full_data = get_data(args.p, args.p, args.p + 1, task_name)
    indices = torch.randperm(len(full_data))
    split = int(len(full_data) * 0.5)
    train_data = full_data[indices[:split]]
    test_data = full_data[indices[split:]]

    # 外层多进程 -> 这里必须 num_workers=0
    train_loader = DataLoader(TensorDataset(train_data), batch_size=args.batch_size, shuffle=True,
                              num_workers=0, pin_memory=True)
    llc_loader = DataLoader(TensorDataset(train_data), batch_size=args.batch_size, shuffle=False,
                            num_workers=0, pin_memory=True)
    test_loader = DataLoader(TensorDataset(test_data), batch_size=len(test_data), shuffle=False,
                             num_workers=0, pin_memory=True)

    def model_ctor():
        return GPT2Decoder(
            dim=1024, num_layers=24, num_heads=16,
            num_tokens=args.p + 2, seq_len=5,
            dropout=0.0,  # 强制 0
            use_checkpoint=bool(args.grad_ckpt)
        )

    model = model_ctor().to(device)

    if args.use_compile:
        try:
            model = torch.compile(model, mode=args.compile_mode)
        except Exception as e:
            print(f"[WARN] torch.compile failed on cuda:{device_id}: {e}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=wd, betas=(0.9, 0.98))

    # AMP
    if args.precision == "bf16":
        amp_dtype = torch.bfloat16
        use_scaler = False
    else:
        amp_dtype = torch.float16
        use_scaler = True

    scaler = torch.amp.GradScaler('cuda', enabled=use_scaler)

    llc_estimator = LLCEstimator(model_ctor, F.cross_entropy, llc_loader, device)

    # 保存目录（只保存到 results_base 下）
    task_safe = task_name.replace("/", "_div_").replace("*", "_mul_").replace("+", "_plus_")
    results_base = args.results_base
    data_dir = os.path.join(results_base, "data", task_safe, f"wd_{wd}")
    checkpoint_dir = os.path.join(results_base, "checkpoints", task_safe, f"wd_{wd}")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint_steps = {100, 1000, 10000, 100000}

    metrics = {
        'steps': [],
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': [],
        'llc': [],
        'l2_norm': [],
        'spectral_entropy': []
    }

    steps = 0
    print(f"[START] cuda:{device_id} task={task_name} wd={wd} seed={seed}")

    # 只显示一个 step tqdm，避免多进程刷屏
    show_step_pbar = (device_id == 0 and float(wd) == 0.0)
    pbar = tqdm(
        total=args.budget,
        desc=f"Steps | cuda:{device_id} wd:{wd} seed:{seed}",
        ncols=100,
        dynamic_ncols=True,
        leave=True,
        disable=not show_step_pbar
    )

    try:
        while steps < args.budget:
            for (batch_x,) in train_loader:
                model.train()
                batch_x = batch_x.to(device, non_blocking=True)
                inp, target = batch_x[:, :-1], batch_x[:, -1]

                optimizer.zero_grad(set_to_none=True)

                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    logits = model(inp)
                    loss = F.cross_entropy(logits[:, -1, :].float(), target)

                if use_scaler:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
                    optimizer.step()

                steps += 1
                pbar.update(1)

                if steps % 100 == 0:
                    # train metric（当前 batch）
                    acc = (logits[:, -1, :].argmax(-1) == target).float().mean().item()
                    loss_val = float(loss.item())

                    # test metric
                    model.eval()
                    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=amp_dtype):
                        for (test_batch,) in test_loader:
                            test_batch = test_batch.to(device, non_blocking=True)
                            t_inp, t_target = test_batch[:, :-1], test_batch[:, -1]
                            t_logits = model(t_inp)
                            test_loss = F.cross_entropy(t_logits[:, -1, :].float(), t_target).item()
                            test_acc = (t_logits[:, -1, :].argmax(-1) == t_target).float().mean().item()

                    # LLC（保持：每 100 step 估一次）
                    llc_val = llc_estimator.estimate(model.state_dict(), num_draws=args.llc_draws, lr=args.llc_lr)

                    l2_val = calc_l2_norm(model)
                    se_val = calc_spectral_entropy(model)

                    metrics['steps'].append(steps)
                    metrics['train_loss'].append(loss_val)
                    metrics['train_acc'].append(float(acc))
                    metrics['test_loss'].append(float(test_loss))
                    metrics['test_acc'].append(float(test_acc))
                    metrics['llc'].append(float(llc_val))
                    metrics['l2_norm'].append(float(l2_val))
                    metrics['spectral_entropy'].append(float(se_val))

                    # ckpt：只保存指定步
                    if steps in checkpoint_steps:
                        ckpt_path = os.path.join(checkpoint_dir, f"seed{seed}_step{steps}.pt")
                        torch.save({
                            'step': steps,
                            'task': task_name,
                            'weight_decay': wd,
                            'seed': seed,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'metrics': {
                                'train_loss': loss_val,
                                'train_acc': float(acc),
                                'test_loss': float(test_loss),
                                'test_acc': float(test_acc),
                                'llc': float(llc_val),
                                'l2_norm': float(l2_val),
                                'spectral_entropy': float(se_val)
                            }
                        }, ckpt_path)

                if steps >= args.budget:
                    break
    finally:
        pbar.close()

    # CSV：每 seed 一个文件
    csv_path = os.path.join(data_dir, f"seed{seed}.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(metrics.keys())
        writer.writerows(zip(*metrics.values()))

    final_train_acc = metrics['train_acc'][-1] if metrics['train_acc'] else 0.0
    final_test_acc = metrics['test_acc'][-1] if metrics['test_acc'] else 0.0
    print(f"✓ [DONE] cuda:{device_id} task={task_name} wd={wd} seed={seed} | Train={final_train_acc:.3f} Test={final_test_acc:.3f}")

    del model, optimizer, scaler, llc_estimator
    torch.cuda.empty_cache()

    return (task_name, wd, seed)


# =============================================================================
# 6. GPU Worker：固定 GPU + 固定 wd（你要求的逻辑）
# =============================================================================

def run_gpu_worker_fixed_wd(gpu_id, wd, seeds, args):
    torch.cuda.set_device(gpu_id)
    setup_torch_runtime(args)

    done = []
    for seed in seeds:
        done.append(run_atomic_experiment(task_name="x-y", wd=wd, seed=seed, device_id=gpu_id, args=args))
    return done


# =============================================================================
# 7. 主程序：cuda0->wd=0.0, cuda1->wd=1.0
# =============================================================================

def parse_seeds(seed_str: str):
    parts = [s.strip() for s in seed_str.split(",") if s.strip()]
    return [int(x) for x in parts]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--p", type=int, default=97)
    parser.add_argument("--budget", type=int, default=100000)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)

    # 性能开关
    parser.add_argument("--precision", type=str, default="bf16", choices=["bf16", "fp16"])
    parser.add_argument("--tf32", type=int, default=1)
    parser.add_argument("--matmul_precision", type=str, default="medium", choices=["high", "medium"])
    parser.add_argument("--cudnn_benchmark", type=int, default=1)

    parser.add_argument("--use_compile", type=int, default=0)
    parser.add_argument("--compile_mode", type=str, default="max-autotune",
                        choices=["default", "reduce-overhead", "max-autotune"])

    # 实验约束（dropout 严禁非 0）
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--grad_ckpt", type=int, default=1)
    parser.add_argument("--grad_clip", type=float, default=1.0)

    # LLC
    parser.add_argument("--llc_draws", type=int, default=20)
    parser.add_argument("--llc_lr", type=float, default=1e-4)

    # 默认两个种子（你说“只要两个种子就行”）
    parser.add_argument("--seeds", type=str, default="42")

    # 只保存到 results_base
    parser.add_argument("--results_base", type=str, default="/data/zjj/test/results")

    args = parser.parse_args()

    # 严格禁止 dropout 非0（提前失败）
    if abs(args.dropout) > 1e-12:
        raise ValueError("Dropout is forbidden by experiment setting. Please set --dropout 0.0")

    # 只用两张卡（剩下六张卡你跑别的）
    required_gpus = 2
    ngpu = torch.cuda.device_count()
    assert ngpu >= required_gpus, f"Need at least {required_gpus} GPUs, but found {ngpu}."

    seeds = parse_seeds(args.seeds)

    # 固定映射：cuda0->wd=0.0, cuda1->wd=1.0
    mapping = [(0, 0.0), (1, 1.0)]

    total_runs = len(seeds) * len(mapping)
    print(f"GPUs used: cuda:0 (wd=0.0), cuda:1 (wd=1.0)")
    print(f"Task: x-y | Seeds: {seeds} | Total runs: {total_runs}")
    print(f"Precision: {args.precision} | TF32: {bool(args.tf32)} | compile: {bool(args.use_compile)}")
    print(f"Results base: {args.results_base}")
    print(f"CSV: {args.results_base}/data/x-y/wd_*/seed*.csv")
    print(f"CKPT: {args.results_base}/checkpoints/x-y/wd_*/seed*_step*.pt")

    # spawn，避免 CUDA+fork 问题
    ctx = multiprocessing.get_context("spawn")

    with ProcessPoolExecutor(max_workers=required_gpus, mp_context=ctx) as executor:
        futures = []
        for gpu_id, wd in mapping:
            futures.append(executor.submit(run_gpu_worker_fixed_wd, gpu_id, wd, seeds, args))

        overall = tqdm(total=total_runs, desc="Overall Progress", ncols=100)

        for fut in as_completed(futures):
            done_list = fut.result()
            overall.update(len(done_list))

        overall.close()

    print("All Done.")


if __name__ == "__main__":
    main()
