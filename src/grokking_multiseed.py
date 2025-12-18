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
from torch.optim import Optimizer
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
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                p.data.add_(d_p, alpha=-0.5 * group['lr'])
                noise_std = torch.sqrt(torch.tensor(group['lr'], device=p.data.device)) * group['noise_scale']
                noise = torch.randn_like(p.data) * noise_std
                p.data.add_(noise)
        return loss


class LLCEstimator:
    """
    - 不 deepcopy(model)，避免与 torch.compile 包装冲突
    - 通过 model_ctor() 造同构模型并 load_state_dict
    """
    def __init__(self, model_ctor, criterion, dataloader, device, amp_dtype=None):
        self.model_ctor = model_ctor
        self.criterion = criterion
        self.dataloader = dataloader
        self.device = device
        self.amp_dtype = amp_dtype

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

            if self.amp_dtype is None:
                logits = sampling_model(inp)
                loss = self.criterion(logits[:, -1, :].float(), target)
            else:
                with torch.autocast(device_type="cuda", dtype=self.amp_dtype):
                    logits = sampling_model(inp)
                    loss = self.criterion(logits[:, -1, :].float(), target)

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

                if self.amp_dtype is None:
                    logits = model(inp)
                    loss = self.criterion(logits[:, -1, :].float(), target)
                else:
                    with torch.autocast(device_type="cuda", dtype=self.amp_dtype):
                        logits = model(inp)
                        loss = self.criterion(logits[:, -1, :].float(), target)

                total_loss += loss.item() * batch_x.size(0)
                total_count += batch_x.size(0)
        return total_loss / max(total_count, 1)


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
    elif task_name == 'x2+y2':
        result = (x**2 + y**2) % p
    elif task_name == 'x2+xy+y2':
        result = (x**2 + x*y + y**2) % p
    elif task_name == 'x2+xy+y2+x':
        result = (x**2 + x*y + y**2 + x) % p
    elif task_name == 'x3+xy':
        result = (x**3 + x*y) % p
    elif task_name == 'x3+xy2+y':
        result = (x**3 + x*y**2 + y) % p
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
    llc_loader = _make_dataloader(TensorDataset(train_data), args, shuffle=False)
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

    llc_estimator = LLCEstimator(
        model_ctor=model_ctor,
        criterion=F.cross_entropy,
        dataloader=llc_loader,
        device=device,
        amp_dtype=amp_dtype if args.llc_use_amp else None
    )

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
        'spectral_entropy': [],
        'attention_spectral_entropy': [],
        'embedding_spectral_entropy': []
    }

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
                acc = (logits[:, -1, :].argmax(-1) == target).float().mean().item()
                loss_val = float(loss.item())

                model.eval()
                with torch.no_grad():
                    for (test_batch,) in test_loader:
                        test_batch = test_batch.to(device, non_blocking=True)
                        t_inp, t_target = test_batch[:, :-1], test_batch[:, -1]
                        if use_amp:
                            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                                t_logits = model(t_inp)
                                test_loss = F.cross_entropy(t_logits[:, -1, :].float(), t_target).item()
                        else:
                            t_logits = model(t_inp)
                            test_loss = F.cross_entropy(t_logits[:, -1, :].float(), t_target).item()
                        test_acc = (t_logits[:, -1, :].argmax(-1) == t_target).float().mean().item()

                llc_val = llc_estimator.estimate(
                    model.state_dict(),
                    num_draws=args.llc_draws,
                    lr=args.llc_lr
                )

                l2_val = calc_l2_norm(model)
                spectral_entropy_val = calc_spectral_entropy(model)
                attention_entropy_val = calc_attention_spectral_entropy(model)
                embedding_entropy_val = calc_embedding_spectral_entropy(model)

                metrics['steps'].append(steps)
                metrics['train_loss'].append(loss_val)
                metrics['train_acc'].append(acc)
                metrics['test_loss'].append(float(test_loss))
                metrics['test_acc'].append(float(test_acc))
                metrics['llc'].append(float(llc_val))
                metrics['l2_norm'].append(float(l2_val))
                metrics['spectral_entropy'].append(float(spectral_entropy_val))
                metrics['attention_spectral_entropy'].append(float(attention_entropy_val))
                metrics['embedding_spectral_entropy'].append(float(embedding_entropy_val))

                if steps in checkpoint_steps:
                    checkpoint_path = os.path.join(checkpoint_dir, f"seed{seed}_step{steps}.pt")
                    torch.save({
                        'step': steps,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'train_loss': loss_val,
                        'train_acc': acc,
                        'test_loss': float(test_loss),
                        'test_acc': float(test_acc)
                    }, checkpoint_path)

            if steps >= args.budget:
                break

    csv_path = os.path.join(data_dir, f"seed{seed}.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(metrics.keys())
        writer.writerows(zip(*metrics.values()))

    final_train_acc = metrics['train_acc'][-1] if metrics['train_acc'] else 0.0
    final_test_acc = metrics['test_acc'][-1] if metrics['test_acc'] else 0.0
    print(f"✓ [DONE] GPU:{device_id} {task_name} WD={wd} S={seed} | Train:{final_train_acc:.3f} Test:{final_test_acc:.3f}")

    del model, optimizer, scaler, llc_estimator
    torch.cuda.empty_cache()
    return (task_name, wd, metrics)


# =============================================================================
# 5. 绘图函数
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
    attn_se_m, attn_se_s = get_stats('attention_spectral_entropy')
    emb_se_m, emb_se_s = get_stats('embedding_spectral_entropy')

    llc_raw = np.array([r['llc'] for r in results_list])
    llc_m = np.nanmean(llc_raw, axis=0)
    llc_s = np.nanstd(llc_raw, axis=0)

    fig, axes = plt.subplots(2, 4, figsize=(32, 10))
    ax1, ax2, ax3, ax4 = axes[0]
    ax5, ax6, ax7, ax8 = axes[1]

    ax1.plot(steps, tr_acc_m, color='blue', label='Train')
    ax1.fill_between(steps, tr_acc_m-tr_acc_s, tr_acc_m+tr_acc_s, color='blue', alpha=0.2)
    ax1.plot(steps, te_acc_m, color='red', label='Test')
    ax1.fill_between(steps, te_acc_m-te_acc_s, te_acc_m+te_acc_s, color='red', alpha=0.2)
    ax1.set_title(f'Accuracy (WD={wd})')
    ax1.set_xlabel('Steps')
    ax1.set_xscale('log')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(steps, tr_loss_m, color='blue')
    ax2.fill_between(steps, tr_loss_m-tr_loss_s, tr_loss_m+tr_loss_s, color='blue', alpha=0.2)
    ax2.plot(steps, te_loss_m, color='red')
    ax2.fill_between(steps, te_loss_m-te_loss_s, te_loss_m+te_loss_s, color='red', alpha=0.2)
    ax2.set_title('Loss')
    ax2.set_xlabel('Steps')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3)

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

    ax4.plot(steps, l2_m, color='purple', label='L2 Norm')
    ax4.fill_between(steps, l2_m-l2_s, l2_m+l2_s, color='purple', alpha=0.2)
    ax4.set_title('Parameter Norm (L2)')
    ax4.set_xlabel('Steps')
    ax4.set_ylabel('L2 Norm')
    ax4.set_xscale('log')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    ax5.plot(steps, se_m, color='orange', label='Overall')
    ax5.fill_between(steps, se_m-se_s, se_m+se_s, color='orange', alpha=0.2)
    ax5.set_title('Spectral Entropy (Overall)')
    ax5.set_xlabel('Steps')
    ax5.set_ylabel('Entropy')
    ax5.set_xscale('log')
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    ax6.plot(steps, attn_se_m, color='cyan', label='Attention')
    ax6.fill_between(steps, attn_se_m-attn_se_s, attn_se_m+attn_se_s, color='cyan', alpha=0.2)
    ax6.set_title('Spectral Entropy (Attention)')
    ax6.set_xlabel('Steps')
    ax6.set_ylabel('Entropy')
    ax6.set_xscale('log')
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    ax7.plot(steps, emb_se_m, color='magenta', label='Embedding')
    ax7.fill_between(steps, emb_se_m-emb_se_s, emb_se_m+emb_se_s, color='magenta', alpha=0.2)
    ax7.set_title('Spectral Entropy (Embedding)')
    ax7.set_xlabel('Steps')
    ax7.set_ylabel('Entropy')
    ax7.set_xscale('log')
    ax7.legend()
    ax7.grid(True, alpha=0.3)

    llc_l2_m = v_llc_m * l2_m[valid_mask]
    llc_l2_s = np.sqrt((v_llc_s * l2_m[valid_mask])**2 + (v_llc_m * l2_s[valid_mask])**2)
    ax8.plot(v_steps, llc_l2_m, color='brown', label='LLC × L2')
    ax8.fill_between(v_steps, llc_l2_m-llc_l2_s, llc_l2_m+llc_l2_s, color='brown', alpha=0.2)
    ax8.set_title('Complexity (LLC × L2)')
    ax8.set_xlabel('Steps')
    ax8.set_ylabel('LLC × L2')
    ax8.set_xscale('log')
    ax8.legend()
    ax8.grid(True, alpha=0.3)

    plt.suptitle(rf"Task: {task_name} | WD: {wd} | (Mean $\pm$ Std over 3 seeds)")
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f"{task_name.replace('/','_')}_wd{wd}.png"), dpi=150)
    plt.close()


# =============================================================================
# 6. 多卡多并发：GPU 绑定 worker（一个进程绑定一张卡，循环抢任务）
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
# 7. 主程序：8 卡 + 每卡并发多个实验进程（一次性跑完 54 个实验）
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

    # LLC（很重）
    parser.add_argument("--llc_draws", type=int, default=20)
    parser.add_argument("--llc_lr", type=float, default=1e-4)
    parser.add_argument("--llc_use_amp", type=int, default=0)

    # CPU intra-op threads
    parser.add_argument("--cpu_threads", type=int, default=1)

    # 输出目录
    parser.add_argument("--results_base", type=str, default="/data/zjj/test/results")

    # 关键：每张 GPU 同时跑多少个实验（多进程并发）
    parser.add_argument("--jobs_per_gpu", type=int, default=1,
                        help="Concurrent experiment processes per GPU (increase to 2/3 if memory allows)")

    args = parser.parse_args()

    # 任务定义
    seeds = [42, 101, 2025]
    weight_decays = [0.0, 1.0]
    tasks = [
        'x+y', 'x-y', 'x*y', 'x_div_y',
        'x2+y2', 'x2+xy+y2', 'x2+xy+y2+x',
        'x3+xy', 'x3+xy2+y'
    ]

    num_gpus = torch.cuda.device_count()
    assert num_gpus > 0, "No CUDA GPUs found."

    base_dir = os.path.abspath(f"grokking_full_parallel_{datetime.now().strftime('%m%d_%H%M')}")
    os.makedirs(base_dir, exist_ok=True)

    # 生成全部 54 个实验任务（不预先绑定 GPU：由 worker 动态抢任务，负载最均匀）
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
    print(f"Results base: {args.results_base}")
    print(f"Plot base dir: {base_dir}")

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

    results_cache = {}
    finished = 0

    pbar = tqdm(total=total_jobs, desc="Overall Progress", ncols=100)

    while finished < total_jobs:
        status, payload = result_queue.get()

        if status == "ok":
            task_name, wd, metrics = payload
            results_cache.setdefault((task_name, wd), []).append(metrics)
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

    # 汇总绘图
    print("\nGenerating plots...")
    for (task, wd), metrics_list in results_cache.items():
        if len(metrics_list) == len(seeds):
            task_safe = task.replace("/", "_div_").replace("*", "_mul_").replace("+", "_plus_")
            save_dir = os.path.join(base_dir, task_safe, f"wd_{wd}")
            plot_multiseed_results(metrics_list, task, wd, save_dir)
        else:
            print(f"Skipping plot for {task} WD={wd} (incomplete data: {len(metrics_list)}/{len(seeds)})")

    print(f"\nAll Done! Plots in: {base_dir}")
    print(f"CSV/checkpoints in: {args.results_base}")


if __name__ == "__main__":
    main()
