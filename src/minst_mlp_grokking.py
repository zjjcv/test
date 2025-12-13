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
from torchvision import datasets, transforms
from tqdm import tqdm
from torch.optim import Optimizer
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

# =============================================================================
# 1. 基础组件 (pSGLD & LLC)
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
        
        # SGLD 参数通常需要比 SGD 小
        sgld_optim = SGLD(sampling_model.parameters(), lr=lr, noise_scale=epsilon)
        loss_trace = []
        iter_dl = iter(self.dataloader)
        
        for _ in range(num_draws):
            try:
                batch = next(iter_dl)
            except StopIteration:
                iter_dl = iter(self.dataloader)
                batch = next(iter_dl)
            
            inp, target = batch[0].to(self.device), batch[1].to(self.device)
            
            sgld_optim.zero_grad()
            logits = sampling_model(inp)
            loss = self.criterion(logits, target)
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
                inp, target = batch[0].to(self.device), batch[1].to(self.device)
                logits = model(inp)
                loss = self.criterion(logits, target)
                total_loss += loss.item() * inp.size(0)
                total_count += inp.size(0)
        return total_loss / total_count

# =============================================================================
# 2. 模型定义 (MLP: 3层, 宽度200, ReLU)
# =============================================================================

class MLP(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=200, output_dim=10):
        super().__init__()
        # 严格按照论文：Depth=3 (Input->Hidden1->Hidden2->Output)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)

def rescale_init(model, init_scale):
    """
    自定义初始化：Kaiming Uniform * Init Scale
    较大的 Init Scale (alpha > 1) 倾向于让模型进入 Lazy Regime (记忆)。
    """
    for m in model.modules():
        if isinstance(m, nn.Linear):
            # Kaiming Uniform 是 PyTorch 默认的 Linear 初始化，这里显式调用以防万一
            nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
            # 关键步骤：扩大初始化方差
            with torch.no_grad():
                m.weight.data *= init_scale
                if m.bias is not None:
                    # Bias 通常初始化为 0 或很小
                    nn.init.zeros_(m.bias)

def calc_l2_norm(model):
    total_norm = 0.0
    for p in model.parameters():
        param_norm = p.data.norm(2)
        total_norm += param_norm.item() ** 2
    return total_norm ** 0.5

# =============================================================================
# 3. 数据生成: MNIST Subset + One-Hot Encoding
# =============================================================================

def get_mnist_data(subset_size, noise_dim, data_root='../data'):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_ds_raw = datasets.MNIST(data_root, train=True, download=False, transform=transform)
    test_ds_raw = datasets.MNIST(data_root, train=False, download=False, transform=transform)
    
    # 1. 随机子集 (Key Modification 1: Small Data)
    indices = torch.randperm(len(train_ds_raw))[:subset_size]
    
    # 处理训练数据
    train_data = train_ds_raw.data[indices].float() / 255.0
    train_data = (train_data - 0.1307) / 0.3081
    train_data = train_data.view(train_data.size(0), -1)
    
    # 2. One-Hot 编码 (Key Modification 3: MSE Loss Requirement)
    train_targets_idx = train_ds_raw.targets[indices]
    train_targets = F.one_hot(train_targets_idx, num_classes=10).float()
    
    # (可选) 增加虚拟噪声维度
    if noise_dim > 0:
        noise = torch.randn(train_data.size(0), noise_dim)
        train_data = torch.cat([train_data, noise], dim=1)
        
    train_dataset = TensorDataset(train_data, train_targets)
    
    # 处理测试数据
    test_data = test_ds_raw.data.float() / 255.0
    test_data = (test_data - 0.1307) / 0.3081
    test_data = test_data.view(test_data.size(0), -1)
    
    test_targets_idx = test_ds_raw.targets
    test_targets = F.one_hot(test_targets_idx, num_classes=10).float()
    
    if noise_dim > 0:
        test_noise = torch.randn(test_data.size(0), noise_dim)
        test_data = torch.cat([test_data, test_noise], dim=1)
        
    test_dataset = TensorDataset(test_data, test_targets)
    
    return train_dataset, test_dataset

# =============================================================================
# 4. 原子训练任务
# =============================================================================

def run_atomic_experiment(config):
    task_name, wd, seed, device_id, args, output_dir = config
    device = torch.device(f"cuda:{device_id}")
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    
    # 1. 数据准备
    train_dataset, test_dataset = get_mnist_data(args.subset_size, args.noise_dim)
    
    # 注意：使用 MSE Loss 时，Batch Size 最好不要太小，否则梯度太吵
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    llc_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False, num_workers=0)

    # 2. 模型初始化
    input_dim = 784 + args.noise_dim
    model = MLP(input_dim=input_dim, hidden_dim=args.hidden_dim, output_dim=10)
    
    # Key Modification 2: Large Initialization Scale
    rescale_init(model, args.init_scale)
    model.to(device)
    
    # 3. 优化器与损失函数
    # 使用 AdamW，One-hot MSE Loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=wd, betas=(0.9, 0.98))
    # MSE Loss 不需要 LogSoftmax，直接作用于 Logits
    criterion = nn.MSELoss() 
    
    scaler = torch.amp.GradScaler('cuda')
    llc_estimator = LLCEstimator(model, criterion, llc_loader, device)

    metrics = {'steps': [], 'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': [], 'llc': [], 'l2_norm': []}
    steps = 0
    
    LOG_INTERVAL = 100
    LLC_INTERVAL = 100 

    while steps < args.budget:
        for inp, target in train_loader:
            model.train()
            inp, target = inp.to(device), target.to(device)

            with torch.amp.autocast('cuda'):
                logits = model(inp)
                loss = criterion(logits, target) # MSE Loss

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            steps += 1
            
            if steps % LOG_INTERVAL == 0:
                # 验证 (MSE 输出的是 Raw Logits，取 Argmax 依然是分类结果)
                acc = (logits.argmax(-1) == target.argmax(-1)).float().mean().item()
                loss_val = loss.item()
                
                model.eval()
                t_loss_sum = 0
                t_acc_sum = 0
                t_batches = 0
                with torch.no_grad():
                    for t_inp, t_target in test_loader:
                        t_inp, t_target = t_inp.to(device), t_target.to(device)
                        t_logits = model(t_inp)
                        t_loss_sum += criterion(t_logits, t_target).item()
                        t_acc_sum += (t_logits.argmax(-1) == t_target.argmax(-1)).float().mean().item()
                        t_batches += 1
                
                test_loss = t_loss_sum / t_batches
                test_acc = t_acc_sum / t_batches
                
                # LLC 计算
                llc_val = np.nan
                if steps % LLC_INTERVAL == 0 or steps == args.budget:
                    # MSE Loss 的数值尺度通常比 CrossEntropy 小很多 (因为是平方误差)
                    # 可能需要调大 epsilon 或 lr 来获得足够的采样波动
                    llc_val = llc_estimator.estimate(model.state_dict(), num_draws=40, lr=1e-4, epsilon=1e-3)
                elif len(metrics['llc']) > 0:
                    llc_val = metrics['llc'][-1]

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
                
    csv_name = f"seed{seed}.csv"
    with open(os.path.join(output_dir, csv_name), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(metrics.keys())
        writer.writerows(zip(*metrics.values()))
    
    return (task_name, wd, metrics)

# =============================================================================
# 5. 绘图函数
# =============================================================================

def plot_multiseed_results(results_list, task_name, wd, save_dir):
    if not results_list: return
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

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(24, 5))
    
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
    ax2.plot(steps, tr_loss_m, color='blue', label='Train')
    ax2.plot(steps, te_loss_m, color='red', label='Test')
    ax2.set_title('MSE Loss')
    ax2.set_xlabel('Steps')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3)

    # LLC
    valid_mask = ~np.isnan(llc_m)
    if np.any(valid_mask):
        v_steps = np.array(steps)[valid_mask]
        v_llc_m = llc_m[valid_mask]
        v_llc_s = llc_s[valid_mask]
        ax3.plot(v_steps, v_llc_m, color='green', label='LLC')
        ax3.fill_between(v_steps, v_llc_m-v_llc_s, v_llc_m+v_llc_s, color='green', alpha=0.2)
    ax3.set_title('Complexity (LLC)')
    ax3.set_xlabel('Steps')
    ax3.set_xscale('log')
    ax3.grid(True, alpha=0.3)

    # L2 Norm
    ax4.plot(steps, l2_m, color='purple', label='L2 Norm')
    ax4.fill_between(steps, l2_m-l2_s, l2_m+l2_s, color='purple', alpha=0.2)
    ax4.set_title('Parameter Norm (L2)')
    ax4.set_xlabel('Steps')
    ax4.set_xscale('log')
    ax4.grid(True, alpha=0.3)

    plt.suptitle(rf"Task: {task_name} | WD: {wd}")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{task_name}_wd{wd}.png"), dpi=150)
    plt.close()

# =============================================================================
# 6. 主程序
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    # 论文设定: 
    # - Budget 需要足够长 (e.g. 10^5 steps)
    # - Learning Rate 1e-3 (AdamW default)
    # - Width 200
    parser.add_argument("--budget", type=int, default=100000, help="Training steps") 
    parser.add_argument("--batch_size", type=int, default=200) # 这里用 Full Batch 1000 其实更好，但 200 也可以
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden_dim", type=int, default=200, help="MLP width (Paper: 200)")
    
    # 诱导 Grokking 的关键参数
    parser.add_argument("--subset_size", type=int, default=1000, help="Reduced training set size (Paper: 1000)")
    parser.add_argument("--noise_dim", type=int, default=20, help="Optional noise dim (Paper didn't stress this, but helps)")
    parser.add_argument("--init_scale", type=float, default=8.0, help="Init Scale alpha (Paper: >1, try 8.0)")
    
    parser.add_argument("--max_workers", type=int, default=8)
    args = parser.parse_args()

    print("Pre-downloading MNIST data...")
    datasets.MNIST('../data', train=True, download=True)
    datasets.MNIST('../data', train=False, download=True)

    seeds = [42, 101, 2025]
    # seeds = [42]
    # Grokking 需要较大的 Weight Decay
    weight_decays = [0.0, 1.0] 
    
    base_dir = os.path.abspath(f"../figures/mnist_grokking_mse_{datetime.now().strftime('%m%d_%H%M')}")
    os.makedirs(base_dir, exist_ok=True)

    task_queue = []
    task_name = f"mnist_sub{args.subset_size}_init{args.init_scale}"
    
    for wd in weight_decays:
        out_dir = os.path.join(base_dir, task_name, f"wd_{wd}")
        os.makedirs(out_dir, exist_ok=True)
        for seed in seeds:
            gpu_id = len(task_queue) % torch.cuda.device_count() if torch.cuda.is_available() else 0
            config = (task_name, wd, seed, gpu_id, args, out_dir)
            task_queue.append(config)

    print(f"Total experiments: {len(task_queue)}")
    print(f"Config: Subset={args.subset_size}, InitScale={args.init_scale}, MSE Loss")
    print(f"Output dir: {base_dir}")

    results_cache = {}
    
    try:
        multiprocessing.set_start_method('spawn')
    except RuntimeError:
        pass

    with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        futures = [executor.submit(run_atomic_experiment, cfg) for cfg in task_queue]
        
        for future in tqdm(futures, total=len(futures), desc="Training"):
            try:
                t_name, wd, metrics = future.result()
                key = (t_name, wd)
                if key not in results_cache: results_cache[key] = []
                results_cache[key].append(metrics)
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"\n!!! Error: {e}")

    print("\nGenerating plots...")
    for (t_name, wd), metrics_list in results_cache.items():
        if len(metrics_list) > 0:
            out_dir = os.path.join(base_dir, t_name, f"wd_{wd}")
            plot_multiseed_results(metrics_list, t_name, wd, out_dir)

    print(f"\nAll Done!")

if __name__ == "__main__":
    main()