import os
import sys
import matplotlib
matplotlib.use('Agg')  # 适用于服务器环境，不显示窗口直接保存图片
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Subset
import numpy as np
from tqdm import tqdm

# =========================================================================
# 0. Path Hack (确保能导入 model 和 utils)
# =========================================================================
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

from model.models import get_model
# 借用 train.py 里的配置，避免重复定义
from src.train import DATASET_CONFIG, load_dataset

# =========================================================================
# 1. Configuration for Grokking Induction
# =========================================================================

DATA_DIR = os.path.join(project_root, 'data')
PLOT_DIR = os.path.join(project_root, 'grok_verification_plots')
os.makedirs(PLOT_DIR, exist_ok=True)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

INDUCTION_CONFIG = {
    'symbolic': {'wd': 0.01,  'fraction': 0.1, 'lr': 1e-4},
    'vision':   {'wd': 0.05,  'fraction': 0.1, 'lr': 1e-3},
    'text':     {'wd': 0.1,   'fraction': 0.1, 'lr': 5e-4},
    'logic':    {'wd': 0.05,  'fraction': 0.1, 'lr': 1e-3},
}

# 你要求：总步数 100K
MAX_STEPS = 100000

# =========================================================================
# 1.1 自动生成 log 采样点（覆盖 1 ~ 100K）
# =========================================================================

def make_log_eval_steps(max_steps: int, points_per_decade: int = 10):
    """
    生成对数间隔的评估步数点（不包含 0，避免 log 坐标报错）
    - points_per_decade: 每个数量级采样多少点（10 表示比较细）
    """
    steps = set([1, max_steps])
    max_pow = int(np.floor(np.log10(max_steps)))

    for p in range(0, max_pow + 1):
        base = 10 ** p
        for k in range(points_per_decade):
            s = int(round(base * (10 ** (k / points_per_decade))))
            if 1 <= s <= max_steps:
                steps.add(s)

    return sorted(steps)

LOG_EVAL_STEPS = make_log_eval_steps(MAX_STEPS, points_per_decade=10)
LOG_EVAL_STEPS_SET = set(LOG_EVAL_STEPS)

# =========================================================================
# 2. Helper Functions
# =========================================================================

def subsample_dataset(loader, fraction):
    dataset = loader.dataset
    n_samples = len(dataset)
    n_keep = max(1, int(n_samples * fraction))

    indices = torch.randperm(n_samples)[:n_keep]
    subset = Subset(dataset, indices)

    print(f"✂️  Data Scarcity Induced: {n_samples} -> {n_keep} samples (Frac: {fraction})")

    new_loader = DataLoader(subset, batch_size=loader.batch_size, shuffle=True)
    return new_loader

def check_grokking_pattern(train_accs, test_accs):
    train_arr = np.array(train_accs)
    test_arr = np.array(test_accs)

    if len(train_arr) == 0:
        return "No Data"

    if train_arr[-1] < 0.98:
        return "Underfitting (No Memorization)"
    if test_arr[-1] < 0.85:
        return "Overfitting (No Generalization)"

    t_mem_idx = np.where(train_arr >= 0.95)[0]
    if len(t_mem_idx) == 0:
        return "Slow Learning"
    t_mem = t_mem_idx[0]

    acc_at_mem = test_arr[t_mem]
    if acc_at_mem > 0.90:
        return "Fast Generalization (Too Easy?)"
    elif acc_at_mem < 0.80:
        return "✅ GROKKING DETECTED"
    else:
        return "Weak Grokking"

# =========================================================================
# 3. Verification Loop
# =========================================================================

def verify_task(dataset_name):
    if dataset_name not in DATASET_CONFIG:
        return
    modality, input_shape, output_dim = DATASET_CONFIG[dataset_name]

    print(f"\n🔍 Verifying: [{dataset_name}] ({modality})")

    # 1. Load & Induce Scarcity
    train_loader, test_loader = load_dataset(dataset_name, modality)
    if train_loader is None:
        return

    cfg = INDUCTION_CONFIG[modality]
    train_loader = subsample_dataset(train_loader, cfg['fraction'])

    # 2. Init Model & Optimizer
    model = get_model(modality, input_shape, output_dim, DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['wd'])

    criterion = nn.CrossEntropyLoss()
    is_regression = False
    if modality == 'symbolic' and 'regression' in dataset_name:
        criterion = nn.MSELoss()
        is_regression = True
    elif 'geometry' in dataset_name:
        criterion = nn.MSELoss()
        is_regression = True

    # 3. Training Loop
    train_acc_history = []
    test_acc_history = []
    steps_logged = []

    train_iter = iter(train_loader)

    # 这里改为 step 从 1 开始，保证 log 横轴合法
    pbar = tqdm(range(1, MAX_STEPS + 1), desc=f"Grokking {dataset_name}", leave=False)

    for step in pbar:
        # Get Batch
        try:
            inputs, targets = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            inputs, targets = next(train_iter)

        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

        # Update
        optimizer.zero_grad(set_to_none=True)
        out = model(inputs)

        if is_regression:
            loss = criterion(out, targets.float())
        else:
            loss = criterion(out, targets.long())

        loss.backward()
        optimizer.step()

        # Eval & Track：改为 log 采样点触发
        if step in LOG_EVAL_STEPS_SET:
            model.eval()

            def get_acc(loader):
                correct = 0
                total = 0
                with torch.no_grad():
                    for x, y in loader:
                        x, y = x.to(DEVICE), y.to(DEVICE)
                        pred = model(x)
                        if is_regression:
                            err = (pred - y.float()).pow(2).sum(dim=1)
                            correct += (err < 0.05).sum().item()
                        else:
                            correct += (pred.argmax(1) == y).sum().item()
                        total += y.size(0)
                return correct / total if total > 0 else 0.0

            tr_acc = get_acc(train_loader)
            te_acc = get_acc(test_loader)

            train_acc_history.append(tr_acc)
            test_acc_history.append(te_acc)
            steps_logged.append(step)

            model.train()
            pbar.set_postfix({'Tr': f'{tr_acc:.2f}', 'Te': f'{te_acc:.2f}'})

            if te_acc > 0.99 and step > 2000:
                break

    # 4. Result Analysis & Plotting
    status = check_grokking_pattern(train_acc_history, test_acc_history)
    print(f"   📊 Result: {status}")

    plt.figure(figsize=(10, 6))
    plt.plot(steps_logged, train_acc_history, label='Train Acc', linestyle='--')
    plt.plot(steps_logged, test_acc_history, label='Test Acc', linewidth=2)

    plt.title(f"Grokking Check: {dataset_name}\nStatus: {status} | WD={cfg['wd']} | Frac={cfg['fraction']}")
    plt.xlabel("Steps (log)")
    plt.ylabel("Accuracy")
    plt.ylim(-0.05, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend()

    # 横轴 log
    plt.xscale("log")

    save_path = os.path.join(PLOT_DIR, f"{dataset_name}.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"   🖼️  Curve saved to: {save_path}")

# =========================================================================
# 4. Main Entry
# =========================================================================

if __name__ == "__main__":
    print("🚀 Starting Grokking Pilot Study...")
    print(f"📂 Plots will be saved to: {PLOT_DIR}")
    print(f"ℹ️  MAX_STEPS={MAX_STEPS}, log-eval points={len(LOG_EVAL_STEPS)}")
    print(f"ℹ️  First 20 LOG_EVAL_STEPS: {LOG_EVAL_STEPS[:20]}")
    print(f"ℹ️  Last 10 LOG_EVAL_STEPS: {LOG_EVAL_STEPS[-10:]}")

    tasks_to_verify = list(DATASET_CONFIG.keys())

    for task in tasks_to_verify:
        try:
            verify_task(task)
        except Exception as e:
            print(f"❌ Error verifying {task}: {e}")
            import traceback
            traceback.print_exc()

    print("\n✅ Verification Complete. Check the 'grok_verification_plots' folder.")
