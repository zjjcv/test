import math
import os
import sys
import csv
import numpy as np
import argparse
from datetime import datetime
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import MNIST, CIFAR10, Omniglot, ImageFolder
import torch.multiprocessing as mp
from tqdm import tqdm

# 动态导入上一级目录的 regularization.py
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from regularization import get_method_config, SAM, ASAM, SAMPa, SSAM, CPR

# =============================================================================
# 0. ViT-Small Model (Vision Transformer)
# =============================================================================

class PatchEmbed(nn.Module):
    def __init__(self, img_size=64, patch_size=8, in_chans=3, embed_dim=384):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=6, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class ViTSmall(nn.Module):
    """
    Standard ViT-Small configuration:
    - Patch Size: 8 (for small images like 32x32 or 64x64)
    - Embed Dim: 384
    - Depth: 12 layers
    - Heads: 6
    - MLP Ratio: 4.0
    """
    def __init__(self, img_size=64, patch_size=8, in_chans=3, num_classes=10, 
                 embed_dim=384, depth=12, num_heads=6, mlp_ratio=4., drop_rate=0.):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_rate, depth)]  # stochastic depth decay rule
        
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(embed_dim),
                Attention(embed_dim, num_heads=num_heads, attn_drop=drop_rate, proj_drop=drop_rate),
                nn.Dropout(drop_rate), # Residual connection handled in forward if needed, but standard Block has shortcut
                nn.LayerNorm(embed_dim),
                MLP(embed_dim, int(embed_dim * mlp_ratio), drop=drop_rate),
                nn.Dropout(drop_rate)
            ) for i in range(depth)
        ])
        
        # Standard ViT Block logic involves residuals, re-implementing block forward for clarity
        self.layers = nn.ModuleList()
        for _ in range(depth):
            self.layers.append(nn.ModuleDict({
                'norm1': nn.LayerNorm(embed_dim),
                'attn': Attention(embed_dim, num_heads, attn_drop=drop_rate, proj_drop=drop_rate),
                'norm2': nn.LayerNorm(embed_dim),
                'mlp': MLP(embed_dim, int(embed_dim * mlp_ratio), drop=drop_rate)
            }))

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        nn.init.trunc_normal_(self.pos_embed, std=.02)
        nn.init.trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for layer in self.layers:
            # Block: x = x + attn(norm1(x))
            x = x + layer['attn'](layer['norm1'](x))
            # Block: x = x + mlp(norm2(x))
            x = x + layer['mlp'](layer['norm2'](x))

        x = self.norm(x)
        return self.head(x[:, 0])

# =============================================================================
# 1. Dataset Factory
# =============================================================================

def get_vision_loader(dataset_name, batch_size, root='./data', num_workers=0):
    """
    Unified loader for MNIST, CIFAR10, Omniglot, Tiny-ImageNet.
    Forces resize to 64x64 for standardizing ViT Patch Size.
    """
    dataset_name = dataset_name.lower()
    
    # Standard transforms
    # Note: To observe Grokking (Memorization -> Generalization), 
    # weak augmentation or NO augmentation is sometimes preferred initially.
    # However, to train ViT well, some aug is usually needed.
    # We use minimal augmentation here.
    
    mean, std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
    
    transform_train = transforms.Compose([
        transforms.Resize((64, 64)),
        # transforms.RandomHorizontalFlip(), # Optional: Comment out for pure memorization test
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    
    transform_test = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    # 1. CIFAR-10
    if dataset_name == 'cifar10':
        train_set = CIFAR10(root=root, train=True, download=True, transform=transform_train)
        test_set = CIFAR10(root=root, train=False, download=True, transform=transform_test)
        num_classes = 10
        in_chans = 3

    # 2. MNIST
    elif dataset_name == 'mnist':
        # MNIST is 1-channel, we duplicate to 3 for compatibility or modify ViT
        # Let's modify transform to output 3 channels for simplicity
        transform_train_mnist = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        transform_test_mnist = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        train_set = MNIST(root=root, train=True, download=True, transform=transform_train_mnist)
        test_set = MNIST(root=root, train=False, download=True, transform=transform_test_mnist)
        num_classes = 10
        in_chans = 3

    # 3. Omniglot
    elif dataset_name == 'omniglot':
        # Omniglot has background and evaluation sets. We combine or use background.
        # It has ~1600 classes.
        transform_omni = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean, std), # Invert colors if needed
        ])
        # Download usually creates 'omniglot-py' folder
        try:
            train_set = Omniglot(root=root, background=True, download=True, transform=transform_omni)
            test_set = Omniglot(root=root, background=False, download=True, transform=transform_omni)
        except:
            print("Omniglot download failed or structure issue. Please check torch version.")
            raise
        num_classes = 964 # Background set classes
        in_chans = 3

    # 4. Tiny-ImageNet
    elif 'tiny' in dataset_name:
        # Assumes folder structure: root/tiny-imagenet-200/train and /val
        # User needs to download this manually usually, or we assume it exists
        tiny_root = os.path.join(root, 'tiny-imagenet-200')
        if not os.path.exists(tiny_root):
            raise FileNotFoundError(f"Tiny ImageNet not found at {tiny_root}. Please download and extract it.")
        
        train_set = ImageFolder(os.path.join(tiny_root, 'train'), transform=transform_train)
        test_set = ImageFolder(os.path.join(tiny_root, 'val'), transform=transform_test)
        num_classes = 200
        in_chans = 3

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size*2, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, test_loader, num_classes, in_chans

# =============================================================================
# 2. Setup Runtime
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

# =============================================================================
# 3. Worker Process
# =============================================================================

def train_worker(config):
    method_name, gpu_id, args, output_dir = config
    
    # Bind GPU
    device = torch.device(f"cuda:{gpu_id}")
    torch.cuda.set_device(gpu_id)
    setup_torch_runtime(args)
    
    # Fix Seed
    seed = 2025
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    print(f"🚀 [Start] {args.dataset.upper()} | {method_name} on GPU {gpu_id}")

    # Load Data
    train_loader, test_loader, num_classes, in_chans = get_vision_loader(
        args.dataset, args.batch_size, root=args.data_root
    )

    # Initialize ViT-Small
    # Standard ViT-Small: patch=8 (for 64x64 img), embed=384, depth=12, heads=6
    model = ViTSmall(
        img_size=64, 
        patch_size=8, 
        in_chans=in_chans, 
        num_classes=num_classes,
        embed_dim=384, 
        depth=12, 
        num_heads=6,
        drop_rate=0.0 # Typically 0 for Grokking studies to allow overfitting
    ).to(device)
    
    # Get Regularization Method
    args.device = device 
    optimizer, loss_fn, hooks = get_method_config(method_name, model, args)
    
    is_sam_type = isinstance(optimizer, (SAM, ASAM, SAMPa, SSAM))
    is_llc_noise = (method_name == "LLCNoise")

    metrics = {'steps': [], 'train_acc': [], 'test_acc': [], 'train_loss': []}
    steps = 0
    epoch = 0
    
    pbar = tqdm(total=args.budget, desc=f"{method_name[:10]}", position=gpu_id)
    
    # Vision tasks use epochs usually, but we stick to 'steps' budget for consistency
    while steps < args.budget:
        epoch += 1
        model.train()
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # --- Training Step ---
            
            if is_sam_type:
                # 1. Forward & Perturb
                optimizer.zero_grad()
                logits = model(inputs)
                loss = loss_fn(logits, targets)
                loss.backward()
                optimizer.first_step(zero_grad=True)
                
                # 2. Forward at Perturbed & Update
                logits_2 = model(inputs)
                loss_2 = loss_fn(logits_2, targets)
                loss_2.backward()
                optimizer.second_step(zero_grad=True)
                
                current_loss = loss.item()
            
            elif is_llc_noise:
                # LLCNoise: Two-Step (Compute LLC, Add Noise, Re-forward)
                
                # Step 1: Normal forward-backward to compute gradients
                optimizer.zero_grad(set_to_none=True)
                logits = model(inputs)
                loss = loss_fn(logits, targets)
                loss.backward()
                
                # Step 2: Use gradients to estimate LLC and inject noise
                hooks.pre_step_noise()
                
                # Step 3: Forward with noisy parameters
                optimizer.zero_grad(set_to_none=True)
                logits_noisy = model(inputs)
                loss_noisy = loss_fn(logits_noisy, targets)
                loss_noisy.backward()
                
                # Step 4: Restore parameters
                hooks.post_step_restore()
                
                # Step 5: Update
                optimizer.step()
                current_loss = loss.item()
            
            else:
                # Standard (L2_Only, CPR, LogitNorm, SD, DeMoss)
                optimizer.zero_grad(set_to_none=True)
                
                if hooks: hooks.pre_step_noise() # DeMoss Noise
                
                logits = model(inputs)
                loss = loss_fn(logits, targets)
                
                if hooks: loss += hooks.spectral_entropy_loss() # DeMoss Spectral
                
                loss.backward()
                
                if hooks: hooks.post_step_restore() # Restore params
                
                optimizer.step()
                current_loss = loss.item()
            
            steps += 1
            pbar.update(1)

            # --- Evaluation (Every 100 steps or epoch end) ---
            if steps % 200 == 0:
                model.eval()
                with torch.no_grad():
                    # Check Test Acc
                    correct = 0
                    total = 0
                    # Test on a subset to save time or full set
                    for t_inputs, t_targets in test_loader:
                        t_inputs, t_targets = t_inputs.to(device), t_targets.to(device)
                        outputs = model(t_inputs)
                        _, predicted = outputs.max(1)
                        total += t_targets.size(0)
                        correct += predicted.eq(t_targets).sum().item()
                    
                    test_acc = correct / total
                    
                    # Estimate Train Acc (using current batch for speed approximation)
                    _, tr_pred = logits.max(1)
                    train_acc = tr_pred.eq(targets).sum().item() / targets.size(0)
                
                metrics['steps'].append(steps)
                metrics['train_acc'].append(train_acc)
                metrics['test_acc'].append(test_acc)
                metrics['train_loss'].append(current_loss)
                
                pbar.set_postfix({'Test': f'{test_acc:.3f}', 'Train': f'{train_acc:.3f}'})
                
                model.train() # Switch back
            
            if steps >= args.budget:
                break
                
    pbar.close()
    return (method_name, metrics)

# =============================================================================
# 4. Plotting
# =============================================================================

def plot_all_results(results_map, save_dir, dataset_name):
    print(f"\n[Plotting] Generating comparison for {dataset_name}...")
    plt.style.use('seaborn-v0_8-paper')
    
    # Colors
    method_groups = {
        "Baseline": ["L2_Only", "SAM", "ASAM", "LogitNorm", "SpectralDecoupling"],
        "SOTA": ["SAMPa", "CPR"],
        "Ours": ["LLCNoise"]
    }
    
    cmap_base = plt.get_cmap("Blues")
    cmap_sota = plt.get_cmap("Greens")
    colors = {}
    for i, m in enumerate(method_groups["Baseline"]): colors[m] = cmap_base(0.3 + i * 0.12)
    for i, m in enumerate(method_groups["SOTA"]): colors[m] = cmap_sota(0.4 + i * 0.2)
    colors["LLCNoise"] = "red"

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    for name, data in results_map.items():
        steps = data['steps']
        acc = data['test_acc']
        
        # Smooth
        if len(acc) > 5:
            acc_smooth = np.convolve(acc, np.ones(3)/3, mode='same')
        else:
            acc_smooth = acc
            
        c = colors.get(name, 'gray')
        lw = 2.5 if "Ours" in name else 1.5
        alpha = 1.0 if "Ours" in name else 0.7
        
        ax.plot(steps, acc_smooth, label=name, color=c, linewidth=lw, alpha=alpha)
    
    ax.set_title(f"Grokking on Vision: {dataset_name.upper()} (ViT-Small)")
    ax.set_xlabel("Steps")
    ax.set_ylabel("Test Accuracy")
    ax.set_xscale("linear") # Vision training usually linear or epochs
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"vision_comparison_{dataset_name}.png"), dpi=200)
    print(f"✅ Plot saved to {os.path.join(save_dir, f'vision_comparison_{dataset_name}.png')}")

# =============================================================================
# 5. Main Entry
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="mnist", 
                        choices=["mnist", "cifar10", "omniglot", "tiny-imagenet"],
                        help="Dataset to train on")
    parser.add_argument("--budget", type=int, default=50000, help="Total training steps")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=5e-4, help="ViT learning rate usually smaller")
    parser.add_argument("--tf32", type=int, default=1)
    parser.add_argument("--data_root", type=str, default="./data", help="Root for dataset download")
    parser.add_argument("--results_base", type=str, default="./vision_results")
    args = parser.parse_args()

    # Create directories
    os.makedirs(args.data_root, exist_ok=True)
    save_dir = os.path.join(args.results_base, args.dataset)
    os.makedirs(save_dir, exist_ok=True)
    
    print("="*80)
    print(f" 🚀 Vision Grokking Experiment: {args.dataset.upper()}")
    print(f"    Model: ViT-Small (Patch=8, Dim=384)")
    print(f"    Steps: {args.budget}")
    print(f"    Methods: 8 Regularization Techniques (Ours: LLCNoise)")
    print("="*80)

    # 8 个方法（去掉 Ours_SSAM 和 DeMoss）
    method_list = [
        "L2_Only",                                         # Pure Baseline
        "SAM", "ASAM", "LogitNorm", "SpectralDecoupling", # Classic
        "SAMPa", "CPR", "LLCNoise"                        # SOTA & Ours
    ]
    
    tasks = []
    available_gpus = torch.cuda.device_count()
    if available_gpus == 0: raise RuntimeError("No CUDA GPUs found.")
    
    for i, method in enumerate(method_list):
        gpu_id = i % available_gpus
        tasks.append((method, gpu_id, args, save_dir))

    # Parallel Execution
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=min(len(tasks), available_gpus)) as pool:
        results = pool.map(train_worker, tasks)
    
    results_map = {res[0]: res[1] for res in results}
    
    # Plot & Save
    plot_all_results(results_map, save_dir, args.dataset)

    for name, data in results_map.items():
        csv_path = os.path.join(save_dir, f"{name}_metrics.csv")
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(data.keys())
            writer.writerows(zip(*data.values()))
    
    print("\n✅ All experiments finished.")

if __name__ == "__main__":
    main()