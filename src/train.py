import os
import sys

# =========================================================================
# 0. Path Setup (让 root 成为 import 搜索根)
# =========================================================================
# train.py 在 root/src 下
current_dir = os.path.dirname(os.path.abspath(__file__))   # root/src
project_root = os.path.dirname(current_dir)                # root

# 用 insert(0) 保证优先级最高，避免被环境里同名包干扰
if project_root not in sys.path:
    sys.path.insert(0, project_root)


import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import time

import utils
print("utils imported from:", utils.__file__)


# --- 修改后的导入路径 ---
from model.models import get_model               # 对应 root/model/models.py
from utils.regularzation import get_regularizer_and_optimizer # 对应 root/utils/regularizers.py

# =========================================================================
# 1. Global Configuration & Dataset Metadata
# =========================================================================

# --- 修改后的数据路径：指向 root/data ---
DATA_DIR = os.path.join(project_root, 'data')
RESULTS_FILE = os.path.join(project_root, 'experiment_results.csv')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"📂 Data Directory: {DATA_DIR}")
print(f"📄 Results File: {RESULTS_FILE}")

# 实验超参数
MAX_STEPS = 20000 
BATCH_SIZE = 512
EVAL_INTERVAL = 100 

# 24个数据集配置
DATASET_CONFIG = {
    # --- Symbolic (MLP) ---
    'symbolic_ellipse_geometry': ('symbolic', (3,), 2),
    'symbolic_poly_regression':  ('symbolic', (20,), 1),
    'symbolic_permutation':      ('symbolic', (2,), 120),
    'symbolic_high_dim_linreg':  ('symbolic', (500,), 1),
    
    # --- Vision (ResNet) ---
    'dsprites':                  ('vision', (1, 64, 64), 3),
    'mnist1d':                   ('vision', (1, 40), 10),
    'vision_synthetic_shapes':   ('vision', (3, 32, 32), 3),
    'cifar10':                   ('vision', (3, 32, 32), 10),
    'svhn':                      ('vision', (3, 32, 32), 10),
    'omniglot':                  ('vision', (1, 105, 105), 1623),
    
    # --- Text (Causal Transformer) ---
    'text_dyck2':                ('text', (24,), 5),
    'text_reverse_string':       ('text', (20,), 26),
    'text_sorting':              ('text', (10,), 100),
    'tinyshakespeare':           ('text', (64,), 65),
    
    # --- Logic (Bidirectional Transformer) ---
    'logic_sparse_parity':       ('logic', (40,), 2),
    'logic_latin_squares':       ('logic', (6,), 6),
    'logic_graph_connectivity':  ('logic', (100,), 2),
    'logic_3sat':                ('logic', (39,), 2),
}

# 12种方法列表
METHODS = [
    'baseline', 'l2', 'logit_norm', 'spectral_decoupling', 'flooding', 
    'swa', 'ema',                                                      
    'sam', 'asam', 'gsam', 'saf',                                      
    'si_llc'                                                           
]

# =========================================================================
# 2. Data Loading Helper
# =========================================================================

def load_dataset(name, modality):
    # 1. 特殊处理 Torchvision
    if name in ['cifar10', 'svhn', 'omniglot']:
        from torchvision import datasets, transforms
        if name == 'cifar10':
            transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
            train_set = datasets.CIFAR10(root=DATA_DIR, train=True, transform=transform, download=False)
            test_set = datasets.CIFAR10(root=DATA_DIR, train=False, transform=transform, download=False)
        elif name == 'svhn':
            transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
            train_set = datasets.SVHN(root=DATA_DIR, split='train', transform=transform, download=False)
            test_set = datasets.SVHN(root=DATA_DIR, split='test', transform=transform, download=False)
        elif name == 'omniglot':
            transform = transforms.Compose([transforms.Resize((28,28)), transforms.ToTensor()])
            train_set = datasets.Omniglot(root=DATA_DIR, background=True, transform=transform, download=False)
            test_set = datasets.Omniglot(root=DATA_DIR, background=False, transform=transform, download=False)
            
    # 2. dSprites (如果之前没转 .pt，这里简单处理，或者假设已转)
    elif name == 'dsprites':
        try:
            data = torch.load(os.path.join(DATA_DIR, f'{name}.pt'))
            train_set = TensorDataset(data['train_x'], data['train_y'])
            test_set = TensorDataset(data['test_x'], data['test_y'])
        except:
            print(f"⚠️ Warning: {name}.pt not found in {DATA_DIR}")
            return None, None
            
    # 3. 标准 .pt 文件
    else:
        try:
            data = torch.load(os.path.join(DATA_DIR, f'{name}.pt'))
            train_set = TensorDataset(data['train_x'], data['train_y'])
            test_set = TensorDataset(data['test_x'], data['test_y'])
        except FileNotFoundError:
            print(f"❌ Dataset {name}.pt not found in {DATA_DIR}")
            return None, None

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    
    return train_loader, test_loader

# =========================================================================
# 3. Training Core (Same Logic as before)
# =========================================================================

def train_experiment(dataset_name, method_name):
    print(f"\n🧪 Experiment: Task=[{dataset_name}] | Method=[{method_name}]")
    
    if dataset_name not in DATASET_CONFIG:
        return None
    modality, input_shape, output_dim = DATASET_CONFIG[dataset_name]
    
    train_loader, test_loader = load_dataset(dataset_name, modality)
    if train_loader is None: return None
    
    model = get_model(modality, input_shape, output_dim, DEVICE)
    
    optimizer, loss_hook, model_wrapper = get_regularizer_and_optimizer(
        method_name, model, base_lr=1e-3, weight_decay=1e-4
    )
    
    criterion = nn.CrossEntropyLoss()
    if modality == 'symbolic' and 'regression' in dataset_name: # Handle regression
        criterion = nn.MSELoss()
    elif 'geometry' in dataset_name: # Ellipse geometry (regression)
        criterion = nn.MSELoss()

    t_mem, t_gen = None, None
    final_acc = 0.0
    
    train_iter = iter(train_loader)
    
    pbar = tqdm(range(MAX_STEPS), desc="Training", leave=False)
    
    for step in pbar:
        try:
            inputs, targets = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            inputs, targets = next(train_iter)
            
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        
        # --- Training Logic ---
        if method_name not in ['sam', 'asam', 'gsam', 'saf']:
            optimizer.zero_grad()
            logits = model(inputs)
            
            if isinstance(criterion, nn.MSELoss):
                 loss = criterion(logits, targets.float())
            else:
                 loss = criterion(logits, targets.long())
                 
            loss = loss_hook(model, loss, logits, targets)
            loss.backward()
            
            # SI-LLC Adaptive Decay
            if method_name == 'si_llc':
                with torch.no_grad():
                    grad_norm = sum(p.grad.norm(2)**2 for p in model.parameters() if p.grad is not None)
                    lambda_proxy = grad_norm / (loss.item() + 1e-6)
                    beta = loss_hook.beta
                    for p in model.parameters():
                        if p.grad is not None:
                            adaptive_decay = 2 * beta * lambda_proxy * p.data
                            p.grad.add_(adaptive_decay)
            
            optimizer.step()
        
        else:
            # SAM steps
            logits = model(inputs)
            if isinstance(criterion, nn.MSELoss): loss = criterion(logits, targets.float())
            else: loss = criterion(logits, targets.long())
            loss.backward()
            optimizer.first_step(zero_grad=True)
            
            logits_2 = model(inputs)
            if isinstance(criterion, nn.MSELoss): loss_2 = criterion(logits_2, targets.float())
            else: loss_2 = criterion(logits_2, targets.long())
            
            if method_name == 'saf': loss_2 = loss_hook(model, loss_2, logits_2, targets)
                
            loss_2.backward()
            optimizer.second_step(zero_grad=True)

        if model_wrapper: model_wrapper.update()

        # --- Evaluation ---
        if step % EVAL_INTERVAL == 0:
            eval_model = model_wrapper.get_averaged_model() if model_wrapper else model
            eval_model.eval()
            
            def evaluate(loader):
                correct, total = 0, 0
                with torch.no_grad():
                    for x, y in loader:
                        x, y = x.to(DEVICE), y.to(DEVICE)
                        out = eval_model(x)
                        if isinstance(criterion, nn.MSELoss):
                            # Regression accuracy: Error < 0.05
                            err = (out - y.float()).pow(2).sum(dim=1)
                            correct += (err < 0.05).sum().item()
                        else:
                            pred = out.argmax(dim=1)
                            correct += (pred == y).sum().item()
                        total += y.size(0)
                return correct / total if total > 0 else 0
            
            train_acc = evaluate(train_loader)
            test_acc = evaluate(test_loader)
            
            if train_acc >= 0.95 and t_mem is None: t_mem = step
            if test_acc >= 0.95 and t_gen is None: t_gen = step
            
            final_acc = test_acc
            eval_model.train()
            
            pbar.set_postfix({'T_Acc': f'{train_acc:.2f}', 'V_Acc': f'{test_acc:.2f}', 'Grok': t_gen is not None})
            
            if test_acc > 0.99 and step > 1000: break

    t_mem_val = t_mem if t_mem else MAX_STEPS
    t_gen_val = t_gen if t_gen else MAX_STEPS
    lag = t_gen_val - t_mem_val
    if lag < 1: lag = 1
    
    if t_gen is None: ges = 0.0
    else: ges = (final_acc * 10) / np.log10(lag + 10)
        
    return {
        'Dataset': dataset_name,
        'Method': method_name,
        'Final_Test_Acc': final_acc,
        'T_mem': t_mem_val,
        'T_gen': t_gen_val,
        'GES': ges
    }

if __name__ == '__main__':
    if not os.path.exists(RESULTS_FILE):
        df = pd.DataFrame(columns=['Dataset', 'Method', 'Final_Test_Acc', 'T_mem', 'T_gen', 'GES'])
        df.to_csv(RESULTS_FILE, index=False)
        print(f"📄 Created {RESULTS_FILE}")
    
    for dataset in DATASET_CONFIG.keys():
        for method in METHODS:
            try:
                # Simple Resume Check
                existing = pd.read_csv(RESULTS_FILE)
                if ((existing['Dataset'] == dataset) & (existing['Method'] == method)).any():
                    print(f"⏩ Skipping {dataset} + {method} (Done)")
                    continue
                
                res = train_experiment(dataset, method)
                if res:
                    pd.DataFrame([res]).to_csv(RESULTS_FILE, mode='a', header=False, index=False)
                    print(f"✅ Saved: GES={res['GES']:.4f}")
                torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"❌ Error: {e}")
                continue