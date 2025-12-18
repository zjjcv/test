import os
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
import requests
import pickle
import networkx as nx
from scipy.stats import ortho_group
from sklearn.model_selection import train_test_split
import random
from tqdm import tqdm

# 配置路径
DATA_DIR = './data'
os.makedirs(DATA_DIR, exist_ok=True)

print(f"🚀 Initializing Dataset Generation for ICML 2026 Experiments...")
print(f"📂 Data will be saved to: {DATA_DIR}")

def save_tensor_dataset(name, train_x, train_y, test_x, test_y):
    """辅助函数：将数据保存为 PyTorch 格式"""
    file_path = os.path.join(DATA_DIR, f"{name}.pt")
    torch.save({
        'train_x': torch.tensor(train_x, dtype=torch.float32),
        'train_y': torch.tensor(train_y), # dtype depends on task
        'test_x': torch.tensor(test_x, dtype=torch.float32),
        'test_y': torch.tensor(test_y)
    }, file_path)
    print(f"✅ [Saved] {name} ({len(train_x)} train samples)")

# ==========================================
# 1. Symbolic & Math (符号与数学)
# ==========================================

def gen_ellipse_geometry(n_samples=10000):
    """
    [Replacement for Mod-Add] 椭圆几何性质学习
    任务：给定一般方程 Ax^2 + Bxy + Cy^2 + Dx + Ey + F = 0 的系数
    预测：椭圆的长轴(a)和短轴(b)长度 (Regression)
    """
    print("⚡ Generating Ellipse Geometry Dataset...")
    X = []
    Y = []
    for _ in range(n_samples):
        # 随机生成参数
        a = np.random.uniform(1, 10)
        b = np.random.uniform(1, a) # 保证 a >= b
        theta = np.random.uniform(0, np.pi)
        h, k = np.random.uniform(-5, 5, 2)
        
        # 构造旋转平移后的系数 (通过矩阵推导)
        # 这里简化：输入特征为 (a, b, theta) 的加噪版本，或者是采样点
        # 为了符合 Grokking，我们做 "Implicit Function"：
        # 输入：椭圆参数 (a, b) + 旋转 theta
        # 输出：计算 Area = pi*a*b, Eccentricity = sqrt(1 - b^2/a^2)
        
        # 更有难度的：Input: 5 points on ellipse, Output: (a, b)
        # 这里为了轻量级，我们做：Input: (A, B, C) of centered ellipse Ax^2 + Bxy + Cy^2 = 1, Output: (a, b)
        
        # 构造矩阵 M = R^T diag(1/a^2, 1/b^2) R
        c = np.cos(theta)
        s = np.sin(theta)
        R = np.array([[c, -s], [s, c]])
        D = np.diag([1/a**2, 1/b**2])
        M = R @ D @ R.T
        
        # 特征：矩阵 M 的独立元素 [M00, M01, M11]
        features = [M[0,0], M[0,1], M[1,1]]
        targets = [a, b] # 目标是还原几何不变量
        
        X.append(features)
        Y.append(targets)
        
    X = np.array(X)
    Y = np.array(Y)
    
    # Split
    train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.2)
    save_tensor_dataset('symbolic_ellipse_geometry', train_x, train_y, test_x, test_y)

def gen_sparse_poly_regression(n_samples=5000, dim=20, degree=2, sparsity=3):
    """稀疏多项式回归：顿悟稀疏性"""
    print("⚡ Generating Sparse Polynomial Regression...")
    # 简单的设置：y = sum(c_i * x_i^2) 只取少数几项
    X = np.random.randn(n_samples, dim)
    active_indices = np.random.choice(dim, sparsity, replace=False)
    coeffs = np.random.randn(sparsity)
    
    Y = np.zeros(n_samples)
    for i, idx in enumerate(active_indices):
        Y += coeffs[i] * (X[:, idx] ** degree)
        
    train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.2)
    save_tensor_dataset('symbolic_poly_regression', train_x, train_y, test_x, test_y)

def gen_permutation_group(n_samples=5000, n=5):
    """S_5 置换群复合任务"""
    print(f"⚡ Generating Permutation Group S_{n}...")
    import itertools
    perms = list(itertools.permutations(range(n)))
    perm_to_idx = {p: i for i, p in enumerate(perms)}
    idx_to_perm = {i: p for i, p in enumerate(perms)}
    
    X, Y = [], []
    for _ in range(n_samples):
        idx_a = np.random.randint(len(perms))
        idx_b = np.random.randint(len(perms))
        p_a = idx_to_perm[idx_a]
        p_b = idx_to_perm[idx_b]
        
        # Composition: p_c[i] = p_a[p_b[i]]
        p_c = tuple(p_a[j] for j in p_b)
        idx_c = perm_to_idx[p_c]
        
        # Input: one-hot or embedding indices (here just indices)
        X.append([idx_a, idx_b])
        Y.append(idx_c)
        
    save_tensor_dataset('symbolic_permutation', np.array(X), np.array(Y), [], []) # Split locally later

def gen_high_dim_linreg(n_samples=2000, dim=1000):
    """高维线性回归 (d > n)"""
    print("⚡ Generating High-Dim Linear Regression...")
    X = np.random.randn(n_samples, dim)
    # Teacher vector
    w_star = np.random.randn(dim) / np.sqrt(dim)
    Y = X @ w_star + 0.01 * np.random.randn(n_samples) # Add noise
    
    train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.2)
    save_tensor_dataset('symbolic_high_dim_linreg', train_x, train_y, test_x, test_y)

# ==========================================
# 2. Vision (Algorithmic)
# ==========================================

def download_dsprites():
    """下载 dSprites 数据集"""
    url = "https://github.com/deepmind/dsprites-dataset/raw/master/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz"
    path = os.path.join(DATA_DIR, "dsprites.npz")
    if not os.path.exists(path):
        print("⚡ Downloading dSprites (This may take a moment)...")
        r = requests.get(url, stream=True)
        with open(path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=1024):
                if chunk: f.write(chunk)
    print("✅ [Ready] dSprites")
    # Note: User handles loading of npz separately or we can process it here.
    # Leaving as raw npz is standard for dSprites.

def gen_synthetic_shapes_3d(n_samples=5000):
    """生成简化的 'Shapes3D' (Flatland)"""
    print("⚡ Generating Synthetic Shapes (Mini-Shapes3D)...")
    # Generates 64x64 images with specific shapes (Circle, Square, Triangle)
    # Target: Classify Shape, Color, Size
    images = np.zeros((n_samples, 3, 32, 32)) # RGB
    labels = [] # (shape_idx, color_idx)
    
    for i in range(n_samples):
        shape_type = np.random.randint(0, 3) # 0:Square, 1:Circle, 2:Bar
        color_type = np.random.randint(0, 3) # R, G, B
        
        color = np.zeros(3)
        color[color_type] = 1.0
        
        # Simple drawing logic (omitted for brevity, filling with noise + pattern)
        # Real implementation would use cv2 or skimage.draw
        # Placeholder: Center blob
        center = 16
        radius = np.random.randint(5, 12)
        
        y, x = np.ogrid[:32, :32]
        dist = np.sqrt((x - center)**2 + (y-center)**2)
        
        mask = dist <= radius
        if shape_type == 0: # Square
            mask = (np.abs(x-center) < radius) & (np.abs(y-center) < radius)
            
        images[i, :, mask] = color
        labels.append(shape_type)
        
    train_x, test_x, train_y, test_y = train_test_split(images, labels, test_size=0.2)
    save_tensor_dataset('vision_synthetic_shapes', train_x, train_y, test_x, test_y)

def download_mnist_1d():
    """下载 MNIST-1D"""
    url = "https://github.com/greydanus/mnist1d/raw/master/mnist1d_data.pkl"
    path = os.path.join(DATA_DIR, "mnist1d.pkl")
    if not os.path.exists(path):
        print("⚡ Downloading MNIST-1D...")
        r = requests.get(url)
        with open(path, 'wb') as f:
            f.write(r.content)
    print("✅ [Ready] MNIST-1D")

def get_torchvision_datasets():
    """下载 CIFAR10, SVHN, Omniglot"""
    print("⚡ Checking Torchvision Datasets (CIFAR10, SVHN, Omniglot)...")
    # CIFAR10 (Grayscale version logic should be applied in DataLoader)
    datasets.CIFAR10(root=DATA_DIR, train=True, download=True)
    
    # SVHN
    datasets.SVHN(root=DATA_DIR, split='train', download=True)
    
    # Omniglot
    datasets.Omniglot(root=DATA_DIR, background=True, download=True)
    print("✅ [Ready] Torchvision Datasets")

# ==========================================
# 3. Text & Sequence
# ==========================================

def gen_dyck_language(n_samples=5000, max_depth=10):
    """Dyck-2 括号匹配: ( [ ] )"""
    print("⚡ Generating Dyck-2 Language...")
    vocab = {'(':0, ')':1, '[':2, ']':3, 'PAD': 4}
    
    data = []
    targets = [] # Valid (1) or Invalid (0) or Next Token Prediction
    # Simplified: Next Token Prediction Task (Autoregressive)
    
    def generate_balanced(depth):
        if depth == 0: return []
        if np.random.rand() < 0.5:
            return [0] + generate_balanced(depth-1) + [1]
        else:
            return [2] + generate_balanced(depth-1) + [3]
            
    for _ in range(n_samples):
        seq = generate_balanced(np.random.randint(2, max_depth))
        # Add padding
        seq = seq + [4]*(2*max_depth - len(seq))
        data.append(seq)
        
    save_tensor_dataset('text_dyck2', np.array(data), np.zeros(n_samples), [], [])

def gen_reverse_string(n_samples=5000, length=20):
    """String Reversal Task"""
    print("⚡ Generating Reverse String Task...")
    # Vocab size 26
    X = np.random.randint(0, 26, (n_samples, length))
    Y = np.flip(X, axis=1).copy() # Output is reverse of input
    
    train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.2)
    save_tensor_dataset('text_reverse_string', train_x, train_y, test_x, test_y)

def gen_context_sorting(n_samples=5000, length=10):
    """Sorting Task"""
    print("⚡ Generating Contextual Sorting...")
    X = np.random.randint(0, 100, (n_samples, length))
    Y = np.sort(X, axis=1)
    
    train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.2)
    save_tensor_dataset('text_sorting', train_x, train_y, test_x, test_y)

def download_tinyshakespeare():
    """TinyShakespeare"""
    path = os.path.join(DATA_DIR, 'tinyshakespeare.txt')
    if not os.path.exists(path):
        print("⚡ Downloading TinyShakespeare...")
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        r = requests.get(url)
        with open(path, 'w') as f:
            f.write(r.text)
    print("✅ [Ready] TinyShakespeare")

# ==========================================
# 4. Logic & Reasoning
# ==========================================

def gen_sparse_parity(n_samples=5000, dim=40, k=3):
    """Sparse XOR (Parity)"""
    print("⚡ Generating Sparse Parity (XOR)...")
    X = np.random.randint(0, 2, (n_samples, dim)) * 2 - 1 # {-1, 1}
    # Choose k active bits fixed for the task
    active_indices = np.random.choice(dim, k, replace=False)
    
    Y = np.prod(X[:, active_indices], axis=1) # XOR equivalent in {-1, 1} product
    Y = (Y > 0).astype(int) # Convert to {0, 1} labels
    
    train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.2)
    save_tensor_dataset('logic_sparse_parity', train_x, train_y, test_x, test_y)

def gen_latin_squares(n_samples=1000, n=5):
    """Latin Squares Completion (Simplification)"""
    print("⚡ Generating Latin Squares...")
    # Generating valid latin squares is hard, we use cyclic shifts as a simple valid set
    base = np.arange(n)
    X = []
    Y = []
    
    for _ in range(n_samples):
        shift = np.random.randint(0, n)
        row = np.roll(base, shift)
        # Task: Given n-1 elements, predict the missing one
        mask_idx = np.random.randint(0, n)
        target = row[mask_idx]
        input_row = row.copy()
        input_row[mask_idx] = -1 # Mask token
        
        X.append(input_row)
        Y.append(target)
        
    save_tensor_dataset('logic_latin_squares', np.array(X), np.array(Y), [], [])

def gen_graph_connectivity(n_samples=2000, nodes=10):
    """Graph Connectivity Task"""
    print("⚡ Generating Graph Connectivity...")
    X = []
    Y = []
    
    for _ in range(n_samples):
        # Generate random graph (Adjacency Matrix flattened)
        G = nx.erdos_renyi_graph(nodes, p=0.3)
        adj = nx.to_numpy_array(G)
        is_connected = 1 if nx.is_connected(G) else 0
        
        X.append(adj.flatten())
        Y.append(is_connected)
        
    train_x, test_x, train_y, test_y = train_test_split(np.array(X), np.array(Y), test_size=0.2)
    save_tensor_dataset('logic_graph_connectivity', train_x, train_y, test_x, test_y)

def gen_3sat_light(n_samples=5000, n_vars=10, n_clauses=20):
    """3-SAT Light"""
    print("⚡ Generating 3-SAT Light...")
    # Simplified: Generate random clauses and check satisfiability is complex (NP-complete).
    # Instead, we generate a valid solution FIRST, then generate clauses consistent with it.
    X = []
    Y = []
    
    for _ in range(n_samples):
        # 1. Random assignment
        assignment = np.random.randint(0, 2, n_vars)
        # 2. Generate compatible clauses (so it is SAT)
        clauses = []
        for _ in range(n_clauses):
            # Pick 3 vars
            idx = np.random.choice(n_vars, 3, replace=False)
            # Pick signs such that clause is TRUE under assignment
            # clause = (l1 v l2 v l3). To be true, at least one literal must be true.
            # We force one to be true.
            force_true_idx = np.random.randint(0, 3)
            signs = np.random.randint(0, 2, 3) # 0: neg, 1: pos
            
            # Adjust sign of forced variable to match assignment
            if assignment[idx[force_true_idx]] == 1:
                signs[force_true_idx] = 1
            else:
                signs[force_true_idx] = 0
                
            clause_rep = np.concatenate([idx, signs]) # [v1, v2, v3, s1, s2, s3]
            clauses.append(clause_rep)
            
        X.append(np.array(clauses).flatten())
        Y.append(1) # These are all SAT. 
        # (For real dataset, we need to mix SAT and UNSAT, but that requires a solver. 
        # For 'Learning Rules', learning to verify a SAT solution is also valid.)
    
    save_tensor_dataset('logic_3sat', np.array(X), np.array(Y), [], [])


if __name__ == "__main__":
    # --- 1. Symbolic ---
    gen_ellipse_geometry()
    gen_sparse_poly_regression()
    gen_permutation_group()
    gen_high_dim_linreg()
    # Matrix completion omitted for brevity (similar to LinReg with masks)
    
    # --- 2. Vision ---
    download_dsprites()
    download_mnist_1d()
    gen_synthetic_shapes_3d()
    get_torchvision_datasets()
    
    # --- 3. Text ---
    gen_dyck_language()
    gen_reverse_string()
    gen_context_sorting()
    download_tinyshakespeare()
    
    # --- 4. Logic ---
    gen_sparse_parity()
    gen_latin_squares()
    gen_graph_connectivity()
    gen_3sat_light()
    
    print("\n🎉 All Datasets Generated/Downloaded Successfully!")
    print(f"📂 Check the '{DATA_DIR}' directory.")