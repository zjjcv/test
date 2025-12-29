import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from tqdm import tqdm

# =========================================================================
# 0. 配置与路径
# =========================================================================
PROJECT_ROOT = "/data/zjj/test"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 权重路径配置
BASE_CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, 'results', 'gpt2-large', 'checkpoints', 'x_plus_y')
PLOT_DIR = os.path.join(PROJECT_ROOT, 'results', 'analysis_plots', 'embeddings')
os.makedirs(PLOT_DIR, exist_ok=True)

STEPS = [100, 1000, 10000, 100000]
WD_SETTINGS = ['wd_0.0', 'wd_1.0']
P = 97  # 模数

# 绘图风格
plt.style.use('seaborn-v0_8-paper')
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 10,
    'axes.labelsize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'figure.titlesize': 14
})

# =========================================================================
# 1. 模型定义 (GPT-2 Large Custom Architecture)
# =========================================================================

class GPT2Decoder(nn.Module):
    def __init__(self, dim=1280, num_layers=36, num_heads=20, num_tokens=99, seq_len=5):
        super().__init__()
        # 仅定义我们需要提取权重的部分结构，无需完整的 forward 逻辑
        self.token_embeddings = nn.Embedding(num_tokens, dim)
        self.position_embeddings = nn.Embedding(seq_len, dim)
        
        # 我们需要访问 Block 0 的 Attention 权重
        self.blocks = nn.ModuleList([
            nn.ModuleDict({
                'attn': nn.Linear(dim, 3 * dim), # c_attn
                'proj': nn.Linear(dim, dim)      # c_proj
            }) for _ in range(num_layers)
        ])
        
        self.ln_f = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_tokens, bias=False)

# =========================================================================
# 2. 核心逻辑：权重提取与降维
# =========================================================================

def load_checkpoint_weights(wd, step):
    path = os.path.join(BASE_CHECKPOINT_DIR, wd, f"seed42_step{step}.pt")
    if not os.path.exists(path):
        print(f"[WARN] Checkpoint not found: {path}")
        return None
    
    ckpt = torch.load(path, map_location='cpu')
    sd = ckpt['model_state_dict']
    
    # 提取关键权重矩阵
    weights = {}
    
    # 1. Input Embeddings [99, 1280]
    weights['wte'] = sd['token_embeddings.weight'].float().numpy()
    
    # 2. Output Embeddings (LM Head) [99, 1280]
    # 注意：如果 head 权重是独立的，取 head.weight；如果是 tied，取 wte
    if 'head.weight' in sd:
        weights['lm_head'] = sd['head.weight'].float().numpy()
    else:
        weights['lm_head'] = weights['wte'] # Weight Tying
        
    # 3. Layer 0 Attention Key Projection
    # 这是一个 [1280, 3840] 的矩阵。直接降维它没有物理意义（那是神经元维度）。
    # 为了展示 Grokking，我们将 Input Embedding 投影过 Layer 0 的 Key 矩阵。
    # 这代表了 "模型第一层是如何 '看' 这些 Token 的"。
    # c_attn weight shape is [out, in] in PyTorch Linear, but [in, out] in GPT2 Conv1D usually.
    # Check shape:
    c_attn_w = sd['blocks.0.attn.c_attn.weight'] # Linear: [3*1280, 1280]
    
    # 拆分 Q, K, V
    dim = 1280
    # Linear layer weight is (out, in), so (3840, 1280)
    # We want K: slice the middle part
    W_K = c_attn_w[dim:2*dim, :].t() # [1280, 1280]
    
    # 计算 Projected Embeddings: E @ W_K
    # 这展示了 Token 在 Attention 空间的几何结构
    wte_tensor = sd['token_embeddings.weight'].float()
    attn_proj = torch.matmul(wte_tensor, W_K).numpy()
    weights['attn_L0_K'] = attn_proj
    
    return weights

def compute_projections(matrix, method='pca'):
    # 只取前 97 个 Token (0-96 是数字)，忽略 OP 和 EQ
    valid_data = matrix[:P, :] 
    
    if method == 'pca':
        reducer = PCA(n_components=2)
        proj = reducer.fit_transform(valid_data)
        # 归一化以便绘图
        proj = (proj - proj.mean(0)) / (proj.std(0) + 1e-6)
        return proj
    
    elif method == 'tsne':
        # Perplexity 设为 30 或更小（因为只有 97 个点）
        # FIX: Removed n_iter=1000 since it is default and caused error
        reducer = TSNE(n_components=2, perplexity=20, random_state=42, init='pca', learning_rate='auto')
        proj = reducer.fit_transform(valid_data)
        # 归一化
        proj = (proj - proj.mean(0)) / (proj.std(0) + 1e-6)
        return proj

# =========================================================================
# 3. 绘图主程序
# =========================================================================

def main():
    print("🚀 Starting PCA & t-SNE Analysis for Weights...")
    
    components = ['wte', 'lm_head', 'attn_L0_K']
    titles = ['Input Embeddings ($W_E$)', 'Output Embeddings ($W_U$)', 'Layer 0 Key Proj ($W_E W_K^0$)']
    
    methods = ['pca', 'tsne']
    
    # 我们为每个 Component + Method 组合生成一张大图
    # 图结构：2行 (WD=0, WD=1) x 4列 (Steps)
    
    for comp_key, comp_title in zip(components, titles):
        for method in methods:
            print(f"👉 Processing {comp_title} - {method.upper()}...")
            
            fig, axes = plt.subplots(2, 4, figsize=(20, 10), dpi=150)
            plt.subplots_adjust(wspace=0.2, hspace=0.3)
            
            # 颜色映射：根据数字的值 (0-96) 着色，红->紫
            colors = np.arange(P)
            
            for row, wd in enumerate(WD_SETTINGS):
                for col, step in enumerate(STEPS):
                    ax = axes[row, col]
                    
                    weights = load_checkpoint_weights(wd, step)
                    if weights is None:
                        ax.text(0.5, 0.5, "Missing", ha='center')
                        continue
                        
                    data = weights[comp_key]
                    proj = compute_projections(data, method=method)
                    
                    # 散点图
                    scatter = ax.scatter(
                        proj[:, 0], proj[:, 1], 
                        c=colors, 
                        cmap='hsv', 
                        s=40, 
                        alpha=0.8, 
                        edgecolors='grey', 
                        linewidth=0.5
                    )
                    
                    # 连线：如果是 Grokking，数字应该形成闭环。
                    # 画一条淡淡的线连接 0->1->...->96->0 辅助观察拓扑结构
                    ax.plot(
                        np.append(proj[:, 0], proj[0, 0]), 
                        np.append(proj[:, 1], proj[0, 1]), 
                        c='gray', alpha=0.3, linestyle='--'
                    )
                    
                    ax.set_title(f"Step {step}", fontsize=12, fontweight='bold')
                    ax.set_xticks([])
                    ax.set_yticks([])
                    
                    # 左侧标注 WD
                    if col == 0:
                        wd_label = "No Reg (WD=0)" if "0.0" in wd else "Reg (WD=1)"
                        ax.set_ylabel(wd_label, fontsize=14, fontweight='bold')

            # 添加 Colorbar
            cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
            cbar = fig.colorbar(scatter, cax=cbar_ax)
            cbar.set_label('Token Value (0-96)', fontsize=12)
            
            fig.suptitle(f"{comp_title} - {method.upper()} Projection\n(Evolution of Representation Geometry)", fontsize=16, y=0.95)
            
            save_name = f"{comp_key}_{method}.png"
            save_path = os.path.join(PLOT_DIR, save_name)
            plt.savefig(save_path, bbox_inches='tight')
            print(f"   Saved to {save_path}")
            plt.close()

    print("\n✅ All projections completed.")

if __name__ == "__main__":
    main()