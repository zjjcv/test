import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.colors import LightSource
from matplotlib.collections import PolyCollection
from scipy import interpolate

# 配置
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
print(f"Using device: {DEVICE}")

# ==========================================
# 1. 定义模型 (Invariant Transformer)
# ==========================================

class NormalizedLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        
    def forward(self, x):
        w_norm = F.normalize(self.weight, p=2, dim=1)
        return F.linear(x, w_norm)

class MLP(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.up = nn.Linear(dim, mult * dim, bias=False)
        self.down = nn.Linear(mult * dim, dim, bias=False)
        self.act = nn.ReLU()

    def forward(self, x):
        return self.down(self.act(self.up(x)))

class Transformer(nn.Module):
    def __init__(self, vocab_size=64, dim=32, n_layers=2, n_head=4, seq_len=16):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, dim)
        self.pos_emb = nn.Embedding(seq_len, dim)
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'attn': nn.MultiheadAttention(dim, n_head, batch_first=True),
                'ln1': nn.LayerNorm(dim),
                'mlp': MLP(dim),
                'ln2': nn.LayerNorm(dim)
            }) for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(dim)
        self.head = NormalizedLinear(dim, vocab_size)
        self.pos_idxs = torch.arange(seq_len).to(DEVICE)

    def forward(self, x):
        h = self.tok_emb(x) + self.pos_emb(self.pos_idxs[:x.size(1)])
        for layer in self.layers:
            normed = layer['ln1'](h)
            attn, _ = layer['attn'](normed, normed, normed)
            h = h + attn
            normed = layer['ln2'](h)
            h = h + layer['mlp'](normed)
        h = self.ln_f(h)
        return self.head(h)

# ==========================================
# 2. 几何变换与投影
# ==========================================

def apply_transform(model, alpha_global, beta_layerwise):
    with torch.no_grad():
        for p in model.parameters():
            p.mul_(alpha_global)
        for layer in model.layers:
            mlp = layer['mlp']
            mlp.up.weight.mul_(beta_layerwise)
            mlp.down.weight.div_(beta_layerwise)

CANONICAL_NORM = 20.0

def canonicalize_model(model):
    with torch.no_grad():
        for layer in model.layers:
            mlp = layer['mlp']
            w_u, w_d = mlp.up.weight, mlp.down.weight
            nu, nd = w_u.norm(), w_d.norm()
            target = (nu * nd).sqrt() + 1e-8
            w_u.mul_(target / (nu + 1e-8))
            w_d.mul_(target / (nd + 1e-8))
        
        curr_norm = sum((p**2).sum() for p in model.parameters()).sqrt()
        scale = CANONICAL_NORM / (curr_norm + 1e-8)
        for p in model.parameters():
            p.mul_(scale)

def project_tangent(model, vecs_dict):
    with torch.no_grad():
        for layer in model.layers:
            w_u, w_d = layer['mlp'].up.weight, layer['mlp'].down.weight
            v_u, v_d = vecs_dict[w_u], vecs_dict[w_d]
            dot = (v_u * w_u).sum() - (v_d * w_d).sum()
            norm = (w_u**2).sum() + (w_d**2).sum() + 1e-8
            alpha = dot / norm
            v_u.sub_(alpha * w_u)
            v_d.add_(alpha * w_d)

        dot_vt = sum((vecs_dict[p] * p).sum() for p in model.parameters())
        dot_tt = sum((p**2).sum() for p in model.parameters()) + 1e-8
        scale = dot_vt / dot_tt
        for p in model.parameters():
            vecs_dict[p].sub_(scale * p)

# ==========================================
# 3. LLC 估计器
# ==========================================

def estimate_llc(model, data, steps=100, lr=5e-5, beta=100.0, mode='euclidean'):
    model_run = copy.deepcopy(model)
    model_run.train()
    
    if mode == 'invariant':
        canonicalize_model(model_run)

    theta0 = {p: p.detach().clone() for p in model_run.parameters()}
    x, y = data
    losses = []
    
    for _ in range(steps):
        model_run.zero_grad()
        out = model_run(x).mean(dim=1)
        loss = F.cross_entropy(out, y)
        loss.backward()
        
        loss_val = loss.item()
        if math.isnan(loss_val): loss_val = 100.0
        losses.append(loss_val)
        
        with torch.no_grad():
            noise_std = math.sqrt(2 * lr / beta)
            grads = {p: p.grad for p in model_run.parameters()}
            noise = {p: torch.randn_like(p) * noise_std for p in model_run.parameters()}
            deltas = {p: (p - theta0[p]) for p in model_run.parameters()}
            
            if mode == 'invariant':
                project_tangent(model_run, grads)
                project_tangent(model_run, noise)
                project_tangent(model_run, deltas)
            
            for p in model_run.parameters():
                update = -lr * (grads[p] + 10.0 * deltas[p]) + noise[p]
                p.add_(update)
            
            if mode == 'invariant':
                canonicalize_model(model_run)

    return beta * len(x) * (np.mean(losses) - losses[0])

# ==========================================
# 4. 绘图辅助函数：Ribbon Plot
# ==========================================

def plot_ribbons(ax, X_grid, Y_grid, Z_data, axis_view='x', cmap=cm.coolwarm, alpha=0.8, z_offset=0):
    """
    绘制切片带状图 (Ribbon Plot)
    axis_view='x': 沿着 X 轴画线 (固定 Y)
    axis_view='y': 沿着 Y 轴画线 (固定 X)
    """
    verts = []
    colors = []
    
    # 决定切片的方向
    if axis_view == 'x':
        # 沿着 X 轴变化，固定 Y（行）
        rows, cols = Z_data.shape
        # 只选取部分切片，避免太密集
        indices = np.linspace(0, rows-1, 6, dtype=int)
        for i in indices:
            y_val = Y_grid[i, 0]
            xs = X_grid[i, :]
            zs = Z_data[i, :]
            
            # 构建多边形顶点：上面是曲线，下面是底座(z_offset)
            # 这样画出来是有面积的“墙”，不仅仅是线
            verts.append(list(zip(xs, [y_val]*len(xs), zs)))
    else:
        # 沿着 Y 轴变化，固定 X（列）
        rows, cols = Z_data.shape
        indices = np.linspace(0, cols-1, 6, dtype=int)
        for j in indices:
            x_val = X_grid[0, j]
            ys = Y_grid[:, j]
            zs = Z_data[:, j]
            verts.append(list(zip([x_val]*len(ys), ys, zs)))

    # 使用 Poly3DCollection 绘制
    # 这里我们简化一下，直接用 plot 画线，效果更清晰，或者 fill_between 的 3D 版
    # 为了美观，我们直接画 3D 线条 + 投影
    
    if axis_view == 'x':
        indices = np.linspace(0, Z_data.shape[0]-1, 8, dtype=int)
        for i in indices:
            ax.plot(X_grid[i,:], Y_grid[i,:], Z_data[i,:], color=cmap(i/len(indices)), linewidth=2, alpha=alpha)
            # 加一点投影阴影
            ax.plot(X_grid[i,:], Y_grid[i,:], [z_offset]*len(X_grid[i,:]), color='gray', linewidth=1, alpha=0.2)
    else:
        indices = np.linspace(0, Z_data.shape[1]-1, 8, dtype=int)
        for j in indices:
            ax.plot(X_grid[:,j], Y_grid[:,j], Z_data[:,j], color=cmap(j/len(indices)), linewidth=2, alpha=alpha)
            ax.plot(X_grid[:,j], Y_grid[:,j], [z_offset]*len(Y_grid[:,j]), color='gray', linewidth=1, alpha=0.2)

# ==========================================
# 5. 主实验
# ==========================================

def run_experiment_and_plot_analytical():
    print("1. Setup & Pretraining...")
    B, V = 128, 64
    x = torch.randint(0, V, (B, 16)).to(DEVICE)
    y = torch.randint(0, V, (B,)).to(DEVICE)
    model = Transformer(vocab_size=V).to(DEVICE)
    
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for i in range(150):
        opt.zero_grad()
        loss = F.cross_entropy(model(x).mean(dim=1), y)
        loss.backward()
        opt.step()
    
    # 扫描范围
    alphas = np.logspace(-1, 1, 8) 
    betas = np.logspace(-1, 1, 8)
    X_raw, Y_raw = np.meshgrid(np.log10(alphas), np.log10(betas))
    Z_euc_raw = np.zeros_like(X_raw)
    Z_inv_raw = np.zeros_like(X_raw)
    
    print("2. Scanning Manifold...")
    total = len(alphas) * len(betas)
    count = 0
    for i in range(len(alphas)):
        for j in range(len(betas)):
            m_temp = copy.deepcopy(model)
            apply_transform(m_temp, alphas[i], betas[j])
            
            Z_euc_raw[j, i] = estimate_llc(m_temp, (x, y), mode='euclidean')
            Z_inv_raw[j, i] = estimate_llc(m_temp, (x, y), mode='invariant')
            
            count += 1
            if count % 10 == 0: print(f"   Progress: {count}/{total}")

    print("3. Interpolating...")
    res = 60
    x_fine = np.linspace(X_raw.min(), X_raw.max(), res)
    y_fine = np.linspace(Y_raw.min(), Y_raw.max(), res)
    X_fine, Y_fine = np.meshgrid(x_fine, y_fine)
    
    interp_euc = interpolate.RectBivariateSpline(Y_raw[:, 0], X_raw[0, :], Z_euc_raw)
    interp_inv = interpolate.RectBivariateSpline(Y_raw[:, 0], X_raw[0, :], Z_inv_raw)
    
    Z_euc_fine = interp_euc(y_fine, x_fine)
    Z_inv_fine = interp_inv(y_fine, x_fine)
    
    # 底部偏移量
    z_min = min(Z_euc_fine.min(), Z_inv_fine.min())
    z_max = max(Z_euc_fine.max(), Z_inv_fine.max())
    offset = z_min - (z_max - z_min) * 0.1

    print("4. Plotting 3 Analytical Views...")
    fig = plt.figure(figsize=(20, 6), dpi=150)
    
    # --- 子图 1: Global Scaling Analysis (Ribbon Plot along X) ---
    ax1 = fig.add_subplot(1, 3, 1, projection='3d')
    ax1.set_title("A. Global Scaling Robustness\n(Sensitivity to $\\alpha$)", fontsize=13, fontweight='bold', pad=10)
    
    # 绘制 Euclidean 的带子 (红色系) - 沿着 X 轴 (Alpha)
    plot_ribbons(ax1, X_fine, Y_fine, Z_euc_fine, axis_view='x', cmap=cm.Reds, alpha=0.9, z_offset=offset)
    # 绘制 Invariant 的带子 (绿色系) - 沿着 X 轴
    plot_ribbons(ax1, X_fine, Y_fine, Z_inv_fine, axis_view='x', cmap=cm.Greens, alpha=0.9, z_offset=offset)
    
    ax1.set_xlabel(r'Global Scale ($\log_{10} \alpha$)', fontsize=10, fontweight='bold')
    ax1.set_ylabel(r'Layerwise Skew ($\log_{10} \beta$)', fontsize=10)
    ax1.set_zlabel('Complexity', fontsize=10)
    ax1.set_zlim(offset, z_max)
    ax1.view_init(elev=20, azim=-75) # 侧视，强调 X 轴变化
    
    # --- 子图 2: Layerwise Symmetry Analysis (Ribbon Plot along Y) ---
    ax2 = fig.add_subplot(1, 3, 2, projection='3d')
    ax2.set_title("B. Layerwise Symmetry Robustness\n(Sensitivity to $\\beta$)", fontsize=13, fontweight='bold', pad=10)
    
    # 绘制 Euclidean 的带子 (红色系) - 沿着 Y 轴 (Beta)
    plot_ribbons(ax2, X_fine, Y_fine, Z_euc_fine, axis_view='y', cmap=cm.Reds, alpha=0.9, z_offset=offset)
    # 绘制 Invariant 的带子 (绿色系) - 沿着 Y 轴
    plot_ribbons(ax2, X_fine, Y_fine, Z_inv_fine, axis_view='y', cmap=cm.Greens, alpha=0.9, z_offset=offset)
    
    ax2.set_xlabel(r'Global Scale ($\log_{10} \alpha$)', fontsize=10)
    ax2.set_ylabel(r'Layerwise Skew ($\log_{10} \beta$)', fontsize=10, fontweight='bold')
    ax2.set_zlim(offset, z_max)
    ax2.view_init(elev=20, azim=-15) # 前视，强调 Y 轴变化
    
    # --- 子图 3: Holistic Manifold Topology (Smooth Surface) ---
    ax3 = fig.add_subplot(1, 3, 3, projection='3d')
    ax3.set_title("C. Holistic Manifold Topology\n(Invariant Plane vs Euclidean Valley)", fontsize=13, fontweight='bold', pad=10)
    
    ls = LightSource(azdeg=315, altdeg=45)
    rgb_euc = ls.shade(Z_euc_fine, cmap=cm.Spectral_r, vert_exag=0.1, blend_mode='soft')
    
    # Euclidean Surface
    ax3.plot_surface(X_fine, Y_fine, Z_euc_fine, facecolors=rgb_euc, 
                     alpha=0.5, linewidth=0, antialiased=True, rstride=2, cstride=2)
    # Invariant Surface (Flat Plane)
    ax3.plot_surface(X_fine, Y_fine, Z_inv_fine, color='#00FA9A', 
                     alpha=0.8, linewidth=0, antialiased=True, shade=True, rstride=2, cstride=2)
    
    # Contours
    ax3.contourf(X_fine, Y_fine, Z_euc_fine, zdir='z', offset=offset, cmap=cm.Spectral_r, alpha=0.3)
    ax3.contour(X_fine, Y_fine, Z_inv_fine, zdir='z', offset=offset, colors='#00FA9A', linewidths=2)
    
    ax3.set_xlabel(r'$\log_{10} \alpha$', fontsize=10)
    ax3.set_ylabel(r'$\log_{10} \beta$', fontsize=10)
    ax3.set_zlim(offset, z_max)
    ax3.view_init(elev=30, azim=-60) # 标准透视

    # 去除背景和网格，统一风格
    for ax in [ax1, ax2, ax3]:
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.grid(False)
        ax.dist = 9.0 # 拉近一点

    # Global Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='#D53E4F', lw=4, label='Euclidean SGLD (Unstable)'),
        Line2D([0], [0], color='#00FA9A', lw=4, label='Quotient SGLD (Stable)'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=2, fontsize=14, frameon=False, bbox_to_anchor=(0.5, 0.02))

    plt.tight_layout()
    # 留出底部给图例
    plt.subplots_adjust(bottom=0.15)
    
    plt.savefig('manifold_analytical_3views.png', dpi=300, bbox_inches='tight', pad_inches=0.1)
    print("Done! Saved 'manifold_analytical_3views.png'")

if __name__ == "__main__":
    run_experiment_and_plot_analytical()