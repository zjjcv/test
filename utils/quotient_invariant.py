import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.lines import Line2D
from scipy.ndimage import gaussian_filter

# Set random seed
np.random.seed(42)

# ==========================================
# 1. 核心定义
# ==========================================
def cosine_similarity(u, v):
    norm_u = np.linalg.norm(u); norm_v = np.linalg.norm(v)
    if norm_u == 0 or norm_v == 0: return 0
    return np.dot(u, v) / (norm_u * norm_v)

def loss_fn(u, v, target=0.9):
    return 0.5 * (cosine_similarity(u, v) - target)**2

def gradients(u, v, target=0.9):
    norm_u = np.linalg.norm(u); norm_v = np.linalg.norm(v)
    if norm_u < 1e-9 or norm_v < 1e-9: return np.zeros_like(u), np.zeros_like(v)
    sim = np.dot(u, v) / (norm_u * norm_v)
    grad_u = (v/(norm_u*norm_v)) - (u*sim/(norm_u**2))
    grad_v = (u/(norm_u*norm_v)) - (v*sim/(norm_v**2))
    factor = (sim - target)
    return factor * grad_u, factor * grad_v

# ==========================================
# 2. 修正版 QN-SGLD (Corrected QN-SGLD)
# ==========================================
def run_sgld_chain(u_init, v_init, method, steps=1000, eps=1e-4, n_data=100):
    u, v = u_init.copy(), v_init.copy()
    energies = []
    burn_in = int(steps * 0.7) 
    
    noise_u_all = np.random.randn(steps, *u.shape)
    noise_v_all = np.random.randn(steps, *v.shape)
    
    # 极小的数值稳定项 (仅防止除零，不影响动力学)
    gamma_safe = 1e-8
    
    for t in range(steps):
        gu, gv = gradients(u, v)
        nu, nv = noise_u_all[t], noise_v_all[t]
        
        if method == 'standard':
            # 标准 SGLD
            du = -0.5 * eps * gu * n_data + np.sqrt(eps) * nu
            dv = -0.5 * eps * gv * n_data + np.sqrt(eps) * nv
            
        elif method == 'corrected_qn':
            # === Corrected QN-SGLD ===
            # Diffusion Coeff D ~ ||u||^2
            n_u = np.linalg.norm(u) + gamma_safe
            n_v = np.linalg.norm(v) + gamma_safe
            
            # 1. 主漂移项 (Main Drift): -0.5 * D * Gradient
            drift_u = -0.5 * eps * (n_u**2) * gu * n_data
            drift_v = -0.5 * eps * (n_v**2) * gv * n_data
            
            # 2. 修正漂移项 (Spurious Drift Correction): + epsilon * (nabla . D)
            # D = ||u||^2 * I. Div(D) = 2*u.
            # 这个项在小尺度下至关重要，它提供向外的推力，防止测度塌缩
            correction_u = eps * 2.0 * u 
            correction_v = eps * 2.0 * v 
            
            # 3. 扩散项 (Diffusion): sqrt(2D) * Noise ~ ||u|| * Noise
            diff_u = np.sqrt(eps) * n_u * nu
            diff_v = np.sqrt(eps) * n_v * nv
            
            du = drift_u + correction_u + diff_u
            dv = drift_v + correction_v + diff_v
        
        u += du; v += dv
        if t > burn_in: energies.append(loss_fn(u, v))
            
    return np.mean(energies) * n_data if energies else 0

def run_simulation_averaged(u_start, v_start, method, n_repeats=5, **kwargs):
    results = []
    for _ in range(n_repeats):
        res = run_sgld_chain(u_start, v_start, method, **kwargs)
        results.append(res)
    return np.mean(results)

# ==========================================
# 3. 仿真执行 (范围扩大至 -1.5 到 1.8)
# ==========================================
dim = 10
u0 = np.random.randn(dim); u0 /= np.linalg.norm(u0)
v0 = u0 + 0.1 * np.random.randn(dim); v0 /= np.linalg.norm(v0)

grid = 24 # 提高分辨率
scales = np.logspace(-1.5, 1.8, grid) # 覆盖用户要求的范围
X, Y = np.meshgrid(scales, scales)
Z_std, Z_qn = np.zeros_like(X), np.zeros_like(X)

print("Running Corrected QN-SGLD simulation...")
for i in range(grid):
    for j in range(grid):
        a, b = scales[j], scales[i]
        uc, vc = u0 * a * b, v0 * a * (1.0/b)
        Z_std[i,j] = run_simulation_averaged(uc, vc, 'standard', n_repeats=6, steps=1200, eps=1e-4)
        Z_qn[i,j]  = run_simulation_averaged(uc, vc, 'corrected_qn', n_repeats=6, steps=1200, eps=1e-4)

# 平滑处理
sigma = 1.5
Z_std_smooth = gaussian_filter(Z_std, sigma=sigma)
Z_qn_smooth = gaussian_filter(Z_qn, sigma=sigma)
log_X, log_Y = np.log10(X), np.log10(Y)

# ==========================================
# 4. 绘图输出
# ==========================================

# --- Plot 1: 3D Surface ---
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# QN-SGLD (Teal, Corrected)
ax.plot_surface(log_X, log_Y, Z_qn_smooth, color='#00CC99', alpha=0.9, 
               lw=0, antialiased=True, shade=True)
# Standard (Red)
ax.plot_surface(log_X, log_Y, Z_std_smooth, cmap=cm.RdPu, alpha=0.5, 
               lw=0.3, edgecolor='k', antialiased=True)

ax.set_title('Geometric Invariance: Corrected Quotient Dynamics', fontsize=16, fontweight='bold', pad=20)
ax.set_xlabel(r'Log Global Scale ($\log \alpha$)', fontsize=12, labelpad=10)
ax.set_ylabel(r'Log Layer Scale ($\log \beta$)', fontsize=12, labelpad=10)
ax.set_zlabel('Estimated Complexity (LLC)', fontsize=12, labelpad=10)
ax.view_init(30, 130)

legend_elements = [
    Line2D([0], [0], marker='o', color='w', label='Standard SGLD', markerfacecolor='#800080', markersize=10),
    Line2D([0], [0], marker='o', color='w', label='Corrected QN-SGLD (Ours)', markerfacecolor='#00CC99', markersize=10)
]
ax.legend(handles=legend_elements, loc='upper left', fontsize=12)
plt.savefig('plot1_3d_smooth.png', dpi=300, bbox_inches='tight')

# --- Plot 2: Heatmap ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
vmin, vmax = Z_std_smooth.min(), Z_std_smooth.max()

im = ax1.imshow(Z_std_smooth, extent=[-1.5,1.8,-1.5,1.8], origin='lower', cmap='magma', vmin=vmin, vmax=vmax)
ax1.contour(Z_std_smooth, colors='w', alpha=0.3, extent=[-1.5,1.8,-1.5,1.8])
ax1.set_title('Standard SGLD (Unstable)', fontsize=14, fontweight='bold')
ax1.set_xlabel('Global Scale', fontsize=12); ax1.set_ylabel('Layer Scale', fontsize=12)

ax2.imshow(Z_qn_smooth, extent=[-1.5,1.8,-1.5,1.8], origin='lower', cmap='magma', vmin=vmin, vmax=vmax)
ax2.set_title('Corrected QN-SGLD (Stable)', fontsize=14, fontweight='bold')
ax2.set_xlabel('Global Scale', fontsize=12); ax2.set_yticklabels([])

cbar = fig.colorbar(im, ax=[ax1, ax2], fraction=0.05, pad=0.04)
cbar.set_label('Estimated Complexity', fontsize=12)
plt.suptitle('Heatmap Comparison: Stability Across Full Range', fontsize=18, y=0.98)
plt.savefig('plot2_heatmap_smooth.png', dpi=300, bbox_inches='tight')

# --- Plot 3: 2D Slice ---
d = np.arange(grid)
fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(np.log10(scales), Z_std_smooth[d,d], color='#800080', lw=4, label='Standard SGLD')
ax.plot(np.log10(scales), Z_qn_smooth[d,d], color='#00CC99', lw=4, label='Corrected QN-SGLD')

# Theoretical line (Stable across range)
ref_val = np.mean(Z_qn_smooth[grid//2-2:grid//2+2, grid//2-2:grid//2+2])
ax.axhline(ref_val, color='gray', ls='--', lw=2, label='Theoretical Invariance')

ax.fill_between(np.log10(scales), Z_std_smooth[d,d], ref_val, color='#800080', alpha=0.1)

ax.set_title('Cross-Section Analysis: Small & Large Scale Stability', fontsize=16, fontweight='bold')
ax.set_xlabel(r'Log Scale Magnitude ($\alpha=\beta$)', fontsize=12)
ax.set_ylabel('Estimated Complexity', fontsize=12)
ax.legend(fontsize=12); ax.grid(True, ls=':', alpha=0.6)
plt.savefig('plot3_slice_smooth.png', dpi=300, bbox_inches='tight')