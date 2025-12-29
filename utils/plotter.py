import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

# 路径配置
RESULTS_DIR = '/data/zjj/test/Results/omnigrok_repro'
SAVE_DIR = os.path.join(RESULTS_DIR, 'plots')
os.makedirs(SAVE_DIR, exist_ok=True)

# 风格设置
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "font.size": 12,
    "figure.dpi": 300,
    "axes.grid": True,
    "grid.alpha": 0.3
})

def plot_3d_manifold():
    csv_path = os.path.join(RESULTS_DIR, 'summary.csv')
    if not os.path.exists(csv_path):
        print("未找到数据文件 summary.csv")
        return

    df = pd.read_csv(csv_path)
    
    # 数据预处理：
    # 1. 对 t_grok 进行 log 处理以便绘图 (因为 Grokking 时间跨度很大)
    # 2. 填充 NaN (未泛化) 为最大步数或特定值
    max_step = df['t_grok'].max()
    df['log_t_grok'] = np.log10(df['t_grok'].fillna(max_step * 1.5))

    # 创建网格
    pivot = df.pivot_table(index='alpha', columns='data_size', values='log_t_grok', aggfunc='mean')
    
    X_vals = pivot.columns.values # Data Size
    Y_vals = pivot.index.values   # Alpha
    X, Y = np.meshgrid(X_vals, Y_vals)
    Z = pivot.values

    # ================= 绘图 =================
    fig = plt.figure(figsize=(14, 8))
    
    # --- 子图 1: 3D 流形热力图 (Manifold) ---
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    
    # Log scale X 轴 (Data Size) 通常也是指数级变化的
    # 但为了 plot_surface 正常显示，我们用 log10(X) 作为坐标
    X_log = np.log10(X)
    
    surf = ax.plot_surface(X_log, Y, Z, cmap=cm.Spectral_r,
                           linewidth=0.2, antialiased=True, alpha=0.9, edgecolor='k')

    ax.set_xlabel('Log10(Data Size)')
    ax.set_ylabel('Init Scale ($\\alpha$)')
    ax.set_zlabel('Log10(Steps to Grok)')
    ax.set_title('3D Grokking Manifold', fontweight='bold')
    
    # 视角调整：俯视侧面，看清相变陡峭程度
    ax.view_init(elev=35, azim=-135)
    
    # Colorbar
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label='Log10(Steps)')

    # --- 子图 2: 平面投影相图 (Phase Diagram) ---
    ax2 = fig.add_subplot(1, 2, 2)
    
    sns.heatmap(pivot, ax=ax2, cmap="Spectral_r", 
                annot=True, fmt=".1f",
                cbar_kws={'label': 'Log10($T_{grok}$)'})
    
    ax2.set_xlabel('Data Size ($N$)')
    ax2.set_ylabel('Initialization Scale ($\\alpha$)')
    ax2.set_title('Phase Diagram: The "Goldilocks Zone"', fontweight='bold')
    # 反转Y轴使大 Alpha 在上
    ax2.invert_yaxis()

    plt.tight_layout()
    save_path = os.path.join(SAVE_DIR, 'omnigrok_manifold.png')
    plt.savefig(save_path)
    print(f"图表已保存: {save_path}")

if __name__ == '__main__':
    plot_3d_manifold()