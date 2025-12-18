import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.neighbors import NearestNeighbors
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr, spearmanr

# 设置科研风格的绘图配置
plt.style.use('seaborn-v0_8-paper')
sns.set_context("paper", font_scale=1.4)

class GrokkingAnalyzer:
    def __init__(self, k_neighbors=20):
        """
        初始化分析器
        :param k_neighbors: 计算局部密度时参考的邻居数量
        """
        self.k = k_neighbors
        self.scaler = StandardScaler()

    def calculate_voi_metrics(self, embeddings):
        """
        计算 Value of Information (VoI)
        方法：KNN Distance (距离越远 -> 密度越低 -> 处于边缘 -> VoI越高)
        """
        # 1. 数据标准化
        X = self.scaler.fit_transform(embeddings)
        
        # 2. 计算 KNN
        nbrs = NearestNeighbors(n_neighbors=self.k + 1).fit(X) 
        distances, _ = nbrs.kneighbors(X)
        
        # 3. VoI Score 定义：取平均邻居距离
        voi_scores = distances[:, 1:].mean(axis=1)
        
        return voi_scores

    def run_analysis(self, hidden_states, losses, step_name="Step ?"):
        """
        执行完整分析流程并保存图片
        """
        # 数据转换
        if isinstance(hidden_states, torch.Tensor):
            hidden_states = hidden_states.detach().cpu().numpy()
        if isinstance(losses, torch.Tensor):
            losses = losses.detach().cpu().numpy()

        print(f"--- Running Analysis for {step_name} ---")
        
        # 1. 计算 VoI
        voi_scores = self.calculate_voi_metrics(hidden_states)
        
        # 2. 统计相关性
        p_corr, p_val = pearsonr(voi_scores, losses)
        s_corr, s_val = spearmanr(voi_scores, losses)
        
        print(f"Stats -> Pearson: {p_corr:.4f}, Spearman: {s_corr:.4f}")

        # 3. 降维可视化 (用于验证 VoI 分布)
        print("Computing t-SNE for visualization...")
        # 【修复1】使用 max_iter 兼容新版 sklearn
        tsne = TSNE(n_components=2, perplexity=30, max_iter=1000, random_state=42)
        X_embedded = tsne.fit_transform(hidden_states)

        # --- 绘图 ---
        fig = plt.figure(figsize=(20, 6))
        
        # 图 1: VoI vs Loss 相关性
        ax1 = fig.add_subplot(131)
        sns.regplot(x=voi_scores, y=losses, ax=ax1,
                    scatter_kws={'alpha': 0.4, 's': 15, 'color': '#6c5ce7'},
                    line_kws={'color': '#d63031', 'linewidth': 2})
        ax1.set_title(f"Correlation Analysis\nPearson r={p_corr:.2f}, Spearman r={s_corr:.2f}")
        ax1.set_xlabel("VoI (Distance to Local Cluster Center)")
        ax1.set_ylabel("Loss (Model Uncertainty)")
        ax1.grid(True, alpha=0.3)

        # 图 2: 脑区划分 (按 VoI 上色)
        ax2 = fig.add_subplot(132)
        scatter2 = ax2.scatter(X_embedded[:, 0], X_embedded[:, 1], 
                               c=voi_scores, cmap='viridis', s=10, alpha=0.8)
        plt.colorbar(scatter2, ax=ax2, label='VoI Score (Local Sparsity)')
        ax2.set_title(f"Brain Partitions (Colored by VoI)\nTarget: Yellow on Boundaries")
        ax2.axis('off')

        # 图 3: 脑区划分 (按 Loss 上色)
        ax3 = fig.add_subplot(133)
        scatter3 = ax3.scatter(X_embedded[:, 0], X_embedded[:, 1], 
                               c=losses, cmap='magma', s=10, alpha=0.8)
        plt.colorbar(scatter3, ax=ax3, label='Actual Loss')
        ax3.set_title(f"Brain Partitions (Colored by Loss)\nGround Truth Distribution")
        ax3.axis('off')

        plt.tight_layout()
        
        # 【修复2】保存图片而不是 show()
        # 处理文件名，去除非法字符
        safe_name = step_name.replace(' ', '_').replace('?', 'X').replace('(', '').replace(')', '')
        filename = f"analysis_{safe_name}.png"
        save_path = os.path.join(os.getcwd(), filename)
        
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[Success] Plot saved to: {save_path}")
        
        plt.close() # 释放内存
        
        return voi_scores, X_embedded

# ==========================================
# 仿真测试
# ==========================================
if __name__ == "__main__":
    # 模拟数据
    np.random.seed(42)
    n_per_cluster = 300
    dim = 64
    
    c1 = np.random.normal(loc=2, scale=0.5, size=(n_per_cluster, dim))
    c2 = np.random.normal(loc=-2, scale=0.5, size=(n_per_cluster, dim))
    c3 = np.random.normal(loc=0, scale=0.5, size=(n_per_cluster, dim))
    c3[:, 0] += 5 
    
    data = np.vstack([c1, c2, c3])
    
    # 模拟 Loss (基于距离中心的远近)
    from sklearn.metrics import pairwise_distances
    centers = np.array([[2]*dim, [-2]*dim, [5] + [0]*(dim-1)])
    dists = np.min(pairwise_distances(data, centers), axis=1)
    simulated_losses = dists * 2 + np.random.normal(0, 0.2, size=len(data))
    
    # 运行
    analyzer = GrokkingAnalyzer(k_neighbors=20)
    analyzer.run_analysis(data, simulated_losses, step_name="Simulation_Grokking")