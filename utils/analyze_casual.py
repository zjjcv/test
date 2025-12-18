import os
import sys
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tqdm import tqdm
from transformers import GPT2LMHeadModel, GPT2Config

# =========================================================================
# 0. Path & Config Setup
# =========================================================================
current_file_path = os.path.abspath(__file__) if '__file__' in locals() else os.getcwd()
PROJECT_ROOT = "/data/zjj/test" 

# 配置
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
STEPS = [100, 1000, 10000, 100000]
WD_SETTINGS = ['wd_0.0', 'wd_1.0'] # Row 0, Row 1
BASE_CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, 'results', 'checkpoints', 'x-y')
PLOT_DIR = os.path.join(PROJECT_ROOT, 'results', 'analysis_plots')
os.makedirs(PLOT_DIR, exist_ok=True)

# 模运算参数
P = 97 

# 设置绘图风格 (类似 NIPS/ICLR 论文风格)
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'font.size': 12,
    'axes.labelsize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.titlesize': 16
})

# =========================================================================
# 1. Utilities (Same as before)
# =========================================================================

def process_checkpoint(state_dict):
    new_sd = {}
    config_updates = {}
    keys = list(state_dict.keys())
    wte_key = next((k for k in keys if 'token_embeddings' in k or 'wte.weight' in k), None)
    if wte_key:
        shape = state_dict[wte_key].shape
        config_updates['vocab_size'] = shape[0]
        config_updates['n_embd'] = shape[1]
    wpe_key = next((k for k in keys if 'position_embeddings' in k or 'wpe.weight' in k), None)
    if wpe_key:
        shape = state_dict[wpe_key].shape
        config_updates['n_positions'] = shape[0]
    for k, v in state_dict.items():
        if k.endswith('.attn.bias') or k == 'causal_mask' or k == 'pos_ids': continue
        new_k = k
        if 'token_embeddings' in k: new_k = k.replace('token_embeddings', 'transformer.wte')
        elif 'position_embeddings' in k: new_k = k.replace('position_embeddings', 'transformer.wpe')
        elif 'blocks.' in k: new_k = k.replace('blocks.', 'transformer.h.')
        elif 'ln_f' in k: new_k = k.replace('ln_f', 'transformer.ln_f')
        elif 'head.weight' in k: new_k = k.replace('head.weight', 'lm_head.weight')
        new_v = v
        if 'weight' in k and any(x in k for x in ['c_attn', 'c_proj', 'c_fc']):
            new_v = v.t()
        new_sd[new_k] = new_v
    if 'transformer.wte.weight' in new_sd and 'lm_head.weight' not in new_sd:
        new_sd['lm_head.weight'] = new_sd['transformer.wte.weight']
    return new_sd, config_updates

# =========================================================================
# 2. Causal Mediation Core
# =========================================================================

class CausalTracer:
    def __init__(self, model):
        self.model = model
        self.model.eval()
        self.n_layers = model.config.n_layer
        self.n_heads = model.config.n_head
        self.head_dim = model.config.n_embd // self.n_heads

    def get_activations(self, input_ids):
        activations = {}
        hooks = []
        for i in range(self.n_layers):
            layer = self.model.transformer.h[i]
            def get_hook_fn(layer_idx):
                def hook_fn(module, args):
                    activations[f'l{layer_idx}'] = args[0].detach().clone()
                return hook_fn
            hooks.append(layer.attn.c_proj.register_forward_pre_hook(get_hook_fn(i)))
        with torch.no_grad():
            self.model(input_ids)
        for h in hooks: h.remove()
        return activations

    def run_cma(self, c1_ids, c2_ids, target_idx_c1, target_idx_c2):
        with torch.no_grad():
            clean_logits = self.model(c1_ids).logits
            base_clean_logit = clean_logits[0, -1, target_idx_c1]

        source_acts = self.get_activations(c2_ids)
        effect_matrix = torch.zeros((self.n_layers, self.n_heads))
        patch_pos = -1 

        for layer_idx in range(self.n_layers):
            src_layer_act = source_acts[f'l{layer_idx}']
            for head_idx in range(self.n_heads):
                def single_head_patch_hook(module, args):
                    inp = args[0] 
                    batch, seq, _ = inp.shape
                    inp_reshaped = inp.view(batch, seq, self.n_heads, self.head_dim)
                    src_reshaped = src_layer_act.view(batch, seq, self.n_heads, self.head_dim)
                    patched_inp = inp_reshaped.clone()
                    patched_inp[:, patch_pos, head_idx, :] = src_reshaped[:, patch_pos, head_idx, :]
                    return patched_inp.view(batch, seq, -1)

                layer_module = self.model.transformer.h[layer_idx].attn.c_proj
                handle = layer_module.register_forward_pre_hook(single_head_patch_hook)
                with torch.no_grad():
                    patched_logits = self.model(c1_ids).logits
                handle.remove()
                patched_clean_logit = patched_logits[0, -1, target_idx_c1]
                effect_matrix[layer_idx, head_idx] = (base_clean_logit - patched_clean_logit).item()
        return effect_matrix

# =========================================================================
# 3. Helpers (Corrected for x-y task)
# =========================================================================

def generate_mod_data():
    """ 生成 x - y (mod P) 的对照样本 """
    a = np.random.randint(0, P)
    b1 = np.random.randint(0, P)
    b2 = (b1 + np.random.randint(1, P)) % P 
    c1 = (a - b1) % P
    c2 = (a - b2) % P
    # 格式必须与训练一致: x op y eq
    prompt_c1 = f"{a} {P} {b1} {P+1}" 
    prompt_c2 = f"{a} {P} {b2} {P+1}"
    return prompt_c1, prompt_c2, c1, c2

class ModTokenizer:
    def __call__(self, text, return_tensors='pt'):
        parts = text.strip().split()
        ids = [int(p) for p in parts]
        return type('obj', (object,), {'input_ids': torch.tensor([ids])})

# =========================================================================
# 4. Main Execution (Refined Plotting)
# =========================================================================

def main():
    print("🚀 Starting Refined CMA Plotting...")
    
    # 定义阶段名称
    PHASE_NAMES = ["Early", "Memorization", "Grokking", "Generalization"]
    
    # 2行4列
    fig, axes = plt.subplots(2, 4, figsize=(18, 8), dpi=150, sharex=True, sharey=True)
    # 调整子图间距，留出右侧给 colorbar
    plt.subplots_adjust(wspace=0.1, hspace=0.1, right=0.9)
    
    tokenizer = ModTokenizer()
    
    # 为了统一 Colorbar，我们需要先收集所有数据或者设定一个全局 vmax
    # 但由于 Step 100 和 Step 100000 的 Impact 差异巨大，建议每张图独立归一化，
    # 或者每行统一。为了展示结构性变化，独立归一化更能看清每一步的内部结构。
    
    heatmap_data_cache = {}

    for row_idx, wd_val in enumerate(WD_SETTINGS):
        for col_idx, step in enumerate(STEPS):
            print(f"👉 Processing {wd_val} - Step {step}")
            
            # --- Load & Calculate ---
            ckpt_path = os.path.join(BASE_CHECKPOINT_DIR, wd_val, f"seed42_step{step}.pt")
            
            if not os.path.exists(ckpt_path):
                heatmap_data_cache[(row_idx, col_idx)] = torch.zeros((24, 16)) # Placeholder
                continue

            try:
                raw_data = torch.load(ckpt_path, map_location=DEVICE)
                raw_sd = raw_data['model_state_dict'] if 'model_state_dict' in raw_data else raw_data
                state_dict, config_updates = process_checkpoint(raw_sd)
                config = GPT2Config.from_pretrained('gpt2-medium')
                if 'vocab_size' in config_updates: config.vocab_size = config_updates['vocab_size']
                if 'n_positions' in config_updates: config.n_positions = config_updates['n_positions']
                
                model = GPT2LMHeadModel(config)
                model.load_state_dict(state_dict, strict=False)
                model.to(DEVICE)
                
                tracer = CausalTracer(model)
                avg_heatmap = torch.zeros((config.n_layer, config.n_head))
                
                # 采样 N 次取平均以获得平滑的热力图
                for _ in range(5): 
                    p1, p2, ans1, ans2 = generate_mod_data()
                    c1_ids = tokenizer(p1, return_tensors='pt').input_ids.to(DEVICE)
                    c2_ids = tokenizer(p2, return_tensors='pt').input_ids.to(DEVICE)
                    avg_heatmap += tracer.run_cma(c1_ids, c2_ids, int(ans1), int(ans2))
                avg_heatmap /= 5
                
                heatmap_data_cache[(row_idx, col_idx)] = avg_heatmap

            except Exception as e:
                print(f"Error: {e}")
                heatmap_data_cache[(row_idx, col_idx)] = torch.zeros((24, 16))

# =========================================================================
    # 替换 main 函数中 "# --- Plotting ---" 之后的所有代码
    # =========================================================================
    print("🎨 Rendering final paper-style plots...")

    # 1. 设置画布：拉宽比例 (22x6)，使用 sharex/y 共享坐标轴
    fig, axes = plt.subplots(2, 4, figsize=(22, 6), dpi=150, sharex=True, sharey=True)
    # 2. 调整间距：极小的 wspace/hspace 实现紧凑布局，预留 left 给行标题，right 给色条
    plt.subplots_adjust(wspace=0.05, hspace=0.1, right=0.91, left=0.08, bottom=0.15)

    # 定义行标题
    ROW_LABELS = ["No Regularization\n(WD=0.0)", "With Regularization\n(WD=1.0)"]

    for row_idx in range(2):
        for col_idx in range(4):
            ax = axes[row_idx, col_idx]
            data = heatmap_data_cache.get((row_idx, col_idx), torch.zeros((24,16)))
            
            # 动态 Vmax 计算 (保持不变)
            robust_max = np.percentile(data.numpy(), 99.5)
            if col_idx == 3: vmax = max(0.01, robust_max * 0.8)
            else: vmax = max(0.01, robust_max)

            # 3. 绘制热力图：移除 square=True 以允许拉伸
            sns.heatmap(
                data.numpy(),
                cmap='magma',
                ax=ax,
                cbar=False,
                # square=True, # <--- 移除此行，允许热力图横向拉伸
                vmin=0,
                vmax=vmax,
                linewidths=0.0
            )
            
            ax.invert_yaxis() # Layer 0 在底部
            
            # --- 4. 坐标轴与刻度美化 ---
            # 统一设置刻度位置和标签文本 (sharex/y 会自动隐藏中间图的标签文本)
            
            # Y轴 (Layer): 每4层标一个数字
            yticks = np.arange(0, 24, 4) + 0.5
            yticklabels = np.arange(0, 24, 4)
            ax.set_yticks(yticks)
            ax.set_yticklabels(yticklabels)
            
            # X轴 (Head): 每4个头标一个数字
            xticks = np.arange(0, 16, 4) + 0.5
            xticklabels = np.arange(0, 16, 4)
            ax.set_xticks(xticks)
            ax.set_xticklabels(xticklabels)

            # 仅在第一列显示 Y 轴标题
            if col_idx == 0:
                ax.set_ylabel("Layer", fontsize=12, fontweight='bold')
            else:
                ax.set_ylabel("")

            # 仅在最后一行显示 X 轴标题
            if row_idx == 1:
                ax.set_xlabel("Head", fontsize=12, fontweight='bold')
            else:
                ax.set_xlabel("")

            # --- 5. 设置顶部标题 (改为 Step 数值) ---
            if row_idx == 0:
                # 直接使用 STEPS 列表中的数值
                ax.set_title(f"Step {STEPS[col_idx]}", fontsize=14, fontweight='bold', pad=12)

    # --- 6. 添加左侧行标题 ---
    for ax, label in zip(axes[:, 0], ROW_LABELS):
        ax.annotate(label, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - 25, 0),
                    xycoords=ax.yaxis.label, textcoords='offset points',
                    size=12, ha='right', va='center', rotation=90, fontweight='bold')

    # --- 7. 添加统一的 Colorbar ---
    # 调整位置以适应新的画布比例
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7]) # [left, bottom, width, height]
    cbar = fig.colorbar(axes[0,0].collections[0], cax=cbar_ax)
    cbar.set_label("Causal Impact (Logit Difference)", fontsize=12, labelpad=10)
    cbar.outline.set_visible(False)

    save_path = os.path.join(PLOT_DIR, "cma_comparison_final_paper_style_wide.png")
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f"\n✨ Final wide-layout plot saved to: {save_path}")

if __name__ == "__main__":
    main()