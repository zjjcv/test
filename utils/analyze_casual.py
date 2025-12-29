import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tqdm import tqdm
import math

# =========================================================================
# 0. Path & Config Setup
# =========================================================================
current_file_path = os.path.abspath(__file__) if '__file__' in locals() else os.getcwd()
PROJECT_ROOT = "/data/zjj/test" 

# Config
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
STEPS = [100, 1000, 10000, 100000]
WD_SETTINGS = ['wd_0.0', 'wd_1.0'] # Row 0, Row 1

# Updated Path for GPT2-Large x_plus_y task
BASE_CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, 'results', 'gpt2_medium', 'checkpoints', 'x_minus_y')
PLOT_DIR = os.path.join(PROJECT_ROOT, 'results', 'analysis_plots')
os.makedirs(PLOT_DIR, exist_ok=True)

# Modulo Arithmetic Parameter
P = 97 

# Plotting Style
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
# 1. Model Architecture (Custom GPT2-Medium)
# =========================================================================

class CausalSelfAttention(nn.Module):
    def __init__(self, dim, num_heads, dropout=0.0):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads

        self.c_attn = nn.Linear(dim, 3 * dim)
        self.c_proj = nn.Linear(dim, dim)

        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.dim, dim=2)

        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

        if mask is not None:
            att = att + mask[:T, :T]

        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    def __init__(self, dim, dropout=0.0):
        super().__init__()
        self.c_fc = nn.Linear(dim, 4 * dim)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class GPT2Block(nn.Module):
    def __init__(self, dim, num_heads, dropout=0.0):
        super().__init__()
        self.ln_1 = nn.LayerNorm(dim)
        self.attn = CausalSelfAttention(dim, num_heads, dropout)
        self.ln_2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, dropout)

    def forward(self, x, mask=None):
        x = x + self.attn(self.ln_1(x), mask)
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT2Decoder(nn.Module):
    # GPT-2 Medium Config: dim=1024, layers=24, heads=16
    def __init__(self, dim=1024, num_layers=24, num_heads=16, num_tokens=99, seq_len=5, dropout=0.0):
        super().__init__()
        self.config = type('Config', (), {'n_layer': num_layers, 'n_head': num_heads, 'n_embd': dim})() # Mock config for Tracer
        self.token_embeddings = nn.Embedding(num_tokens, dim)
        self.position_embeddings = nn.Embedding(seq_len, dim)
        self.drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([GPT2Block(dim, num_heads, dropout) for _ in range(num_layers)])
        self.ln_f = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_tokens, bias=False)

        mask = torch.triu(torch.full((seq_len, seq_len), float('-inf')), diagonal=1)
        self.register_buffer("causal_mask", mask)
        self.register_buffer("pos_ids", torch.arange(seq_len))

    def forward(self, x):
        B, T = x.size()
        tok_emb = self.token_embeddings(x)
        pos_emb = self.position_embeddings(self.pos_ids[:T])
        h = self.drop(tok_emb + pos_emb)

        mask = self.causal_mask[:T, :T]

        for block in self.blocks:
            h = block(h, mask)

        h = self.ln_f(h)
        logits = self.head(h)
        # return logits to be consistent, but for tracer we usually wrap output in an object
        return type('obj', (object,), {'logits': logits})

# =========================================================================
# 2. Causal Mediation Core (Adapted for Custom Model)
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
            layer = self.model.blocks[i] # Access via .blocks
            def get_hook_fn(layer_idx):
                def hook_fn(module, args):
                    # args[0] is the input to c_proj, which is the output of attn context
                    activations[f'l{layer_idx}'] = args[0].detach().clone()
                return hook_fn
            # Register on attn.c_proj
            hooks.append(layer.attn.c_proj.register_forward_pre_hook(get_hook_fn(i)))
        
        with torch.no_grad():
            self.model(input_ids)
        for h in hooks: h.remove()
        return activations

    def run_cma(self, c1_ids, c2_ids, target_idx_c1, target_idx_c2):
        with torch.no_grad():
            clean_output = self.model(c1_ids)
            # --- 修改开始: 转换为概率 ---
            clean_probs = F.softmax(clean_output.logits, dim=-1) 
            base_clean_prob = clean_probs[0, -1, target_idx_c1]
            # --- 修改结束 ---

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

                layer_module = self.model.blocks[layer_idx].attn.c_proj
                handle = layer_module.register_forward_pre_hook(single_head_patch_hook)
                with torch.no_grad():
                    patched_output = self.model(c1_ids)
                    # --- 修改开始: 转换为概率 ---
                    patched_probs = F.softmax(patched_output.logits, dim=-1)
                    patched_clean_prob = patched_probs[0, -1, target_idx_c1]
                    # --- 修改结束 ---
                
                handle.remove()
                
                # 计算概率差
                effect_matrix[layer_idx, head_idx] = (base_clean_prob - patched_clean_prob).item()
        
        return effect_matrix

# =========================================================================
# 3. Helpers
# =========================================================================

def generate_mod_data_x_plus_y():
    """ 生成 x + y (mod P) 的对照样本 """
    a = np.random.randint(0, P)
    b1 = np.random.randint(0, P)
    b2 = (b1 + np.random.randint(1, P)) % P 
    c1 = (a + b1) % P # Changed to + for x_plus_y task
    c2 = (a + b2) % P
    # 格式: x op y eq -> result
    # P=97, so op_token=97, eq_token=98
    prompt_c1 = f"{a} {P} {b1} {P+1}" 
    prompt_c2 = f"{a} {P} {b2} {P+1}"
    return prompt_c1, prompt_c2, c1, c2

class ModTokenizer:
    def __call__(self, text, return_tensors='pt'):
        parts = text.strip().split()
        ids = [int(p) for p in parts]
        return type('obj', (object,), {'input_ids': torch.tensor([ids])})

# =========================================================================
# 4. Main Execution
# =========================================================================

def main():
    print("🚀 Starting Refined CMA Plotting for GPT2-Medium (x-y)...")
    
    # 2行4列
    # GPT2-Medium has 24 layers, 16 heads. Adjust figsize ratio.
    # 16 heads vs 24 layers -> Height is larger.
    fig, axes = plt.subplots(2, 4, figsize=(20, 8), dpi=150, sharex=True, sharey=True)
    plt.subplots_adjust(wspace=0.05, hspace=0.1, right=0.91, left=0.08, bottom=0.15)
    
    tokenizer = ModTokenizer()
    heatmap_data_cache = {}

    for row_idx, wd_val in enumerate(WD_SETTINGS):
        for col_idx, step in enumerate(STEPS):
            print(f"👉 Processing {wd_val} - Step {step}")
            
            # --- Load Path ---
            ckpt_path = os.path.join(BASE_CHECKPOINT_DIR, wd_val, f"seed42_step{step}.pt")
            
            if not os.path.exists(ckpt_path):
                print(f"  [Warn] Not found: {ckpt_path}")
                heatmap_data_cache[(row_idx, col_idx)] = torch.zeros((24, 16)) # Placeholder
                continue

            try:
                # Load State Dict
                checkpoint = torch.load(ckpt_path, map_location=DEVICE)
                state_dict = checkpoint['model_state_dict']
                
                # Init Model (GPT2-Medium Config)
                model = GPT2Decoder(
                    dim=1024, num_layers=24, num_heads=16,
                    num_tokens=P+2, seq_len=5, dropout=0.0
                )
                model.load_state_dict(state_dict)
                model.to(DEVICE)
                
                tracer = CausalTracer(model)
                avg_heatmap = torch.zeros((24, 16))
                
                # Average over N samples
                for _ in range(5): 
                    p1, p2, ans1, ans2 = generate_mod_data_x_plus_y()
                    c1_ids = tokenizer(p1, return_tensors='pt').input_ids.to(DEVICE)
                    c2_ids = tokenizer(p2, return_tensors='pt').input_ids.to(DEVICE)
                    avg_heatmap += tracer.run_cma(c1_ids, c2_ids, int(ans1), int(ans2))
                avg_heatmap /= 5
                
                heatmap_data_cache[(row_idx, col_idx)] = avg_heatmap

            except Exception as e:
                print(f"  [Error] Failed to process {ckpt_path}: {e}")
                heatmap_data_cache[(row_idx, col_idx)] = torch.zeros((36, 20))

    # --- Plotting ---
    print("🎨 Rendering plots...")

    ROW_LABELS = ["No Regularization\n(WD=0.0)", "With Regularization\n(WD=1.0)"]

    for row_idx in range(2):
        for col_idx in range(4):
            ax = axes[row_idx, col_idx]
            data = heatmap_data_cache.get((row_idx, col_idx), torch.zeros((24, 16)))
            
            # Robust Max for visualization
            robust_max = np.percentile(data.numpy(), 99.5)
            # Ensure a minimum visibility even if values are low
            vmax = max(0.01, robust_max) 
            if col_idx == 3: vmax = max(0.01, robust_max * 0.8)

            sns.heatmap(
                data.numpy(),
                cmap='magma',
                ax=ax,
                cbar=False,
                vmin=0,
                vmax=vmax,
                linewidths=0.0
            )
            
            ax.invert_yaxis() # Layer 0 at bottom
            
            # Ticks for Medium Model (24 layers, 16 heads)
            # Y-axis: Layer every 4
            yticks = np.arange(0, 24, 4) + 0.5
            yticklabels = np.arange(0, 24, 4)
            ax.set_yticks(yticks)
            ax.set_yticklabels(yticklabels)
            
            # X-axis: Head every 4
            xticks = np.arange(0, 16, 4) + 0.5
            xticklabels = np.arange(0, 16, 4)
            ax.set_xticks(xticks)
            ax.set_xticklabels(xticklabels)

            if col_idx == 0:
                ax.set_ylabel("Layer", fontsize=12, fontweight='bold')
            else:
                ax.set_ylabel("")

            if row_idx == 1:
                ax.set_xlabel("Head", fontsize=12, fontweight='bold')
            else:
                ax.set_xlabel("")

            if row_idx == 0:
                ax.set_title(f"Step {STEPS[col_idx]}", fontsize=14, fontweight='bold', pad=12)

    # Row Labels
    for ax, label in zip(axes[:, 0], ROW_LABELS):
        ax.annotate(label, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - 35, 0),
                    xycoords=ax.yaxis.label, textcoords='offset points',
                    size=12, ha='right', va='center', rotation=90, fontweight='bold')

    # Colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7]) 
    cbar = fig.colorbar(axes[0,0].collections[0], cax=cbar_ax)
    cbar.set_label("Causal Mediation Effect (Probability Difference)", fontsize=12, labelpad=10)
    cbar.outline.set_visible(False)

    save_path = os.path.join(PLOT_DIR, "cma_gpt2_medium_x_medium_y.png")
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f"\n✨ 最终图表已保存到: {save_path}")

if __name__ == "__main__":
    main()