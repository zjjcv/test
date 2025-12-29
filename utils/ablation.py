import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import math
from tqdm import tqdm

# =========================================================================
# 0. Config & Setup
# =========================================================================
PROJECT_ROOT = "/data/zjj/test" 
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Updated Path for GPT2-Large x_plus_y task
# Assuming we want to analyze the final converged model (e.g., step 100,000 with WD=1.0)
CKPT_PATH = os.path.join(PROJECT_ROOT, 'results', 'gpt2-large', 'checkpoints', 'x_plus_y', 'wd_0.0', 'seed42_step100000.pt')
PLOT_DIR = os.path.join(PROJECT_ROOT, 'results', 'analysis_plots')
os.makedirs(PLOT_DIR, exist_ok=True)

P = 97 

# Plotting Style
plt.style.use('seaborn-v0_8-paper')
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 12,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'lines.linewidth': 2.0
})

# =========================================================================
# 1. Model Architecture (Custom GPT2-Large)
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
    # GPT-2 Large Config: dim=1280, layers=36, heads=20
    def __init__(self, dim=1280, num_layers=36, num_heads=20, num_tokens=99, seq_len=5, dropout=0.0, use_checkpoint=True):
        super().__init__()
        self.token_embeddings = nn.Embedding(num_tokens, dim)
        self.position_embeddings = nn.Embedding(seq_len, dim)
        self.drop = nn.Dropout(dropout)
        self.use_checkpoint = use_checkpoint
        self.blocks = nn.ModuleList([GPT2Block(dim, num_heads, dropout) for _ in range(num_layers)])
        self.ln_f = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_tokens, bias=False)
        mask = torch.triu(torch.full((seq_len, seq_len), float('-inf')), diagonal=1)
        self.register_buffer("causal_mask", mask)
        self.register_buffer("pos_ids", torch.arange(seq_len))
        
        self.config = type('Config', (), {'n_layer': num_layers, 'n_head': num_heads, 'n_embd': dim})()

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
        return logits

# =========================================================================
# 2. Helper Functions
# =========================================================================

def load_native_model():
    print(f"📂 Loading Native Model from: {CKPT_PATH}")
    if not os.path.exists(CKPT_PATH):
        raise FileNotFoundError(f"Checkpoint not found: {CKPT_PATH}")
        
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
    
    # Init Model (GPT2-Large Config)
    model = GPT2Decoder(
        dim=1280, num_layers=36, num_heads=20,
        num_tokens=P + 2, 
        seq_len=5, dropout=0.0, use_checkpoint=False
    )
    
    state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
    
    new_sd = {}
    for k, v in state_dict.items():
        if k.startswith('module.'): new_sd[k[7:]] = v
        else: new_sd[k] = v
        
    keys = model.load_state_dict(new_sd, strict=True)
    print(f"   Load Status: {keys}")
    
    model.to(DEVICE)
    model.eval()
    return model

def generate_batch_x_plus_y(batch_size=1024):
    """ 生成 (x + y) mod P 的数据 """
    a = torch.randint(0, P, (batch_size,))
    b = torch.randint(1, P, (batch_size,)) 
    
    input_ids = torch.stack([
        a,
        torch.full_like(a, P),      # op token
        b,
        torch.full_like(a, P+1)     # eq token
    ], dim=1).to(DEVICE)
    
    targets = (a + b) % P # Corrected for x_plus_y
    targets = targets.to(DEVICE)
    return input_ids, targets

def get_accuracy(model, inputs, targets):
    with torch.no_grad():
        logits = model(inputs) 
        preds = logits[:, -1, :].argmax(dim=-1)
        acc = (preds == targets).float().mean().item()
    return acc

# =========================================================================
# 3. Ablation Logic
# =========================================================================

class HeadAblator:
    def __init__(self, model):
        self.model = model
        self.n_layer = model.config.n_layer
        self.n_head = model.config.n_head
        self.head_dim = model.config.n_embd // self.n_head
        self.mask = torch.ones((self.n_layer, self.n_head), device=DEVICE)
        self.hooks = []
        self._register_hooks()
        
    def _register_hooks(self):
        for i in range(self.n_layer):
            layer = self.model.blocks[i]
            def get_hook(layer_idx):
                def hook_fn(module, args):
                    x = args[0]
                    B, T, C = x.shape
                    x = x.view(B, T, self.n_head, self.head_dim)
                    mask = self.mask[layer_idx].view(1, 1, self.n_head, 1)
                    x = x * mask
                    return x.view(B, T, C)
                return hook_fn
            self.hooks.append(layer.attn.c_proj.register_forward_pre_hook(get_hook(i)))

    def reset(self):
        self.mask.fill_(1.0)
        
    def ablate(self, heads_list):
        for l, h in heads_list:
            self.mask[l, h] = 0.0
            
    def close(self):
        for h in self.hooks: h.remove()

# =========================================================================
# 4. CMA Importance
# =========================================================================

def calculate_importance(model):
    print("Running CMA scan...")
    ablator = HeadAblator(model) 
    ablator.close() 
    
    batch_size = 64
    a = torch.randint(0, P, (batch_size,))
    b1 = torch.randint(1, P, (batch_size,))
    b2 = (b1 + 1) % P 
    b2[b2==0] = 1 
    
    c1_ids = torch.stack([a, torch.full_like(a, P), b1, torch.full_like(a, P+1)], dim=1).to(DEVICE)
    c2_ids = torch.stack([a, torch.full_like(a, P), b2, torch.full_like(a, P+1)], dim=1).to(DEVICE)
    targets = (a + b1) % P # x_plus_y
    targets = targets.to(DEVICE)
    
    source_acts = {}
    temp_hooks = []
    for i in range(model.config.n_layer):
        def get_cache(l):
            def hook(m, args): source_acts[f'l{l}'] = args[0].detach().clone()
            return hook
        temp_hooks.append(model.blocks[i].attn.c_proj.register_forward_pre_hook(get_cache(i)))
    with torch.no_grad(): model(c2_ids)
    for h in temp_hooks: h.remove()
    
    base_acc = get_accuracy(model, c1_ids, targets)
    
    importances = []
    n_head = model.config.n_head
    head_dim = model.config.n_embd // n_head
    
    for l in tqdm(range(model.config.n_layer), leave=False):
        src_layer = source_acts[f'l{l}']
        for h in range(n_head):
            def patch_hook(m, args):
                inp = args[0]
                B, T, _ = inp.shape
                inp = inp.view(B, T, n_head, head_dim)
                src = src_layer.view(B, T, n_head, head_dim)
                patched = inp.clone()
                patched[:, -1, h, :] = src[:, -1, h, :]
                return patched.view(B, T, -1)
            
            handle = model.blocks[l].attn.c_proj.register_forward_pre_hook(patch_hook)
            with torch.no_grad():
                logits = model(c1_ids)
            handle.remove()
            
            probs = torch.softmax(logits[:, -1, :], dim=-1)
            patched_score = probs.gather(1, targets.unsqueeze(1)).mean().item()
            
            importances.append( ((l, h), base_acc - patched_score) )
            
    return sorted(importances, key=lambda x: x[1], reverse=True)

# =========================================================================
# 5. Main Execution
# =========================================================================

def run_curve(model, ablator, inputs, targets, head_order):
    ablator.reset()
    accs = [get_accuracy(model, inputs, targets)]
    steps = [0]
    
    chunk_size = 1
    total_heads = len(head_order)
    
    # Adaptive chunking for speed
    if total_heads > 200: chunk_size = 5
    
    for i in range(0, total_heads, chunk_size):
        batch = head_order[i : i+chunk_size]
        ablator.ablate(batch)
        
        # Measure more frequently at start
        if i < 50 or i % (chunk_size * 5) == 0:
            acc = get_accuracy(model, inputs, targets)
            accs.append(acc)
            steps.append(i + len(batch))
            
    return steps, accs

def main():
    print("🚀 Starting GPT2-Large Ablation Analysis (36 Layers, 20 Heads)...")
    model = load_native_model()
    
    inputs, targets = generate_batch_x_plus_y(1024)
    base_acc = get_accuracy(model, inputs, targets)
    print(f"   Base Accuracy: {base_acc:.4f}")
    if base_acc < 0.9:
        print("❌ Warning: Accuracy < 90%.")

    sorted_heads = calculate_importance(model)
    ablator = HeadAblator(model)
    all_heads = [h[0] for h in sorted_heads]
    
    # --- Split into 12-layer chunks ---
    # Shallow: 0-11, Middle: 12-23, Deep: 24-35
    l0_11_heads  = [h for h in all_heads if 0 <= h[0] <= 11]
    l12_23_heads = [h for h in all_heads if 12 <= h[0] <= 23]
    l24_35_heads = [h for h in all_heads if 24 <= h[0] <= 35]
    
    experiments = [
        ("Global (All Layers)", all_heads),
        ("Layers 0-11 (Shallow)", l0_11_heads),
        ("Layers 12-23 (Middle)", l12_23_heads),
        ("Layers 24-35 (Deep)", l24_35_heads)
    ]
    
    fig, axes = plt.subplots(1, 4, figsize=(22, 5.5), dpi=150, sharey=True)
    plt.subplots_adjust(wspace=0.1)
    
    for idx, (title, scope_heads) in enumerate(experiments):
        print(f"👉 Running {title} ({len(scope_heads)} heads)...")
        ax = axes[idx]
        scope_set = set(scope_heads)
        
        # Order: High CMA -> Low CMA
        red_order = [h for h, s in sorted_heads if h in scope_set]
        
        # --- Run Random (Blue) ---
        rand_res = []
        rand_x = []
        for _ in range(3): # Reduce iterations for speed on large model
            np.random.shuffle(scope_heads)
            rx, ry = run_curve(model, ablator, inputs, targets, scope_heads)
            rand_res.append(ry)
            rand_x = rx
        
        # Align lengths if needed (simple approach: trim to min)
        min_len = min([len(r) for r in rand_res])
        rand_res = [r[:min_len] for r in rand_res]
        rand_mean = np.mean(rand_res, axis=0)
        rand_std = np.std(rand_res, axis=0)
        rand_x = rand_x[:min_len]
        
        ax.plot(rand_x, rand_mean, color='blue', label='Random', alpha=0.8)
        ax.fill_between(rand_x, rand_mean-rand_std, rand_mean+rand_std, color='blue', alpha=0.1)
        
        # --- Run Ablation (Red) ---
        rx, ry = run_curve(model, ablator, inputs, targets, red_order)
        ax.plot(rx, ry, color='red', label='Ablation (High CMA first)')
        
        ax.set_title(title, fontweight='bold', fontsize=14)
        ax.set_xlabel("Number of heads ablated")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, linestyle=':', alpha=0.4)
        
        # Ticks based on head count
        total_heads_in_scope = len(scope_heads)
        if total_heads_in_scope > 200: # Global (720)
            ax.set_xticks(np.arange(0, total_heads_in_scope + 1, 100))
        else: # Groups (240)
            ax.set_xticks(np.arange(0, total_heads_in_scope + 1, 40))
        
        if idx == 0:
            ax.set_ylabel("P(Correct Answer)", fontsize=12)
            ax.legend(loc='upper right', frameon=True)
            
    ablator.close()
    
    save_path = os.path.join(PLOT_DIR, "ablation_gpt2_large_x_plus_y_wd0.0_100k.png")
    plt.savefig(save_path, bbox_inches='tight')
    print(f"\n✅ Plot saved to: {save_path}")

if __name__ == "__main__":
    main()