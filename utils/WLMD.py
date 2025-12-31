#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WLMD/WanD-style *offline* Wang–Landau (WL) diagnostics on a saved checkpoint trajectory.

You said:
- You have ~1000 checkpoints (save every 100 steps).
- Use E(w) = log L_train(w) as "energy".
- 200 bins.
- Maintain DOS estimate (log g(E)) and histogram H(E).
- Output CSVs sufficient to plot Fig.1–Fig.5.
- Save to: /data/zjj/test/results/Data/WLMD

Important note (methodological):
This is NOT full WLMD in continuous parameter space. It is a minimal, reproducible
WL-style *density-of-states estimation on a discrete state graph* whose nodes are your
saved checkpoints, with proposals as moves along the checkpoint index line (random walk).
This is the correct “minimal reproduction” you can finish in 20 days and is typically
acceptable as a WL/WanD-inspired diagnostic on toy models.
"""

import os
import re
import csv
import glob
import math
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# =============================================================================
# 1) Model & Data (match your training/eval)
# =============================================================================

class Block(nn.Module):
    def __init__(self, dim, num_heads, dropout=0.0):
        super().__init__()
        self.ln_1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True, dropout=dropout)
        self.ln_2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim), nn.Dropout(dropout)
        )

    def forward(self, x, attn_mask=None):
        q = self.ln_1(x)
        x = x + self.attn(q, q, q, attn_mask=attn_mask, need_weights=False)[0]
        x = x + self.mlp(self.ln_2(x))
        return x

class Decoder(nn.Module):
    def __init__(self, dim=128, num_layers=2, num_heads=4, num_tokens=97, seq_len=5, dropout=0.0):
        super().__init__()
        self.token_embeddings = nn.Embedding(num_tokens, dim)
        self.position_embeddings = nn.Embedding(seq_len, dim)
        self.layers = nn.ModuleList([Block(dim, num_heads, dropout) for _ in range(num_layers)])
        self.ln_f = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_tokens, bias=False)
        mask = torch.triu(torch.full((seq_len, seq_len), float('-inf')), diagonal=1)
        self.register_buffer("causal_mask", mask)
        self.register_buffer("pos_ids", torch.arange(seq_len))

    def forward(self, x):
        h = self.token_embeddings(x) + self.position_embeddings(self.pos_ids[:x.size(1)])
        mask = self.causal_mask[:x.size(1), :x.size(1)]
        for layer in self.layers:
            h = layer(h, attn_mask=mask)
        return self.head(self.ln_f(h))

def get_data(p, eq_token, op_token, task_name: str):
    x = torch.arange(p)
    y = torch.arange(1, p)
    x, y = torch.cartesian_prod(x, y).T
    eq = torch.ones_like(x) * eq_token
    op = torch.ones_like(x) * op_token

    if 'plus' in task_name or '+' in task_name:
        result = (x + y) % p
    elif 'minus' in task_name or '-' in task_name:
        result = (x - y) % p
    elif 'div' in task_name:
        res_list = [(xi.item() * pow(yi.item(), p - 2, p)) % p for xi, yi in zip(x, y)]
        result = torch.tensor(res_list)
    elif 'mul' in task_name or '*' in task_name:
        result = (x * y) % p
    else:
        if task_name == 'x+y': result = (x + y) % p
        elif task_name == 'x-y': result = (x - y) % p
        elif task_name == 'x*y': result = (x * y) % p
        else: raise ValueError(f"Unknown task: {task_name}")

    data = torch.stack([x, op, y, eq, result]).T
    return data

@torch.no_grad()
def evaluate_dataset(model, loader, device):
    """Return avg CE loss and accuracy on the dataset."""
    model.eval()
    total_loss = 0.0
    total_correct = 0.0
    total_samples = 0
    for (batch,) in loader:
        batch = batch.to(device)
        inp, target = batch[:, :-1], batch[:, -1]
        logits = model(inp)
        loss = F.cross_entropy(logits[:, -1, :].float(), target, reduction='sum')
        preds = logits[:, -1, :].argmax(-1)
        correct = (preds == target).float().sum()
        total_loss += loss.item()
        total_correct += correct.item()
        total_samples += batch.size(0)
    return total_loss / total_samples, total_correct / total_samples

def parse_step(path: str) -> int:
    m = re.search(r"step(\d+)", os.path.basename(path))
    return int(m.group(1)) if m else -1

def load_checkpoint_into_model(model, ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
    elif isinstance(ckpt, dict) and 'model' in ckpt and isinstance(ckpt['model'], dict):
        model.load_state_dict(ckpt['model'])
    else:
        model.load_state_dict(ckpt)
    return model

# =============================================================================
# 2) Offline Wang–Landau over checkpoint-index random walk
# =============================================================================

def build_energy_bins(E_vals, n_bins=200, pad=1e-6):
    E_min = float(np.min(E_vals))
    E_max = float(np.max(E_vals))
    if abs(E_max - E_min) < 1e-12:
        E_max = E_min + 1e-3
    E_min -= pad
    E_max += pad
    edges = np.linspace(E_min, E_max, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return edges, centers

def bin_index(E, edges):
    # edges length = n_bins+1
    # return idx in [0, n_bins-1]
    idx = int(np.searchsorted(edges, E, side='right') - 1)
    if idx < 0: idx = 0
    if idx >= len(edges) - 1: idx = len(edges) - 2
    return idx

def histogram_flat(H, flatness=0.8, min_visit=1):
    """Standard WL flatness criterion: min(H) >= flatness * mean(H), but only over visited bins."""
    H = np.asarray(H, dtype=np.float64)
    visited = H >= min_visit
    if visited.sum() < max(10, int(0.05 * len(H))):
        return False
    H_v = H[visited]
    mean = float(H_v.mean())
    if mean <= 0:
        return False
    return float(H_v.min()) >= flatness * mean

def wang_landau_on_discrete_states(
    E_vals,
    edges,
    n_levels=20,
    logf_init=1.0,
    logf_min=1e-4,
    sweeps_per_level=5,
    flatness=0.8,
    seed=0,
):
    """
    Discrete WL on a line graph of states {0,...,N-1}.
    Proposal: random-walk to i+1 or i-1 (reflect at ends).
    Acceptance: min(1, exp(logg[bin(curr)] - logg[bin(prop)])).
    Update: logg[bin(curr)] += logf; H[bin(curr)] += 1.
    """
    rng = np.random.default_rng(seed)
    N = len(E_vals)
    n_bins = len(edges) - 1

    logg = np.zeros(n_bins, dtype=np.float64)
    H = np.zeros(n_bins, dtype=np.int64)

    # start somewhere in the middle to avoid edge bias
    state = int(N // 2)

    trace_rows = []
    logf = float(logf_init)

    for level in range(n_levels):
        H[:] = 0
        moves = 0
        accepts = 0

        # we do multiple sweeps at this logf to give histogram chance to flatten
        for sweep in range(sweeps_per_level):
            # one sweep = N proposals
            for _ in range(N):
                moves += 1
                # propose neighbor
                step = 1 if rng.random() < 0.5 else -1
                prop = state + step
                if prop < 0: prop = 1
                if prop >= N: prop = N - 2

                b_curr = bin_index(E_vals[state], edges)
                b_prop = bin_index(E_vals[prop], edges)

                # WL accept
                a = math.exp(min(0.0, logg[b_curr] - logg[b_prop]))  # min(1, exp(...))
                if rng.random() < a:
                    state = prop
                    accepts += 1
                    b_curr = b_prop

                # update DOS + hist at current state
                logg[b_curr] += logf
                H[b_curr] += 1

            # optional early stop if flat
            if histogram_flat(H, flatness=flatness, min_visit=1):
                break

        flat = histogram_flat(H, flatness=flatness, min_visit=1)
        acc_rate = accepts / max(moves, 1)

        trace_rows.append({
            "level": level,
            "logf": logf,
            "flat": int(flat),
            "acc_rate": acc_rate,
            "visited_bins": int((H > 0).sum()),
            "H_min": int(H[H > 0].min()) if (H > 0).any() else 0,
            "H_mean": float(H[H > 0].mean()) if (H > 0).any() else 0.0,
            "H_max": int(H.max()),
        })

        # reduce modification factor
        logf = logf / 2.0
        if logf < logf_min:
            break

    return logg, H, trace_rows

# =============================================================================
# 3) Main: scan checkpoints -> compute energies + (train/test) -> run WL -> save CSVs
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_root", type=str, default="results/checkpoint_transformer_2_4_128",
                        help="Root directory containing task/wd_*/seed*_step*.pt")
    parser.add_argument("--task", type=str, default=None,
                        help="Optional: only process one task folder name (e.g., x-y). If None, process all.")
    parser.add_argument("--wd", type=float, default=None,
                        help="Optional: only process one wd value (e.g., 1.0). If None, process all.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Optional: only process one seed. If None, process all.")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--bins", type=int, default=200)
    parser.add_argument("--eval_bs", type=int, default=2048)
    parser.add_argument("--wl_levels", type=int, default=20)
    parser.add_argument("--wl_logf_init", type=float, default=1.0)
    parser.add_argument("--wl_logf_min", type=float, default=1e-4)
    parser.add_argument("--wl_sweeps_per_level", type=int, default=5)
    parser.add_argument("--wl_flatness", type=float, default=0.8)
    parser.add_argument("--out_dir", type=str, default="/data/zjj/test/results/Data/WLMD")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device)

    # -------------------------------------------------------------------------
    # Discover checkpoint groups: (task, wd, seed) -> list of paths
    # -------------------------------------------------------------------------
    all_pts = glob.glob(os.path.join(args.checkpoint_root, "**", "*.pt"), recursive=True)
    groups = {}
    
    # Define task mapping for folder names
    task_folder_map = {
        'x_plus_y': 'x_plus_y',
        'x_minus_y': 'x-y',
        'x_mul_y': 'x_mul_y',
        'x_div_y': 'x_div_y'
    }
    
    # Reverse map to get canonical task name from folder name
    folder_to_task = {v: k for k, v in task_folder_map.items()}
    
    for pt in all_pts:
        fname = os.path.basename(pt)

        m_seed = re.search(r"seed(\d+)", fname)
        if not m_seed:
            continue
        seed = int(m_seed.group(1))

        # parse wd and task from directory structure: .../<task>/wd_<wd>/seed..pt
        wd = None
        task_name = None
        curr = os.path.dirname(pt)
        root_abs = os.path.abspath(args.checkpoint_root)
        
        # Traverse up to find wd and task folders
        while True:
            if os.path.abspath(curr) == os.path.abspath(os.path.dirname(curr)):
                break
            d = os.path.basename(curr)
            
            # Check for wd folder
            if d.startswith("wd_"):
                try:
                    wd = float(d.replace("wd_", ""))
                    # Task folder should be parent of wd folder
                    task_folder = os.path.basename(os.path.dirname(curr))
                    
                    # Map folder name to canonical task name if possible
                    if task_folder in folder_to_task:
                        task_name = folder_to_task[task_folder]
                    else:
                        # Fallback: use folder name as task name
                        task_name = task_folder
                        
                except:
                    wd = None
                break
            
            if os.path.abspath(curr) == root_abs:
                break
            curr = os.path.dirname(curr)

        if wd is None or task_name is None:
            continue

        if args.task is not None and task_name != args.task:
            continue
        if args.wd is not None and abs(wd - args.wd) > 1e-12:
            continue
        if args.seed is not None and seed != args.seed:
            continue

        key = (task_name, wd, seed)
        groups.setdefault(key, []).append(pt)

    if not groups:
        print("[ERROR] No matching checkpoints found.")
        return

    print(f"Found {len(groups)} experiment groups to process.")
    print(f"Output directory: {args.out_dir}")

    # -------------------------------------------------------------------------
    # Process each group independently
    # -------------------------------------------------------------------------
    p = 97
    num_tokens = p + 2

    for (task_name, wd, seed), files in groups.items():
        files = sorted(files, key=parse_step)
        steps = [parse_step(f) for f in files]
        if len(files) < 5:
            print(f"[WARN] Skip {task_name} wd={wd} seed={seed}: too few checkpoints ({len(files)}).")
            continue

        print(f"\n[GROUP] task={task_name}, wd={wd}, seed={seed}, ckpts={len(files)}")

        # --- Reproduce split exactly like your eval script ---
        torch.manual_seed(seed)
        full_data = get_data(p, p, p + 1, task_name)
        indices = torch.randperm(len(full_data))
        split = int(len(full_data) * 0.5)
        train_data = full_data[indices[:split]]
        test_data  = full_data[indices[split:]]
        train_loader = DataLoader(TensorDataset(train_data), batch_size=args.eval_bs, shuffle=False)
        test_loader  = DataLoader(TensorDataset(test_data),  batch_size=args.eval_bs, shuffle=False)

        # --- Init model ---
        model = Decoder(dim=128, num_layers=2, num_heads=4, num_tokens=num_tokens, seq_len=5).to(device)
        model.eval()

        # ---------------------------------------------------------------------
        # 1) Scan checkpoints -> compute curves + energies
        #    Energy definition: E(w) = log L_train(w)
        # ---------------------------------------------------------------------
        rows_curve = []
        E_vals = np.zeros(len(files), dtype=np.float64)

        for i, ckpt_path in enumerate(tqdm(files, desc="Eval checkpoints")):
            step = steps[i]
            load_checkpoint_into_model(model, ckpt_path, device)

            train_loss, train_acc = evaluate_dataset(model, train_loader, device)
            test_loss,  test_acc  = evaluate_dataset(model, test_loader, device)

            # energy = log L_train(w)
            E = float(np.log(train_loss + 1e-12))
            E_vals[i] = E

            rows_curve.append({
                "step": step,
                "train_loss": float(train_loss),
                "train_acc": float(train_acc),
                "test_loss": float(test_loss),
                "test_acc": float(test_acc),
                "energy_logLtrain": E,
            })

        # Save CSV for Fig.1 (and also used in Fig.5)
        curve_csv = os.path.join(args.out_dir, f"fig1_curves_{task_name}_wd{wd}_seed{seed}.csv")
        with open(curve_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows_curve[0].keys()))
            w.writeheader()
            w.writerows(rows_curve)
        print(f"[OK] Fig.1 curves CSV -> {curve_csv}")

        # ---------------------------------------------------------------------
        # 2) Build 200 bins from observed energies
        # ---------------------------------------------------------------------
        edges, centers = build_energy_bins(E_vals, n_bins=args.bins, pad=1e-6)
        state_bins = np.array([bin_index(e, edges) for e in E_vals], dtype=np.int32)

        # ---------------------------------------------------------------------
        # 3) Wang–Landau DOS estimation on discrete checkpoint states
        # ---------------------------------------------------------------------
        logg, H, trace_rows = wang_landau_on_discrete_states(
            E_vals=E_vals,
            edges=edges,
            n_levels=args.wl_levels,
            logf_init=args.wl_logf_init,
            logf_min=args.wl_logf_min,
            sweeps_per_level=args.wl_sweeps_per_level,
            flatness=args.wl_flatness,
            seed=seed,
        )

        # Fig.2: DOS / entropy curve (S(E)=log g(E) up to const)
        dos_csv = os.path.join(args.out_dir, f"fig2_dos_{task_name}_wd{wd}_seed{seed}.csv")
        with open(dos_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["bin", "E_left", "E_right", "E_center", "log_g", "S_E"])
            for b in range(args.bins):
                w.writerow([b, edges[b], edges[b+1], centers[b], logg[b], logg[b]])
        print(f"[OK] Fig.2 DOS CSV -> {dos_csv}")

        # Fig.3: Final histogram H(E)
        hist_csv = os.path.join(args.out_dir, f"fig3_hist_{task_name}_wd{wd}_seed{seed}.csv")
        with open(hist_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["bin", "E_center", "H"])
            for b in range(args.bins):
                w.writerow([b, centers[b], int(H[b])])
        print(f"[OK] Fig.3 Hist CSV -> {hist_csv}")

        # Fig.3 (optional): WL trace by level (flatness, logf, acc rate)
        trace_csv = os.path.join(args.out_dir, f"fig3_wl_trace_{task_name}_wd{wd}_seed{seed}.csv")
        with open(trace_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(trace_rows[0].keys()))
            w.writeheader()
            w.writerows(trace_rows)
        print(f"[OK] WL trace CSV -> {trace_csv}")

        # ---------------------------------------------------------------------
        # 4) Fig.5: Align dynamics (step) with DOS signal S(E) via bin mapping
        #     Provide per-checkpoint: bin, S(E_bin), H(bin) so you can overlay.
        # ---------------------------------------------------------------------
        fig5_rows = []
        for i, r in enumerate(rows_curve):
            b = int(state_bins[i])
            fig5_rows.append({
                "step": int(r["step"]),
                "energy_logLtrain": float(r["energy_logLtrain"]),
                "bin": b,
                "S_Ebin": float(logg[b]),
                "H_bin": int(H[b]),
                "train_loss": float(r["train_loss"]),
                "train_acc": float(r["train_acc"]),
                "test_loss": float(r["test_loss"]),
                "test_acc": float(r["test_acc"]),
            })
        fig5_csv = os.path.join(args.out_dir, f"fig5_align_{task_name}_wd{wd}_seed{seed}.csv")
        with open(fig5_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(fig5_rows[0].keys()))
            w.writeheader()
            w.writerows(fig5_rows)
        print(f"[OK] Fig.5 alignment CSV -> {fig5_csv}")

        # ---------------------------------------------------------------------
        # 5) Fig.4 (WanD vs SGD) requires at least TWO trajectories.
        #     Here we just emit a placeholder manifest to make plotting scripts uniform.
        #     You can re-run this script with different checkpoint_roots or task/wd to
        #     get another group, then plot fig4 by combining two fig1_curves CSVs.
        # ---------------------------------------------------------------------
        manifest_csv = os.path.join(args.out_dir, f"fig4_manifest_{task_name}_wd{wd}_seed{seed}.csv")
        with open(manifest_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["label", "curve_csv"])
            w.writerow([f"{task_name}_wd{wd}_seed{seed}", curve_csv])
        print(f"[OK] Fig.4 manifest (single-trajectory) -> {manifest_csv}")

    print("\nAll done. CSVs saved under:", args.out_dir)
    print("You can plot:")
    print(" - Fig.1 from fig1_curves_*.csv")
    print(" - Fig.2 from fig2_dos_*.csv")
    print(" - Fig.3 from fig3_hist_*.csv and fig3_wl_trace_*.csv")
    print(" - Fig.4 by combining multiple fig1_curves_*.csv (different optimizers/roots)")
    print(" - Fig.5 from fig5_align_*.csv")


if __name__ == "__main__":
    main()
