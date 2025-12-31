import torch
import numpy as np
import pandas as pd
import os
import glob
from tqdm import tqdm
import sys
import warnings
import gzip
from collections import Counter

# Suppress warnings
warnings.filterwarnings("ignore")

# Try to import pybdm
try:
    from pybdm import BDM
    PYBDM_AVAILABLE = True
except ImportError:
    print("Warning: pybdm library not found. Using Gzip fallback.")
    PYBDM_AVAILABLE = False

def quantile_binning(weights):
    """
    Discretize weights using quantile-based hard binning (0-3).
    bins = np.quantile(weights, [0.25, 0.5, 0.75])
    """
    # Handle edge case of constant weights
    if np.min(weights) == np.max(weights):
        return np.zeros_like(weights, dtype=int)
        
    quantiles = np.quantile(weights, [0.25, 0.5, 0.75])
    # np.digitize returns 0 for x < q1, 1 for q1 <= x < q2, etc.
    # bins: (-inf, q1), [q1, q2), [q2, q3), [q3, inf) -> 0, 1, 2, 3
    return np.digitize(weights, quantiles)

def calculate_block_bdm(matrix, bdm_solver, block_size=(4, 4), use_gzip=False):
    """
    Calculate BDM using non-overlapping block decomposition + frequency aggregation.
    Formula: Sum_{unique_block u} ( K(u) + log2(count_u) )
    """
    h, w = matrix.shape
    bh, bw = block_size
    
    # Pad matrix to be divisible by block size
    pad_h = (bh - h % bh) % bh
    pad_w = (bw - w % bw) % bw
    if pad_h > 0 or pad_w > 0:
        matrix = np.pad(matrix, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=0)
    
    h_pad, w_pad = matrix.shape
    
    # Extract blocks
    # Reshape to (h//bh, bh, w//bw, bw) and swap axes to (h//bh, w//bw, bh, bw)
    # Then reshape to (-1, bh, bw)
    n_rows = h_pad // bh
    n_cols = w_pad // bw
    
    blocks = matrix.reshape(n_rows, bh, n_cols, bw).swapaxes(1, 2).reshape(-1, bh, bw)
    
    # Convert to tuples for hashing/counting
    # Flatten each block to tuple
    block_tuples = [tuple(b.flatten()) for b in blocks]
    counts = Counter(block_tuples)
    
    total_complexity = 0
    
    # Calculate complexity for each unique block
    for block_tuple, count in counts.items():
        # Reconstruct block
        block_arr = np.array(block_tuple, dtype=np.int8).reshape(bh, bw)
        
        k_u = 0
        if use_gzip or bdm_solver is None:
            # Gzip proxy
            k_u = len(gzip.compress(block_arr.tobytes()))
        else:
            try:
                # pybdm expects integer array
                # Ensure it's the right type/shape
                k_u = bdm_solver.bdm(block_arr)
                if k_u == 0: # Fallback if 0
                     k_u = len(gzip.compress(block_arr.tobytes()))
            except:
                k_u = len(gzip.compress(block_arr.tobytes()))
        
        # Add entropy term: log2(count)
        # If a block repeats N times, we describe it once (K(u)) and specify N positions?
        # The standard BDM aggregation for a dataset is sum(BDM(u) + log2(n_u))
        term = k_u + np.log2(count)
        total_complexity += term
        
    return total_complexity

def process_layer(weights, bdm_solver, use_gzip=False):
    """
    Process a single weight matrix: Quantile Binning -> Block BDM.
    """
    # 1. Quantile Binning (Hard Discretization)
    quantized = quantile_binning(weights)
    
    # 2. Block BDM Calculation
    kc = calculate_block_bdm(quantized, bdm_solver, block_size=(4, 4), use_gzip=use_gzip)
    
    return kc, quantized

def find_best_tensor_dict(obj, max_depth=6):
    """
    在嵌套 dict/list/tuple 中递归寻找：包含最多 torch.Tensor 的 dict。
    返回 (best_dict, tensor_count)；若找不到返回 (None, 0)
    """
    best = (None, 0)

    def count_tensors_in_dict(d):
        return sum(1 for v in d.values() if isinstance(v, torch.Tensor))

    def dfs(x, depth):
        nonlocal best
        if depth > max_depth:
            return
        if isinstance(x, dict):
            c = count_tensors_in_dict(x)
            if c > best[1]:
                best = (x, c)

            # 优先沿常见 key 深入
            priority_keys = [
                "model_state_dict", "state_dict", "model", "module", "net",
                "params", "weights", "optimizer", "state", "ema"
            ]
            for k in priority_keys:
                if k in x:
                    dfs(x[k], depth + 1)

            # 也遍历所有值
            for v in x.values():
                dfs(v, depth + 1)

        elif isinstance(x, (list, tuple)):
            for v in x:
                dfs(v, depth + 1)

    dfs(obj, 0)
    return best

def extract_state_dict(ckpt):
    # 1) 常见扁平格式直接命中
    if isinstance(ckpt, dict):
        for k in ["model_state_dict", "state_dict"]:
            if k in ckpt and isinstance(ckpt[k], dict):
                sd = ckpt[k]
                if any(isinstance(v, torch.Tensor) for v in sd.values()):
                    return sd

        # 2) 常见深层结构
        for k in ["model", "module", "net", "state"]:
            if k in ckpt:
                sd, cnt = find_best_tensor_dict(ckpt[k])
                if cnt > 0:
                    return sd

    # 3) 递归兜底：全对象搜索
    sd, cnt = find_best_tensor_dict(ckpt)
    return sd

def main():
    # 1. Configuration
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(base_dir, "results", "Data", "BDM")
    os.makedirs(output_dir, exist_ok=True)
    
    tasks = ['x_plus_y', 'x_minus_y', 'x_mul_y', 'x_div_y']
    
    # 2. Initialize BDM Solver
    bdm = None
    use_gzip = False
    n_clusters = 4 # 4 bins -> alphabet size 4
    
    if PYBDM_AVAILABLE:
        print(f"Initializing BDM solver with alphabet={n_clusters}...")
        try:
            bdm = BDM(ndim=2, alphabet=n_clusters)
            # Self-test
            test_mat = np.random.randint(0, n_clusters, (4, 4), dtype=int)
            test_score = bdm.bdm(test_mat)
            print(f"BDM Test Score (4x4): {test_score}")
            if test_score == 0:
                print("BDM returned 0. Switching to Gzip proxy.")
                use_gzip = True
        except TypeError:
            print("Warning: BDM alphabet param not supported. Using default.")
            try:
                bdm = BDM(ndim=2)
            except:
                use_gzip = True
        except Exception as e:
            print(f"BDM Init failed: {e}. Using Gzip proxy.")
            use_gzip = True
    else:
        print("pybdm not found. Using Gzip proxy.")
        use_gzip = True

    for task in tasks:
        print(f"\n=== Processing Task: {task} ===")
        
        # Map task name to folder name if necessary
        folder_name = task
        if task == 'x_minus_y':
            folder_name = 'x-y'
            
        checkpoint_dir = os.path.join(base_dir, "results", "checkpoint_transformer_2_4_128", folder_name, "wd_1.0")
        
        print(f"Input Directory: {checkpoint_dir}")
        print(f"Output Directory: {output_dir}")

        # 3. Find Checkpoints
        pattern = os.path.join(checkpoint_dir, "*.pt")
        files = glob.glob(pattern)
        
        checkpoints = []
        for fpath in files:
            fname = os.path.basename(fpath)
            if "seed42" in fname:
                try:
                    parts = fname.replace('.pt', '').split('_')
                    step = None
                    for part in parts:
                        if part.startswith('step'):
                            step = int(part.replace('step', ''))
                            break
                    if step is not None:
                        checkpoints.append((step, fpath))
                except:
                    pass
        
        checkpoints.sort(key=lambda x: x[0])
        
        if not checkpoints:
            print(f"No checkpoints found for task {task}.")
            continue

        print(f"Found {len(checkpoints)} checkpoints for {task}.")

        # 4. Define Sampling Steps
        target_sample_steps = [100, 1000, 10000, 100000]
        
        results = []
        quantized_samples = {}
        
        # 5. Process Loop
        for step, fpath in tqdm(checkpoints, desc=f"Processing {task}"):
            try:
                model_data = torch.load(fpath, map_location="cpu")
                state_dict = extract_state_dict(model_data)
                
                tensor_cnt = 0
                mat2_cnt = 0
                mat3_cnt = 0

                if state_dict is None:
                    print(f"[Step {step}] No tensor dict found in checkpoint: {fpath}")
                    continue

                tensor_cnt = sum(isinstance(v, torch.Tensor) for v in state_dict.values())
                mat2_cnt = sum(isinstance(v, torch.Tensor) and v.dim()==2 for v in state_dict.values())
                mat3_cnt = sum(isinstance(v, torch.Tensor) and v.dim()==3 for v in state_dict.values())
                # print(f"[Step {step}] tensors={tensor_cnt}, 2D={mat2_cnt}, 3D={mat3_cnt}")

                total_kc = 0
                layer_count = 0
                
                for name, param in state_dict.items():
                    if not isinstance(param, torch.Tensor):
                        continue
                    
                    # Process 2D matrices
                    if param.dim() == 2:
                        weights = param.detach().cpu().numpy()
                        kc, quantized = process_layer(weights, bdm, use_gzip=use_gzip)
                        total_kc += kc
                        layer_count += 1
                        
                        if step in target_sample_steps:
                            quantized_samples[f"step_{step}_{name}"] = quantized

                    # Process 3D matrices (Heads)
                    elif param.dim() == 3:
                        for i in range(param.shape[0]):
                            weights = param[i].detach().cpu().numpy()
                            kc, quantized = process_layer(weights, bdm, use_gzip=use_gzip)
                            total_kc += kc
                            
                            if step in target_sample_steps:
                                quantized_samples[f"step_{step}_{name}_dim0_{i}"] = quantized

                results.append({
                    'Step': step,
                    'Total_BDM_Value': total_kc,
                    'Layer_Count': layer_count
                })
                
            except Exception as e:
                print(f"Error step {step}: {e}")

        # 6. Save Results
        if results:
            df_results = pd.DataFrame(results)
            traj_path = os.path.join(output_dir, f"bdm_trajectory_{task}.csv")
            df_results.to_csv(traj_path, index=False)
            print(f"Saved trajectory to {traj_path}")
            print(df_results.head())

        # 7. Save Samples
        if quantized_samples:
            samples_dfs = []
            for key, matrix in quantized_samples.items():
                parts = key.split('_')
                step_val = parts[1]
                layer_name = "_".join(parts[2:])
                
                df_mat = pd.DataFrame(matrix)
                df_mat.columns = [f'col_{i}' for i in range(df_mat.shape[1])]
                df_mat['step'] = step_val
                df_mat['layer'] = layer_name
                df_mat['row_idx'] = range(df_mat.shape[0])
                
                cols = ['step', 'layer', 'row_idx'] + [c for c in df_mat.columns if c.startswith('col_')]
                samples_dfs.append(df_mat[cols])
            
            if samples_dfs:
                full_samples_df = pd.concat(samples_dfs, ignore_index=True)
                samples_path = os.path.join(output_dir, f"quantized_weights_samples_{task}.csv")
                full_samples_df.to_csv(samples_path, index=False)
                print(f"Saved samples to {samples_path}")

    print("Done.")

if __name__ == "__main__":
    main()