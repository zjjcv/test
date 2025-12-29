import torch
import numpy as np
import pandas as pd
import os
import glob
from sklearn.cluster import KMeans
from tqdm import tqdm
import sys

# Try to import pybdm
try:
    from pybdm import BDM
except ImportError:
    print("Error: pybdm library not found. Please install it using: pip install pybdm")
    BDM = None

def z_score_normalization(weights):
    """
    Normalize weights using Z-Score (Standardization).
    """
    mean = np.mean(weights)
    std = np.std(weights)
    if std == 0:
        return np.zeros_like(weights)
    return (weights - mean) / std

def quantize_weights(weights, n_clusters=4):
    """
    Quantize continuous weights into discrete symbols (0 to n_clusters-1) using K-Means.
    """
    # Reshape to 1D for clustering
    shape = weights.shape
    weights_flat = weights.reshape(-1, 1)
    
    # K-Means clustering
    # n_init='auto' or explicit integer to suppress warnings
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(weights_flat)
    
    # Reshape back to original shape
    quantized = labels.reshape(shape)
    return quantized

import gzip

# ... (imports)

def calculate_complexity(quantized_weights, bdm_solver=None, use_gzip=False):
    """
    Calculate Complexity: BDM or Gzip (as proxy).
    """
    # Ensure integer type
    data = np.ascontiguousarray(quantized_weights.astype(np.int8))
    
    if use_gzip or bdm_solver is None:
        # Gzip Compression Size as KC proxy
        # Convert to bytes
        return len(gzip.compress(data.tobytes()))
    
    try:
        val = bdm_solver.bdm(data)
        if val == 0:
            # Fallback if BDM returns 0 (unexpected)
            return len(gzip.compress(data.tobytes()))
        return val
    except Exception as e:
        # Fallback
        return len(gzip.compress(data.tobytes()))

def main():
    # ...
    
    # 2. Initialize BDM
    bdm = None
    use_gzip = False
    
    if BDM is not None:
        print("Initializing BDM solver...")
        n_clusters = 4
        try:
            bdm = BDM(ndim=2, alphabet=n_clusters)
            # Self-test
            test_mat = np.random.randint(0, n_clusters, (20, 20), dtype=int)
            test_score = bdm.bdm(test_mat)
            print(f"BDM Test Score: {test_score}")
            if test_score == 0:
                print("BDM returned 0 for random matrix. Switching to Gzip proxy.")
                use_gzip = True
        except Exception as e:
            print(f"BDM Init failed ({e}). Switching to Gzip proxy.")
            use_gzip = True
    else:
        print("pybdm not found. Using Gzip proxy.")
        use_gzip = True

    # ...
    
    # In loop:
    # kc = calculate_complexity(quantized, bdm, use_gzip)


def process_2d_matrix(weights, bdm_solver, n_clusters=4, use_gzip=False):
    """
    Helper to process a single 2D weight matrix: Normalize -> Quantize -> Complexity.
    Returns: (kc_value, quantized_matrix, normalized_matrix)
    """
    # A. Z-Score Normalization
    norm_weights = z_score_normalization(weights)
    
    # B. Adaptive Quantization (K-Means)
    quantized = quantize_weights(norm_weights, n_clusters=n_clusters)
    
    # C. Complexity Calculation
    kc = calculate_complexity(quantized, bdm_solver, use_gzip=use_gzip)
    
    return kc, quantized, norm_weights

def main():
    if BDM is None:
        print("Exiting due to missing pybdm library.")
        return

    # 1. Define Paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Target Checkpoint Directory: results/checkpoint_transformer_2_4_128/x_minus_y/wd_1.0
    # Adjust this path if your checkpoints are located elsewhere
    checkpoint_dir = os.path.join(base_dir, "results", "checkpoint_transformer_2_4_128", "x-y", "wd_1.0")
    
    # Output Directory
    output_dir = os.path.join(base_dir, "results", "Data", "BDM")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Looking for checkpoints in: {checkpoint_dir}")
    
    # 2. Initialize BDM
    bdm = None
    use_gzip = False
    n_clusters = 4
    
    if BDM is not None:
        print("Initializing BDM solver...")
        try:
            bdm = BDM(ndim=2, alphabet=n_clusters)
            # Self-test
            test_mat = np.random.randint(0, n_clusters, (20, 20), dtype=int)
            test_score = bdm.bdm(test_mat)
            print(f"BDM Test Score: {test_score}")
            if test_score == 0:
                print("BDM returned 0 for random matrix. Switching to Gzip proxy.")
                use_gzip = True
        except TypeError:
             # Fallback
            print("Warning: BDM alphabet param issue. Trying default.")
            try:
                bdm = BDM(ndim=2)
                test_score = bdm.bdm(np.random.randint(0, 2, (20, 20), dtype=int))
                if test_score == 0: use_gzip = True
            except:
                use_gzip = True
        except Exception as e:
            print(f"BDM Init failed ({e}). Switching to Gzip proxy.")
            use_gzip = True
    else:
        print("pybdm not found. Using Gzip proxy.")
        use_gzip = True

    if use_gzip:
        print("Using Gzip Compression Size as Kolmogorov Complexity Proxy.")
    
    # 3. Find and Sort Checkpoints
    pattern = os.path.join(checkpoint_dir, "*.pt")
    files = glob.glob(pattern)
    
    # Filter for seed 42 (consistent trajectory)
    checkpoints = []
    for fpath in files:
        fname = os.path.basename(fpath)
        if "seed42" in fname:
            try:
                # Expected format: seed42_step100.pt or similar
                parts = fname.replace('.pt', '').split('_')
                step = None
                for part in parts:
                    if part.startswith('step'):
                        step = int(part.replace('step', ''))
                        break
                
                if step is not None:
                    checkpoints.append((step, fpath))
            except Exception:
                pass
    
    # Sort by step
    checkpoints.sort(key=lambda x: x[0])
    
    if not checkpoints:
        print("No checkpoints found matching 'seed42' and 'step*'.")
        return

    print(f"Found {len(checkpoints)} checkpoints. Starting processing...")

    # 4. Define Sampling Strategy for Heatmaps
    # Fixed steps as requested: 100, 1000, 10000, 100000
    target_steps = [100, 1000, 10000, 100000]
    available_steps = {c[0] for c in checkpoints}
    sample_steps = set()
    
    for t in target_steps:
        if t in available_steps:
            sample_steps.add(t)
    
    print(f"Sampling quantized weights for steps: {sorted(list(sample_steps))}")
    
    if not sample_steps:
        print(f"Warning: No target steps found in checkpoints. Available steps (first 10): {sorted(list(available_steps))[:10]}")
    
    results = []
    quantized_samples = {}
    normalized_samples = {}
    
    # 5. Process Checkpoints
    for step, fpath in tqdm(checkpoints, desc="Calculating KC"):
        try:
            # Load model
            model_data = torch.load(fpath, map_location='cpu')
            if isinstance(model_data, dict) and 'model' in model_data:
                state_dict = model_data['model']
            else:
                state_dict = model_data
            
            total_kc = 0
            
            # Iterate through all parameters
            for name, param in state_dict.items():
                if not isinstance(param, torch.Tensor):
                    continue
                
                # DEBUG: Print layer names for the first checkpoint to help debug selection
                if step == checkpoints[0][0] and not results: 
                    # Only print for the very first file processed
                    print(f"DEBUG: Layer found: {name} | Shape: {param.shape} | Dim: {param.dim()}")

                # Process 2D weight matrices (Linear, Embedding)
                if param.dim() == 2:
                    weights = param.detach().cpu().numpy()
                    kc, quantized, norm_weights = process_2d_matrix(weights, bdm, n_clusters=n_clusters, use_gzip=use_gzip)
                    total_kc += kc
                    
                    # Save Sample
                    if step in sample_steps:
                        sample_key = f"step_{step}_{name}"
                        quantized_samples[sample_key] = quantized
                        normalized_samples[sample_key] = norm_weights

                # Process 3D weight matrices (e.g. Multi-head Attention weights stored as [num_heads, d_head, d_model])
                elif param.dim() == 3:
                    # Iterate over the first dimension (e.g. heads)
                    for i in range(param.shape[0]):
                        weights = param[i].detach().cpu().numpy()
                        kc, quantized, norm_weights = process_2d_matrix(weights, bdm, n_clusters=n_clusters, use_gzip=use_gzip)
                        total_kc += kc
                        
                        # Save Sample (append index to name)
                        if step in sample_steps:
                            sample_key = f"step_{step}_{name}_dim0_{i}"
                            quantized_samples[sample_key] = quantized
                            normalized_samples[sample_key] = norm_weights
                
                # Process 4D weight matrices (e.g. Conv2D [out, in, k, k]) - Treat as collection of 2D filters?
                # Or just skip for now unless requested.
                # For now, we focus on 2D and 3D.

            results.append({'Step': step, 'Total_BDM_Value': total_kc})

            results.append({'Step': step, 'Total_BDM_Value': total_kc})
            
        except Exception as e:
            print(f"Error processing step {step}: {e}")
            continue

    # 6. Save Results
    # Trajectory CSV
    df_results = pd.DataFrame(results)
    traj_path = os.path.join(output_dir, "bdm_trajectory.csv")
    df_results.to_csv(traj_path, index=False)
    print(f"\nSaved BDM trajectory to: {traj_path}")
    
    # Quantized Samples CSV
    samples_dfs = []
    for key, matrix in quantized_samples.items():
        # Parse key: step_{step}_{name}
        parts = key.split('_')
        step_val = parts[1]
        layer_name = "_".join(parts[2:])
        
        # Create DataFrame from matrix
        df_mat = pd.DataFrame(matrix)
        # Rename columns to col_0, col_1, ...
        df_mat.columns = [f'col_{i}' for i in range(df_mat.shape[1])]
        
        # Add metadata columns
        df_mat['step'] = step_val
        df_mat['layer'] = layer_name
        df_mat['row_idx'] = range(df_mat.shape[0])
        
        # Reorder columns to put metadata first
        cols = ['step', 'layer', 'row_idx'] + [c for c in df_mat.columns if c.startswith('col_')]
        df_mat = df_mat[cols]
        
        samples_dfs.append(df_mat)

    if samples_dfs:
        full_samples_df = pd.concat(samples_dfs, ignore_index=True)
        samples_path = os.path.join(output_dir, "quantized_weights_samples.csv")
        full_samples_df.to_csv(samples_path, index=False)
        print(f"Saved quantized weight samples to: {samples_path}")
    else:
        print("No quantized samples collected.")
    
    if normalized_samples:
        npz_path = os.path.join(output_dir, "normalized_weights_samples.npz")
        np.savez_compressed(npz_path, **normalized_samples)
        print(f"Saved normalized weight samples to: {npz_path}")

    print("Done.")

if __name__ == "__main__":
    main()
