import torch
import numpy as np
import pandas as pd
import os
import glob
from sklearn.decomposition import PCA
from tqdm import tqdm

def main():
    # 1. 定义路径
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # 输出路径: results/Data/t-sne (虽然文件名叫 t-sne.py，但根据要求实际执行的是 PCA 投影)
    output_dir = os.path.join(base_dir, "results", "Data", "t-sne")
    os.makedirs(output_dir, exist_ok=True)
    
    tasks = ['x_plus_y', 'x_minus_y', 'x_mul_y', 'x_div_y']
    
    for task in tasks:
        print(f"\n=== Processing Task: {task} ===")
        
        # Map task name to folder name
        folder_name = task
        if task == 'x_minus_y':
            folder_name = 'x-y'
            
        # Checkpoint 路径: results/checkpoint_transformer_2_4_128/{folder_name}/wd_1.0
        checkpoint_dir = os.path.join(base_dir, "results", "checkpoint_transformer_2_4_128", folder_name, "wd_0.0")
        
        print(f"Looking for checkpoints in: {checkpoint_dir}")
        
        # 2. 获取所有 .pt 文件并解析 Step
        pattern = os.path.join(checkpoint_dir, "*.pt")
        files = glob.glob(pattern)
        
        if not files:
            print(f"No checkpoint files found for {task}.")
            continue

        step_file_map = {}
        for fpath in files:
            try:
                fname = os.path.basename(fpath)
                # 格式示例: seed42_step100.pt
                parts = fname.replace('.pt', '').split('_')
                step = None
                seed = None
                for part in parts:
                    if part.startswith('step'):
                        step = int(part.replace('step', ''))
                    if part.startswith('seed'):
                        seed = int(part.replace('seed', ''))
                
                # 这里我们只取 seed=42 的轨迹，保证轨迹连贯性
                if step is not None and seed == 42:
                    step_file_map[step] = fpath
            except Exception:
                continue
                
        if not step_file_map:
            print(f"No valid checkpoints found for seed 42 in {task}.")
            continue

        sorted_steps = sorted(step_file_map.keys())
        max_step = sorted_steps[-1]
        print(f"Found {len(sorted_steps)} checkpoints for seed 42. Max step: {max_step}")

        # 3. 锁定基准坐标系：加载最后一步 (Step 100,000)
        last_ckpt_path = step_file_map[max_step]
        print(f"Loading reference checkpoint: {last_ckpt_path}")
        
        try:
            ckpt = torch.load(last_ckpt_path, map_location='cpu')
            state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
            
            # 提取嵌入层权重 [Vocab_Size, Dim]
            ref_emb = state_dict['token_embeddings.weight'].float().numpy()
            
            # 拟合 PCA
            pca = PCA(n_components=2)
            pca.fit(ref_emb)
            
            print(f"PCA Fitted on Step {max_step}.")
            print(f"Explained Variance Ratio: {pca.explained_variance_ratio_}")
            print(f"Total Explained Variance: {np.sum(pca.explained_variance_ratio_):.4f}")
            
        except Exception as e:
            print(f"Error loading reference checkpoint: {e}")
            continue

        # 4. 投影历史轨迹
        all_data = []
        
        print(f"Projecting historical trajectories for {task}...")
        for step in tqdm(sorted_steps, desc=f"PCA Projection ({task})"):
            fpath = step_file_map[step]
            try:
                ckpt = torch.load(fpath, map_location='cpu')
                state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
                
                # 获取当前步的嵌入
                curr_emb = state_dict['token_embeddings.weight'].float().numpy()
                
                # 使用基准 PCA 进行投影
                projected = pca.transform(curr_emb) # [Vocab_Size, 2]
                
                # 记录数据
                num_tokens = projected.shape[0]
                for token_idx in range(num_tokens):
                    all_data.append({
                        'step': step,
                        'token_id': token_idx,
                        'pc1': projected[token_idx, 0],
                        'pc2': projected[token_idx, 1]
                    })
                    
            except Exception as e:
                print(f"[Warn] Failed to process step {step}: {e}")

        # 5. 保存结果
        if all_data:
            df = pd.DataFrame(all_data)
            save_path = os.path.join(output_dir, f"pca_trajectory_{task}_wd0.csv")
            df.to_csv(save_path, index=False)
            print(f"\nTrajectory data saved to:\n{save_path}")
            print(f"Total data points: {len(df)}")
        else:
            print(f"No data generated for {task}.")

if __name__ == "__main__":
    main()
