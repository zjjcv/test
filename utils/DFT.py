import torch
import numpy as np
import pandas as pd
import os
import glob
from tqdm import tqdm

def calculate_gini(array):
    """
    计算数组的基尼系数 (Gini Coefficient)。
    衡量频谱的稀疏性/不平等度。
    """
    # 确保为非负
    array = np.abs(array)
    # 处理全0情况
    if np.sum(array) == 0:
        return 0
    
    array = np.sort(array)
    index = np.arange(1, array.shape[0] + 1)
    n = array.shape[0]
    
    return ((np.sum((2 * index - n - 1) * array)) / (n * np.sum(array)))

def calculate_ipr(array):
    """
    计算逆参与率 (Inverse Participation Ratio, IPR)。
    衡量能量在频域的集中程度。
    IPR = sum(x^4) / (sum(x^2))^2
    """
    # 确保为非负（如果是复数频谱，取模）
    mag = np.abs(array)
    
    sum_sq = np.sum(mag**2)
    if sum_sq == 0:
        return 0
        
    sum_fourth = np.sum(mag**4)
    return sum_fourth / (sum_sq**2)

def main():
    # 1. 定义路径
    # 假设脚本位于 utils/DFT.py，向上两级是项目根目录
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # 输出路径: results/Data/DFT
    output_dir = os.path.join(base_dir, "results", "Data", "DFT")
    os.makedirs(output_dir, exist_ok=True)
    
    tasks = ['x_plus_y', 'x_minus_y', 'x_mul_y', 'x_div_y']
    
    for task in tasks:
        print(f"\n=== Processing Task: {task} ===")
        
        # Map task name to folder name if necessary
        folder_name = task
        if task == 'x_minus_y':
            folder_name = 'x-y'
            
        # Checkpoint 路径: results/checkpoint_transformer_2_4_128/{task}/wd_1.0
        checkpoint_dir = os.path.join(base_dir, "results", "checkpoint_transformer_2_4_128", folder_name, "wd_0.0")
        
        print(f"Looking for checkpoints in: {checkpoint_dir}")
        
        # 2. 获取所有 .pt 文件
        pattern = os.path.join(checkpoint_dir, "*.pt")
        files = glob.glob(pattern)
        
        if not files:
            print(f"No checkpoint files found for {task}. Please check the path.")
            continue

        print(f"Found {len(files)} checkpoints. Starting analysis...")
        
        results = []
        
        # 3. 遍历处理
        for fpath in tqdm(files, desc=f"DFT Analysis ({task})"):
            try:
                # 解析文件名获取 step 和 seed
                # 格式示例: seed42_step100.pt
                fname = os.path.basename(fpath)
                name_parts = fname.replace('.pt', '').split('_')
                
                seed = None
                step = None
                
                for part in name_parts:
                    if part.startswith('seed'):
                        seed = int(part.replace('seed', ''))
                    elif part.startswith('step'):
                        step = int(part.replace('step', ''))
                
                if seed is None or step is None:
                    print(f"[Warn] Could not parse seed/step from {fname}, skipping.")
                    continue
                
                # 加载模型权重 (使用 CPU 以节省显存)
                ckpt = torch.load(fpath, map_location='cpu')
                
                # 提取嵌入层权重
                # 检查 state_dict 结构
                if 'model_state_dict' in ckpt:
                    state_dict = ckpt['model_state_dict']
                else:
                    state_dict = ckpt
                    
                if 'token_embeddings.weight' not in state_dict:
                    print(f"[Warn] 'token_embeddings.weight' not found in {fname}, skipping.")
                    continue
                    
                # 获取权重矩阵 (Vocab Size x Dim)
                # 通常是 [99, 128]
                emb_weights = state_dict['token_embeddings.weight'].float().numpy()
                
                # 4. 执行 DFT 分析
                # 对整个矩阵进行二维 FFT
                fft_vals = np.fft.fft2(emb_weights)
                fft_mag = np.abs(fft_vals)
                
                # 展平以计算统计指标
                fft_flat = fft_mag.flatten()
                
                # 计算指标
                gini = calculate_gini(fft_flat)
                ipr = calculate_ipr(fft_flat)
                
                results.append({
                    'step': step,
                    'seed': seed,
                    'gini': gini,
                    'ipr': ipr
                })
                
            except Exception as e:
                print(f"[Error] Failed to process {fpath}: {e}")
                
        # 5. 保存结果
        if results:
            df = pd.DataFrame(results)
            # 按 seed 和 step 排序
            df = df.sort_values(by=['seed', 'step'])
            
            # Include task in filename
            save_path = os.path.join(output_dir, f"dft_analysis_{task}_wd0.csv")
            df.to_csv(save_path, index=False)
            print(f"\nAnalysis complete for {task}. Results saved to:\n{save_path}")
            print(f"Total records: {len(df)}")
            print(df.head())
        else:
            print(f"No results generated for {task}.")

if __name__ == "__main__":
    main()
