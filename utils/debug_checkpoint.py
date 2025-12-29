import torch
import os
import sys

# Path to a sample checkpoint
fpath = "/data/zjj/test/results/checkpoint_transformer_2_4_128/x-y/wd_1.0/seed42_step100.pt"

if not os.path.exists(fpath):
    print(f"File not found: {fpath}")
    # Try to find any .pt file in that dir
    dir_path = os.path.dirname(fpath)
    if os.path.exists(dir_path):
        files = [f for f in os.listdir(dir_path) if f.endswith('.pt') and 'seed42' in f]
        if files:
            fpath = os.path.join(dir_path, files[0])
            print(f"Using alternative file: {fpath}")
        else:
            print("No seed42 .pt files found in directory.")
            sys.exit(1)
    else:
        print(f"Directory not found: {dir_path}")
        sys.exit(1)

print(f"Inspecting: {fpath}")

try:
    model_data = torch.load(fpath, map_location='cpu')
    print(f"Type of loaded data: {type(model_data)}")
    
    state_dict = None
    if isinstance(model_data, dict):
        print(f"Keys in loaded dict: {model_data.keys()}")
        if 'model' in model_data:
            state_dict = model_data['model']
            print("Found 'model' key.")
        else:
            state_dict = model_data
            print("Using loaded dict as state_dict.")
    else:
        state_dict = model_data
        print("Using loaded object as state_dict.")

    if state_dict:
        print(f"Number of items in state_dict: {len(state_dict)}")
        for name, param in state_dict.items():
            if isinstance(param, torch.Tensor):
                print(f"Param: {name} | Shape: {param.shape} | Dim: {param.dim()} | Type: {param.dtype}")
            else:
                print(f"Item: {name} | Type: {type(param)}")
    else:
        print("state_dict is empty or None.")

except Exception as e:
    print(f"Error loading checkpoint: {e}")
