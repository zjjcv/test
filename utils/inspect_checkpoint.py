import torch
import os
import sys

fpath = "test/results/checkpoint_transformer_2_4_128/x-y/wd_1.0/seed42_step100.pt"
if not os.path.exists(fpath):
    print(f"File not found: {fpath}")
    sys.exit(1)

try:
    model_data = torch.load(fpath, map_location='cpu')
    if isinstance(model_data, dict) and 'model' in model_data:
        state_dict = model_data['model']
    else:
        state_dict = model_data

    print(f"State dict keys: {len(state_dict)}")
    for name, param in state_dict.items():
        if isinstance(param, torch.Tensor):
            print(f"{name}: {param.shape}, dim={param.dim()}")
        else:
            print(f"{name}: {type(param)}")
            
except Exception as e:
    print(f"Error: {e}")
