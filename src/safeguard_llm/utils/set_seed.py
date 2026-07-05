import os
import random
import numpy as np
import torch

#This function was AI generated per description
def set_global_seed(seed: int = 40) -> None:
    # 1. Python standard library
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    
    # 2. NumPy
    np.random.seed(seed)
    
    # 3. PyTorch (Covers CPU and GPU)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # Seeds all GPUs if using multi-GPU
        
        # 4. Configure CuDNN backend for determinism
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    print(f"Global seed set to: {seed}")