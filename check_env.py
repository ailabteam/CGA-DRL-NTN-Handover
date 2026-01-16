import sys
import torch
import clifford
import gymnasium as gym
import stable_baselines3
import numpy as np

def print_status(component, status, detail=""):
    print(f"[{component}]".ljust(15) + f": {status} {detail}")

print("-" * 50)
print(f"Python Version : {sys.version.split()[0]}")
print("-" * 50)

# 1. Kiểm tra PyTorch & GPU
try:
    gpu_avail = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if gpu_avail else "N/A"
    print_status("PyTorch", torch.__version__)
    print_status("CUDA/GPU", "OK" if gpu_avail else "FAIL", f"({gpu_name})")
    
    # Test tensor trên GPU
    if gpu_avail:
        x = torch.rand(1000, 1000).cuda()
        y = x @ x
        print_status("GPU Compute", "OK", "(Matrix Mul Success)")
except Exception as e:
    print_status("GPU Compute", "FAIL", str(e))

# 2. Kiểm tra Clifford (CGA)
try:
    from clifford.g3c import e1, e2, e3
    v = e1 + e2
    print_status("Clifford", clifford.__version__, f"(Vector check: {v})")
except ImportError:
    print_status("Clifford", "NOT INSTALLED")
except Exception as e:
    print_status("Clifford", "ERROR", str(e))

# 3. Kiểm tra RL (Gym + SB3)
try:
    print_status("Gymnasium", gym.__version__)
    print_status("SB3", stable_baselines3.__version__)
    
    # Test tạo môi trường đơn giản
    env = gym.make("CartPole-v1")
    env.reset()
    print_status("Env Create", "OK", "(CartPole-v1)")
except Exception as e:
    print_status("RL Libs", "FAIL", str(e))

print("-" * 50)
