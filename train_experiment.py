import argparse
import os
import time
import random
import numpy as np
import torch
import gymnasium as gym
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor

from src.envs.ntn_env import SatelliteHandoverEnv

# Hàm thiết lập seed toàn cục (Phase 3 Checklist)
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def make_env(feature_type, seed):
    def _init():
        env = SatelliteHandoverEnv(k_nearest=5, feature_type=feature_type)
        env.reset(seed=seed)
        return Monitor(env)
    return _init

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", type=str, default="PPO", choices=["PPO", "A2C"], help="RL Algorithm")
    parser.add_argument("--feature", type=str, default="cga", choices=["cga", "xyz"], help="Feature Representation")
    parser.add_argument("--seed", type=int, default=42, help="Random Seed")
    parser.add_argument("--steps", type=int, default=100_000, help="Total Timesteps")
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID")
    args = parser.parse_args()

    # 1. Setup Environment & Logging
    set_seed(args.seed)
    run_name = f"{args.algo}_{args.feature}_seed{args.seed}"
    log_dir = f"./results/logs/{args.algo}_{args.feature}/"
    model_dir = f"./results/models/{args.algo}_{args.feature}/"
    
    print(f"--- START EXPERIMENT: {run_name} ---")
    print(f"Device: cuda:{args.gpu} | Feature: {args.feature} | Seed: {args.seed}")

    # Chạy song song 4 envs
    n_envs = 4
    env = make_vec_env(make_env(args.feature, args.seed), n_envs=n_envs)

    # 2. Setup Model
    model_cls = PPO if args.algo == "PPO" else A2C
    
    model = model_cls(
        "MlpPolicy", 
        env, 
        verbose=0, # Tắt log console để đỡ rối khi chạy batch
        tensorboard_log=log_dir,
        device=f"cuda:{args.gpu}",
        seed=args.seed
    )

    # 3. Training
    start_time = time.time()
    model.learn(total_timesteps=args.steps, tb_log_name=f"seed_{args.seed}")
    
    # 4. Save Final
    os.makedirs(model_dir, exist_ok=True)
    model.save(f"{model_dir}/{run_name}_final")
    
    print(f"--- FINISHED: {run_name} in {time.time()-start_time:.1f}s ---")
    env.close()

if __name__ == "__main__":
    main()
