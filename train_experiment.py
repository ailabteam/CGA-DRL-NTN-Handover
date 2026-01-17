# train_experiment.py
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

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def make_env(feature_type, scenario, seed):
    def _init():
        env = SatelliteHandoverEnv(k_nearest=5, feature_type=feature_type, scenario=scenario)
        env.reset(seed=seed)
        return Monitor(env)
    return _init

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", type=str, default="PPO", choices=["PPO", "A2C"])
    parser.add_argument("--feature", type=str, default="cga", choices=["cga", "xyz"])
    parser.add_argument("--scenario", type=str, default="static", choices=["static", "random"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=100_000)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    # Setup Logging Path: results/logs/SCENARIO/ALGO_FEATURE/
    set_seed(args.seed)
    run_name = f"{args.algo}_{args.feature}_seed{args.seed}"
    
    # Quan trọng: Tách thư mục theo Scenario
    log_dir = f"./results/logs/{args.scenario}/{args.algo}_{args.feature}/"
    model_dir = f"./results/models/{args.scenario}/{args.algo}_{args.feature}/"
    
    print(f"--- START: {run_name} | Scenario: {args.scenario} ---")
    
    n_envs = 4
    # Truyền scenario vào make_env
    env = make_vec_env(make_env(args.feature, args.scenario, args.seed), n_envs=n_envs)

    model_cls = PPO if args.algo == "PPO" else A2C
    
    model = model_cls(
        "MlpPolicy", 
        env, 
        verbose=0,
        tensorboard_log=log_dir,
        device=f"cuda:{args.gpu}",
        seed=args.seed
    )

    start_time = time.time()
    model.learn(total_timesteps=args.steps, tb_log_name=f"seed_{args.seed}")
    
    os.makedirs(model_dir, exist_ok=True)
    model.save(f"{model_dir}/{run_name}_final")
    
    print(f"--- FINISHED in {time.time()-start_time:.1f}s ---")
    env.close()

if __name__ == "__main__":
    main()
