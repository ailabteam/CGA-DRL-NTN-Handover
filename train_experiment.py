import argparse
import os
import time
import random
import numpy as np
import torch
import gymnasium as gym
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from src.envs.ntn_env import SatelliteHandoverEnv

# FIX: Class wrapper cập nhật logic info
class MetricsWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.ep_ho = 0
        self.ep_tp = 0
        self.ep_outage = 0
        
    def reset(self, **kwargs):
        self.ep_ho = 0
        self.ep_tp = 0
        self.ep_outage = 0
        return self.env.reset(**kwargs)
        
    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        
        self.ep_ho += info.get('is_ho', 0)
        self.ep_tp += info.get('throughput', 0)
        self.ep_outage += info.get('outage', 0)
        
        if done:
            # Chỉ cần gán vào info root, Monitor sẽ tự pick up nhờ info_keywords
            info['ho_count'] = self.ep_ho
            # Tránh chia cho 0
            steps = self.env.step_count if self.env.step_count > 0 else 1
            info['avg_throughput'] = self.ep_tp / steps
            info['outage_steps'] = self.ep_outage
            
        return obs, reward, done, truncated, info

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def make_env(feature_type, scenario, seed):
    def _init():
        env = SatelliteHandoverEnv(k_nearest=5, feature_type=feature_type, scenario=scenario)
        # Wrap Metrics trước
        env = MetricsWrapper(env) 
        # Wrap Monitor sau cùng, khai báo keywords cần log
        env = Monitor(env, info_keywords=("ho_count", "avg_throughput", "outage_steps"))
        env.reset(seed=seed)
        return env
    return _init

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", type=str, default="PPO")
    parser.add_argument("--feature", type=str, default="cga")
    parser.add_argument("--scenario", type=str, default="static")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=100_000)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    set_seed(args.seed)
    run_name = f"{args.algo}_{args.feature}_seed{args.seed}"
    log_dir = f"./results/logs/{args.scenario}/{args.algo}_{args.feature}/"
    model_dir = f"./results/models/{args.scenario}/{args.algo}_{args.feature}/"
    
    print(f"--- START: {run_name} | Scenario: {args.scenario} ---")
    
    n_envs = 4
    env = make_vec_env(make_env(args.feature, args.scenario, args.seed), n_envs=n_envs)

    model = PPO("MlpPolicy", env, verbose=0, tensorboard_log=log_dir, device=f"cuda:{args.gpu}", seed=args.seed)

    start_time = time.time()
    model.learn(total_timesteps=args.steps, tb_log_name=f"seed_{args.seed}")
    
    os.makedirs(model_dir, exist_ok=True)
    model.save(f"{model_dir}/{run_name}_final")
    print(f"--- FINISHED in {time.time()-start_time:.1f}s ---")
    env.close()

if __name__ == "__main__":
    main()
