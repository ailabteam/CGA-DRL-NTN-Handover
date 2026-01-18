import argparse
import os
import numpy as np
import random
from torch.utils.tensorboard import SummaryWriter
from src.envs.ntn_env import SatelliteHandoverEnv

# 1. Định nghĩa Heuristic Agents
class MaxElevationAgent:
    def predict(self, env):
        candidates = env._get_candidates()
        if not candidates: return 0
        # Chọn vệ tinh có feature[1] (cos_el) lớn nhất
        # candidates đã sort theo dist, nên ta phải tìm max cos
        best_idx = np.argmax([c['features_cga'][1] for c in candidates])
        return best_idx

class MinHandoverAgent:
    def predict(self, env):
        # Nếu đang kết nối tốt (không outage), giữ nguyên
        if env.current_sat_id != -1:
            candidates = env._get_candidates()
            for i, c in enumerate(candidates):
                if c['id'] == env.current_sat_id:
                    return i # Giữ nguyên index
        
        # Nếu mất kết nối, fallback về Max-Elevation
        return MaxElevationAgent().predict(env)

# 2. Hàm chạy đánh giá
def run_heuristic(algo_name, scenario, seed, total_steps=200000):
    print(f"--- Running {algo_name} | Scenario: {scenario} | Seed: {seed} ---")
    
    # Setup Env
    env = SatelliteHandoverEnv(k_nearest=5, scenario=scenario)
    env.reset(seed=seed)
    random.seed(seed)
    np.random.seed(seed)
    
    # Setup Logging giống PPO
    log_dir = f"./results/logs/{scenario}/{algo_name}/seed_{seed}_1"
    writer = SummaryWriter(log_dir)
    
    agent = MaxElevationAgent() if algo_name == "Max-Elevation" else MinHandoverAgent()
    
    obs, _ = env.reset()
    ep_rew = 0
    ep_ho = 0
    ep_len = 0
    
    for step in range(total_steps):
        action = agent.predict(env)
        obs, reward, done, _, info = env.step(action)
        
        ep_rew += reward
        ep_ho += info.get('is_ho', 0)
        ep_len += 1
        
        # Log giống PPO (ghi từng bước hoặc từng ep)
        # Để file log nhẹ, ta chỉ ghi khi done episode
        if done:
            writer.add_scalar("rollout/ep_rew_mean", ep_rew, step)
            writer.add_scalar("rollout/ep_len_mean", ep_len, step)
            # Log thêm metrics phụ nếu paper_reporter hỗ trợ
            # Hiện tại paper_reporter chỉ đọc rew/len, ta có thể mở rộng sau
            
            obs, _ = env.reset()
            ep_rew = 0
            ep_ho = 0
            ep_len = 0
            
    writer.close()
    print(f"Finished {algo_name} seed {seed}")

if __name__ == "__main__":
    SEEDS = [101, 102, 103, 201, 202]
    SCENARIOS = ["static", "random"]
    ALGOS = ["Max-Elevation", "Min-Handover"]
    
    for scen in SCENARIOS:
        for algo in ALGOS:
            for seed in SEEDS:
                run_heuristic(algo, scen, seed)
