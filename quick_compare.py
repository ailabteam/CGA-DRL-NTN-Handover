# quick_compare.py
import time
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
from src.envs.ntn_env import SatelliteHandoverEnv

# Test 20k bước
STEPS = 20000 
# Chọn scenario 'random' để thấy rõ sự khác biệt
TEST_SCENARIO = 'random'

class ProgressCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(ProgressCallback, self).__init__(verbose)
        self.rewards = []

    def _on_step(self) -> bool:
        infos = self.locals['infos']
        for info in infos:
            if 'episode' in info:
                r = info['episode']['r']
                self.rewards.append(r)
        return True

def run_quick_test(feature_type):
    print(f"\n========================================")
    print(f"--- Feature: {feature_type.upper()} | Scenario: {TEST_SCENARIO} ---")
    print(f"========================================")
    
    env = make_vec_env(lambda: SatelliteHandoverEnv(k_nearest=5, feature_type=feature_type, scenario=TEST_SCENARIO), n_envs=1)
    
    model = PPO("MlpPolicy", env, verbose=1, learning_rate=3e-4, seed=123)
    callback = ProgressCallback()
    
    start = time.time()
    model.learn(total_timesteps=STEPS, callback=callback)
    end = time.time()
    
    if len(callback.rewards) > 0:
        avg_rew = np.mean(callback.rewards[-5:])
        print(f"-> Average Reward (Last 5): {avg_rew:.2f}")
    else:
        avg_rew = 0
    print(f"-> Time: {end-start:.1f}s")
    return avg_rew

if __name__ == "__main__":
    print(f"Quick comparing CGA vs XYZ on '{TEST_SCENARIO}' scenario...")
    
    rew_cga = run_quick_test('cga')
    rew_xyz = run_quick_test('xyz')
    
    print("\n----------------------------------------")
    print(f"FINAL VERDICT ({TEST_SCENARIO}):")
    print(f"CGA Reward: {rew_cga:.2f}")
    print(f"XYZ Reward: {rew_xyz:.2f}")
    
    if rew_cga > rew_xyz:
        print(">>> SUCCESS: CGA outperforms Baseline in Random Scenario!")
    else:
        print(">>> OBSERVATION: Check results.")
