import time
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
from src.envs.ntn_env import SatelliteHandoverEnv

# Wrapper dummy để test local (không cần log tensorboard)
class MetricsWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.ep_ho = 0
    def reset(self, **kwargs):
        self.ep_ho = 0
        return self.env.reset(**kwargs)
    def step(self, action):
        obs, reward, done, trunc, info = self.env.step(action)
        self.ep_ho += info.get('is_ho', 0)
        if done: info['episode'] = {'r': 0, 'ho_count': self.ep_ho}
        return obs, reward, done, trunc, info

STEPS = 5000 
SCENARIO = 'random'

class LogCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
    def _on_step(self) -> bool:
        infos = self.locals['infos']
        for info in infos:
            if 'episode' in info:
                ho = info['episode'].get('ho_count', 0)
                print(f"   >>> Episode Done. HO Count: {ho}")
        return True

def run_test():
    print(f"Testing Environment with Rotation Noise (Scenario: {SCENARIO})...")
    
    # Wrapper stack thủ công cho test
    env = make_vec_env(lambda: MetricsWrapper(SatelliteHandoverEnv(feature_type='cga', scenario=SCENARIO)), n_envs=1)
    
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=STEPS, callback=LogCallback())
    print("Test Finished.")

if __name__ == "__main__":
    run_test()
