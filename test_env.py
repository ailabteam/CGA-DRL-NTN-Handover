# test_env.py
import time
import numpy as np
from src.envs.ntn_env import SatelliteHandoverEnv

def test_simulation_speed():
    print("=== TESTING GYM ENVIRONMENT ===")
    
    env = SatelliteHandoverEnv(k_nearest=5)
    obs, _ = env.reset()
    
    print(f"Observation Shape: {obs.shape}")
    print(f"Sample Observation (Top 1 Sat): {obs[0:3]}") # Dist, Cos, IsConn
    
    start_time = time.time()
    n_steps = 1000
    
    total_reward = 0
    handovers = 0
    last_sat = -1
    
    print(f"\nRunning {n_steps} simulation steps...")
    for i in range(n_steps):
        # Random Action: Luôn chọn vệ tinh tốt nhất (index 0) để test logic
        action = 0 
        obs, reward, done, _, _ = env.step(action)
        total_reward += reward
        
        # Check Handover
        current_sat_dist = obs[0]
        # Hacky way to check ID change via env internals (for debug only)
        if env.current_sat_id != last_sat and last_sat != -1:
            handovers += 1
        last_sat = env.current_sat_id
        
        if done:
            break
            
    end_time = time.time()
    duration = end_time - start_time
    fps = n_steps / duration
    
    print(f"Done in {duration:.2f} seconds.")
    print(f"Simulation Speed: {fps:.1f} steps/second")
    print(f"Total Reward: {total_reward:.2f}")
    print(f"Total Handovers: {handovers}")
    
    if fps > 50:
        print("-> [PASS] Simulation speed is acceptable for DRL.")
    else:
        print("-> [WARNING] Simulation is too slow!")

if __name__ == "__main__":
    test_simulation_speed()
