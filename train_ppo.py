# train_ppo.py
import os
import time
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor

from src.envs.ntn_env import SatelliteHandoverEnv

# Cấu hình Hyperparameters (Theo chuẩn Paper DRL)
TOTAL_TIMESTEPS = 100_000  # Chạy thử 100k bước (Khoảng 20 phút)
LEARNING_RATE = 3e-4
N_STEPS = 2048
BATCH_SIZE = 64
GAMMA = 0.99
ENT_COEF = 0.01 # Khuyến khích khám phá (Exploration)

def make_env():
    """Hàm wrapper để tạo env tương thích với Monitor của SB3"""
    env = SatelliteHandoverEnv(k_nearest=5)
    # Monitor giúp log lại reward từng episode để vẽ biểu đồ
    return Monitor(env)

def main():
    # 1. Thiết lập đường dẫn lưu log
    log_dir = "./results/logs/ppo_cga/"
    model_dir = "./results/models/ppo_cga/"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    print(f"=== TRAINING PPO AGENT ON CGA ENVIRONMENT ===")
    print(f"Device: GPU 0 (CUDA)")
    
    # 2. Tạo Vectorized Environment (Chạy 4 luồng song song để tăng tốc thu thập dữ liệu)
    # Lưu ý: Vì code CGA của ta đang dùng CPU, việc dùng 4-8 envs song song sẽ tận dụng hết 40 vCPU của bạn
    n_envs = 8 
    env = make_vec_env(make_env, n_envs=n_envs)

    # 3. Khởi tạo Agent PPO
    model = PPO(
        "MlpPolicy", 
        env, 
        learning_rate=LEARNING_RATE,
        n_steps=N_STEPS,
        batch_size=BATCH_SIZE,
        gamma=GAMMA,
        ent_coef=ENT_COEF,
        verbose=1,
        tensorboard_log=log_dir,
        device="cuda:0" # Chỉ định rõ GPU 0
    )

    # 4. Callback để lưu model định kỳ
    checkpoint_callback = CheckpointCallback(
        save_freq=10000 // n_envs, 
        save_path=model_dir,
        name_prefix="ppo_cga_model"
    )

    # 5. Bắt đầu Training
    start_time = time.time()
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS, 
        callback=checkpoint_callback,
        tb_log_name="run_1"
    )
    end_time = time.time()

    # 6. Lưu Final Model
    model.save(f"{model_dir}/final_model")
    print(f"Training finished in {(end_time - start_time)/60:.2f} minutes.")
    
    env.close()

if __name__ == "__main__":
    main()
