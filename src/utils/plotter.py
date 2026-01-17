import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# Cấu hình Style cho IEEE Transactions
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("tab10")
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "figure.figsize": (8, 5) # Kích thước chuẩn cho 1 cột hoặc 1/2 trang
})

def extract_tensorboard_data(root_log_dir):
    data = []
    
    # Duyệt qua các thuật toán (PPO_cga, PPO_xyz)
    for algo_name in ["PPO_cga", "PPO_xyz"]:
        algo_path = os.path.join(root_log_dir, algo_name)
        if not os.path.exists(algo_path):
            continue
            
        # Duyệt qua các seed folders
        seed_folders = glob.glob(os.path.join(algo_path, "seed_*"))
        
        for seed_folder in seed_folders:
            # Tìm file tfevents
            event_files = glob.glob(os.path.join(seed_folder, "events.out.tfevents*"))
            if not event_files:
                continue
            
            # Lấy file mới nhất (đề phòng trường hợp chạy lại)
            latest_file = max(event_files, key=os.path.getctime)
            
            try:
                ea = EventAccumulator(latest_file)
                ea.Reload()
                
                # Lấy metric quan trọng nhất: Reward
                # Tag thường là 'rollout/ep_rew_mean' trong SB3
                tags = ea.Tags()['scalars']
                if 'rollout/ep_rew_mean' in tags:
                    events = ea.Scalars('rollout/ep_rew_mean')
                    for e in events:
                        data.append({
                            'Timesteps': e.step,
                            'Mean Reward': e.value,
                            'Algorithm': "CGA-PPO (Proposed)" if "cga" in algo_name else "XYZ-PPO (Baseline)",
                            'Seed': os.path.basename(seed_folder)
                        })
            except Exception as e:
                print(f"Error reading {latest_file}: {e}")

    return pd.DataFrame(data)

def plot_convergence(df, output_path):
    plt.figure()
    
    # Vẽ Shadow Curve tự động bằng Seaborn
    # errorbar='sd' vẽ vùng bóng mờ là độ lệch chuẩn (Standard Deviation)
    sns.lineplot(
        data=df, 
        x="Timesteps", 
        y="Mean Reward", 
        hue="Algorithm", 
        style="Algorithm",
        markers=False, 
        dashes=False,
        errorbar='sd', 
        linewidth=2.5
    )
    
    plt.title("Convergence Analysis: CGA vs. Euclidean Representation")
    plt.xlabel("Training Timesteps")
    plt.ylabel("Average Episode Reward")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc="lower right", frameon=True)
    plt.tight_layout()
    
    # Lưu file PDF vector (chuẩn nộp báo) và PNG (để xem nhanh)
    plt.savefig(output_path + ".pdf", dpi=300, format='pdf', bbox_inches='tight')
    plt.savefig(output_path + ".png", dpi=300, format='png', bbox_inches='tight')
    print(f"Saved plot to {output_path}.png/pdf")

if __name__ == "__main__":
    log_dir = "./results/logs"
    save_dir = "./results/plots"
    os.makedirs(save_dir, exist_ok=True)
    
    print("Extracting data from TensorBoard logs...")
    df = extract_tensorboard_data(log_dir)
    
    if not df.empty:
        print(f"Extracted {len(df)} data points.")
        print("Plotting results...")
        plot_convergence(df, os.path.join(save_dir, "convergence_comparison"))
        
        # Lưu file CSV thống kê để đưa vào báo cáo nếu cần
        df.to_csv(os.path.join(save_dir, "raw_results.csv"), index=False)
    else:
        print("No data found! Check your log directory.")
