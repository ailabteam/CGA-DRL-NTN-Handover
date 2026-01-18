import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# Cấu hình Style chuẩn IEEE
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
    "figure.figsize": (8, 5)
})

def extract_data(root_dir, scenario):
    data = []
    scenario_path = os.path.join(root_dir, scenario)
    
    if not os.path.exists(scenario_path):
        print(f"Warning: Path not found {scenario_path}")
        return pd.DataFrame()

    # Duyệt qua các thuật toán trong scenario folder
    # Cấu trúc: results/logs/random/PPO_cga/
    for algo_folder in glob.glob(os.path.join(scenario_path, "PPO_*")):
        algo_name = os.path.basename(algo_folder) # PPO_cga hoặc PPO_xyz
        
        # Label hiển thị trên biểu đồ
        if "cga" in algo_name:
            label = "CGA-PPO (Proposed)"
        else:
            label = "XYZ-PPO (Baseline)"
            
        # Duyệt qua các seed
        for seed_folder in glob.glob(os.path.join(algo_folder, "seed_*")):
            event_files = glob.glob(os.path.join(seed_folder, "events.out.tfevents*"))
            if not event_files: continue
            
            latest_file = max(event_files, key=os.path.getctime)
            
            try:
                ea = EventAccumulator(latest_file)
                ea.Reload()
                if 'rollout/ep_rew_mean' in ea.Tags()['scalars']:
                    events = ea.Scalars('rollout/ep_rew_mean')
                    for e in events:
                        data.append({
                            'Timesteps': e.step,
                            'Mean Reward': e.value,
                            'Algorithm': label,
                            'Seed': os.path.basename(seed_folder)
                        })
            except Exception as e:
                print(f"Error reading {latest_file}: {e}")
                
    return pd.DataFrame(data)

def plot_scenario(df, scenario_name, output_dir):
    if df.empty:
        print(f"No data for scenario: {scenario_name}")
        return

    plt.figure()
    sns.lineplot(
        data=df, 
        x="Timesteps", 
        y="Mean Reward", 
        hue="Algorithm", 
        style="Algorithm",
        errorbar='sd', # Vùng bóng mờ là độ lệch chuẩn
        linewidth=2.5
    )
    
    title = "Validation: Fixed User Location" if scenario_name == "static" else "Generalization: Random User Locations"
    plt.title(title)
    plt.xlabel("Training Timesteps")
    plt.ylabel("Average Episode Reward")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc="lower right", frameon=True)
    plt.tight_layout()
    
    filename = f"convergence_{scenario_name}"
    plt.savefig(os.path.join(output_dir, filename + ".pdf"), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, filename + ".png"), dpi=300, bbox_inches='tight')
    print(f"Saved plot: {filename}")

if __name__ == "__main__":
    LOG_ROOT = "./results/logs"
    PLOT_ROOT = "./results/plots"
    os.makedirs(PLOT_ROOT, exist_ok=True)
    
    for scen in ["static", "random"]:
        print(f"Processing scenario: {scen}...")
        df = extract_data(LOG_ROOT, scen)
        plot_scenario(df, scen, PLOT_ROOT)
