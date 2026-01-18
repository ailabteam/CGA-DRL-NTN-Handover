import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# --- CONFIGURATION FOR IEEE TRANS ---
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("tab10")
# Font chữ chuẩn Latex
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.figsize": (3.5, 2.5), # Kích thước chuẩn cho 1 cột (single column width)
    "figure.dpi": 300
})

COLORS = {"CGA-PPO": "#d62728", "XYZ-PPO": "#1f77b4"}
LINE_STYLES = {"CGA-PPO": "-", "XYZ-PPO": "--"}

def load_all_data(root_dir):
    """Đọc toàn bộ log từ cấu trúc folder mới"""
    data = []
    
    # Cấu trúc: root/scenario/algo/seed
    for scenario in ["static", "random"]:
        scen_path = os.path.join(root_dir, scenario)
        if not os.path.exists(scen_path): continue
        
        for algo_dir in glob.glob(os.path.join(scen_path, "PPO_*")):
            algo_raw = os.path.basename(algo_dir)
            algo_name = "CGA-PPO" if "cga" in algo_raw else "XYZ-PPO"
            
            for seed_dir in glob.glob(os.path.join(algo_dir, "seed_*")):
                seed = os.path.basename(seed_dir).split('_')[1]
                
                # Tìm file log mới nhất
                files = glob.glob(os.path.join(seed_dir, "events.out.tfevents*"))
                if not files: continue
                latest_file = max(files, key=os.path.getctime)
                
                try:
                    ea = EventAccumulator(latest_file)
                    ea.Reload()
                    tags = ea.Tags()['scalars']
                    
                    # Lấy Reward
                    if 'rollout/ep_rew_mean' in tags:
                        for e in ea.Scalars('rollout/ep_rew_mean'):
                            data.append({
                                'Scenario': scenario.capitalize(),
                                'Algorithm': algo_name,
                                'Seed': seed,
                                'Step': e.step,
                                'Metric': 'Mean Reward',
                                'Value': e.value
                            })
                            
                    # Lấy Episode Length (Độ ổn định kết nối)
                    if 'rollout/ep_len_mean' in tags:
                        for e in ea.Scalars('rollout/ep_len_mean'):
                            data.append({
                                'Scenario': scenario.capitalize(),
                                'Algorithm': algo_name,
                                'Seed': seed,
                                'Step': e.step,
                                'Metric': 'Episode Length',
                                'Value': e.value
                            })
                            
                except Exception as e:
                    print(f"Err: {latest_file} - {e}")
                    
    return pd.DataFrame(data)

def plot_convergence_grid(df, output_dir):
    """Fig 1: Learning Curves cho cả 2 kịch bản"""
    df_rew = df[df['Metric'] == 'Mean Reward'].copy()
    
    # Smoothing
    df_rew['Value'] = df_rew.groupby(['Scenario', 'Algorithm', 'Seed'])['Value'] \
                            .transform(lambda x: x.rolling(window=10, min_periods=1).mean())

    g = sns.FacetGrid(df_rew, col="Scenario", sharey=False, height=3, aspect=1.3)
    g.map_dataframe(sns.lineplot, x="Step", y="Value", hue="Algorithm", style="Algorithm", 
                    palette=COLORS, errorbar=('ci', 95), linewidth=2)
    g.add_legend()
    g.set_axis_labels("Timesteps", "Average Reward")
    g.set_titles("{col_name} Scenario")
    
    g.savefig(os.path.join(output_dir, "Fig1_Convergence.pdf"))
    print("Generated Fig 1.")

def plot_stability_boxplot(df, output_dir):
    """Fig 2: Boxplot phân phối hiệu năng cuối cùng (Robustness)"""
    # Lấy dữ liệu của 10% steps cuối cùng (giai đoạn hội tụ)
    max_step = df['Step'].max()
    threshold = max_step * 0.9
    df_final = df[(df['Metric'] == 'Mean Reward') & (df['Step'] > threshold)]
    
    plt.figure(figsize=(5, 4))
    sns.boxplot(data=df_final, x="Scenario", y="Value", hue="Algorithm", 
                palette=COLORS, showfliers=False)
    plt.title("Performance Stability (Final 10% Steps)")
    plt.ylabel("Converged Reward Distribution")
    plt.grid(True, axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, "Fig2_Stability_Boxplot.pdf"))
    print("Generated Fig 2.")

def plot_episode_len(df, output_dir):
    """Fig 3: Connection Stability (Episode Length)"""
    df_len = df[df['Metric'] == 'Episode Length'].copy()
    # Smoothing
    df_len['Value'] = df_len.groupby(['Scenario', 'Algorithm', 'Seed'])['Value'] \
                            .transform(lambda x: x.rolling(window=10, min_periods=1).mean())
    
    plt.figure(figsize=(5, 3))
    # Chỉ vẽ kịch bản Random (vì Static thường max length)
    df_random = df_len[df_len['Scenario'] == 'Random']
    
    if not df_random.empty:
        sns.lineplot(data=df_random, x="Step", y="Value", hue="Algorithm", 
                    palette=COLORS, style="Algorithm")
        plt.title("Connection Durability (Random Scenario)")
        plt.ylabel("Avg Episode Length (Steps)")
        plt.xlabel("Training Steps")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "Fig3_Connection_Length.pdf"))
        print("Generated Fig 3.")

def generate_stats_tables(df, output_dir):
    """Tạo Table II và Table III"""
    # Lấy dữ liệu cuối
    max_step = df['Step'].max()
    df_final = df[(df['Metric'] == 'Mean Reward') & (df['Step'] > max_step * 0.95)]
    
    # 1. Tính Mean +- Std (Table II)
    stats_df = df_final.groupby(['Scenario', 'Algorithm'])['Value'].agg(['mean', 'std']).reset_index()
    stats_df.to_csv(os.path.join(output_dir, "Table2_Raw.csv"))
    
    print("\n=== TABLE II: OVERALL PERFORMANCE ===")
    print(stats_df)
    
    # 2. T-Test (Table III)
    print("\n=== TABLE III: STATISTICAL SIGNIFICANCE (T-TEST) ===")
    p_values = []
    for scen in df_final['Scenario'].unique():
        data_cga = df_final[(df_final['Scenario'] == scen) & (df_final['Algorithm'] == 'CGA-PPO')]['Value']
        data_xyz = df_final[(df_final['Scenario'] == scen) & (df_final['Algorithm'] == 'XYZ-PPO')]['Value']
        
        # Welch's t-test (không giả định phương sai bằng nhau)
        t_stat, p_val = stats.ttest_ind(data_cga, data_xyz, equal_var=False)
        
        res = "Significant" if p_val < 0.05 else "Not Sig"
        print(f"Scenario {scen}: P-value = {p_val:.5f} ({res})")
        
        p_values.append({'Scenario': scen, 'P-value': p_val, 'Result': res})
        
    pd.DataFrame(p_values).to_csv(os.path.join(output_dir, "Table3_P_Values.csv"))

if __name__ == "__main__":
    LOG_ROOT = "./results/logs"
    REPORT_DIR = "./results/report"
    os.makedirs(REPORT_DIR, exist_ok=True)
    
    print("Loading data...")
    df = load_all_data(LOG_ROOT)
    
    if not df.empty:
        print("Generating Figures...")
        plot_convergence_grid(df, REPORT_DIR)
        plot_stability_boxplot(df, REPORT_DIR)
        plot_episode_len(df, REPORT_DIR)
        
        print("Generating Tables...")
        generate_stats_tables(df, REPORT_DIR)
        
        print(f"\nDone! Report saved to {REPORT_DIR}")
    else:
        print("No data found!")
