import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# --- CONFIGURATION ---
plt.style.use('seaborn-v0_8-paper')
# Custom palette
PALETTE = {
    "CGA-PPO": "#d62728",      
    "XYZ-PPO": "#1f77b4",      
    "Max-Elevation": "#2ca02c", 
    "Min-Handover": "#ff7f0e"
}

def load_all_data(root_dir):
    """Load logs from all algorithms (PPO & Heuristic)"""
    data = []
    print(f"Scanning logs in {root_dir}...")
    
    # Structure: root/scenario/algo/seed
    for scenario in ["static", "random"]:
        scen_path = os.path.join(root_dir, scenario)
        if not os.path.exists(scen_path): 
            print(f"  [Skip] Scenario dir not found: {scen_path}")
            continue
        
        # Iterate over algorithm folders
        # glob * để lấy hết folder con
        for algo_dir in glob.glob(os.path.join(scen_path, "*")):
            if not os.path.isdir(algo_dir): continue
            
            algo_raw = os.path.basename(algo_dir)
            
            # Map folder name to display name
            if "PPO_cga" in algo_raw: algo_name = "CGA-PPO"
            elif "PPO_xyz" in algo_raw: algo_name = "XYZ-PPO"
            elif "Max-Elevation" in algo_raw: algo_name = "Max-Elevation"
            elif "Min-Handover" in algo_raw: algo_name = "Min-Handover"
            else: 
                print(f"  [Skip] Unknown algo folder: {algo_raw}")
                continue 
            
            # Iterate seeds
            for seed_dir in glob.glob(os.path.join(algo_dir, "seed_*")):
                seed = os.path.basename(seed_dir)
                
                # Find tfevents
                files = glob.glob(os.path.join(seed_dir, "events.out.tfevents*"))
                if not files: 
                    print(f"  [Warn] No events file in {seed_dir}")
                    continue
                
                latest_file = max(files, key=os.path.getctime)
                
                try:
                    ea = EventAccumulator(latest_file)
                    ea.Reload()
                    tags = ea.Tags()['scalars']
                    
                    # Map log tag sang Metric Name
                    # PPO logs: rollout/ep_rew_mean, rollout/ep_len_mean
                    # Custom wrapper logs: ho_count, avg_throughput, outage_steps
                    metrics_map = {
                        'rollout/ep_rew_mean': 'Reward',
                        'rollout/ep_len_mean': 'Ep Length',
                        'ho_count': 'Handover Count',
                        'outage_steps': 'Outage Duration',
                        'avg_throughput': 'Throughput'
                    }
                    
                    count = 0
                    for tag, metric_name in metrics_map.items():
                        if tag in tags:
                            for e in ea.Scalars(tag):
                                data.append({
                                    'Scenario': scenario.capitalize(),
                                    'Algorithm': algo_name,
                                    'Seed': seed,
                                    'Step': e.step,
                                    'Metric': metric_name,
                                    'Value': e.value
                                })
                                count += 1
                    
                    if count > 0:
                        # Print debug info for first few seeds
                        pass 
                        
                except Exception as e:
                    print(f"  [Err] Reading {latest_file}: {e}")
                    
    df = pd.DataFrame(data)
    print(f"Loaded total {len(df)} rows.")
    if not df.empty:
        print("Algorithms found:", df['Algorithm'].unique())
        print("Metrics found:", df['Metric'].unique())
    return df

def generate_stats_tables(df, output_dir):
    if df.empty:
        print("DataFrame is empty. Cannot generate tables.")
        return

    # Filter final data (Last 20%)
    # Nếu dữ liệu ít quá (chưa chạy xong), lấy 5 điểm cuối cùng
    max_step = df['Step'].max()
    
    # Logic lọc linh hoạt hơn:
    # Nếu max_step > 10000 (đã chạy nhiều), lấy 20% cuối.
    # Nếu mới chạy test (max_step nhỏ), lấy hết.
    if max_step > 50000:
        threshold = max_step * 0.8
        df_final = df[df['Step'] > threshold]
        print(f"Filtering data > step {threshold}")
    else:
        print("Short run detected. Using all data for stats.")
        df_final = df

    if df_final.empty: 
        print("No data after filtering.")
        return

    print("\n=== DETAILED PERFORMANCE REPORT ===")
    
    # Pivot table: Index=Algo, Columns=Metric, Values="Mean (Std)"
    for scen in df_final['Scenario'].unique():
        print(f"\n--- SCENARIO: {scen} ---")
        df_scen = df_final[df_final['Scenario'] == scen]
        
        # Calculate mean and std
        summary = df_scen.groupby(['Algorithm', 'Metric'])['Value'].agg(['mean', 'std']).reset_index()
        
        # Format string
        summary['Display'] = summary.apply(lambda x: f"{x['mean']:.2f} ({x['std']:.2f})", axis=1)
        
        # Pivot
        try:
            pivot_table = summary.pivot(index='Algorithm', columns='Metric', values='Display')
            print(pivot_table)
            pivot_table.to_csv(os.path.join(output_dir, f"Table_{scen}.csv"))
        except Exception as e:
            print(f"Error pivoting table: {e}")
            print(summary)

def plot_convergence(df, output_dir):
    """Plot Reward Curve"""
    df_rew = df[df['Metric'] == 'Reward']
    if df_rew.empty: return

    plt.figure(figsize=(6, 4))
    g = sns.FacetGrid(df_rew, col="Scenario", sharey=False, height=3.5, aspect=1.3)
    g.map_dataframe(sns.lineplot, x="Step", y="Value", hue="Algorithm", style="Algorithm", 
                    palette=PALETTE, errorbar=('ci', 95), linewidth=1.5)
    g.add_legend()
    g.set_titles("{col_name}")
    g.set_axis_labels("Steps", "Reward")
    plt.savefig(os.path.join(output_dir, "Fig_Convergence.png"), dpi=150)
    print("Saved Fig_Convergence.png")

if __name__ == "__main__":
    LOG_ROOT = "./results/logs"
    REPORT_DIR = "./results/report"
    os.makedirs(REPORT_DIR, exist_ok=True)
    
    df = load_all_data(LOG_ROOT)
    
    if not df.empty:
        generate_stats_tables(df, REPORT_DIR)
        plot_convergence(df, REPORT_DIR)
    else:
        print("No data loaded. Please check folder structure.")
