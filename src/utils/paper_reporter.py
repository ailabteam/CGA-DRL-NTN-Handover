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
# Set custom palette for 4 algorithms
# CGA-PPO (Red), XYZ-PPO (Blue), Max-Elevation (Green), Min-Handover (Orange)
PALETTE = {
    "CGA-PPO": "#d62728",      
    "XYZ-PPO": "#1f77b4",      
    "Max-Elevation": "#2ca02c", 
    "Min-Handover": "#ff7f0e"
}

# Line styles: Solid for AI, Dashed for Heuristic
LINE_STYLES = {
    "CGA-PPO": "-", 
    "XYZ-PPO": "--",
    "Max-Elevation": ":",
    "Min-Handover": "-."
}

# Font settings for LaTeX-like quality
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.figsize": (3.5, 2.5), # Standard single column width
    "figure.dpi": 300
})

def load_all_data(root_dir):
    """Load logs from all algorithms (PPO & Heuristic)"""
    data = []
    
    # Structure: root/scenario/algo/seed
    for scenario in ["static", "random"]:
        scen_path = os.path.join(root_dir, scenario)
        if not os.path.exists(scen_path): continue
        
        # Iterate over all algorithm folders
        for algo_dir in glob.glob(os.path.join(scen_path, "*")):
            algo_raw = os.path.basename(algo_dir)
            
            # Map folder name to display name
            if "PPO_cga" in algo_raw: algo_name = "CGA-PPO"
            elif "PPO_xyz" in algo_raw: algo_name = "XYZ-PPO"
            elif "Max-Elevation" in algo_raw: algo_name = "Max-Elevation"
            elif "Min-Handover" in algo_raw: algo_name = "Min-Handover"
            else: continue # Skip unknown folders
            
            for seed_dir in glob.glob(os.path.join(algo_dir, "seed_*")):
                # Extract seed number if needed, e.g. seed_101_1 -> 101
                # But here just use full folder name as ID to avoid duplicates
                run_id = os.path.basename(seed_dir)
                
                # Find latest tfevents file
                files = glob.glob(os.path.join(seed_dir, "events.out.tfevents*"))
                if not files: continue
                latest_file = max(files, key=os.path.getctime)
                
                try:
                    ea = EventAccumulator(latest_file)
                    ea.Reload()
                    tags = ea.Tags()['scalars']
                    
                    # 1. Mean Reward
                    if 'rollout/ep_rew_mean' in tags:
                        for e in ea.Scalars('rollout/ep_rew_mean'):
                            data.append({
                                'Scenario': scenario.capitalize(),
                                'Algorithm': algo_name,
                                'Seed': run_id,
                                'Step': e.step,
                                'Metric': 'Mean Reward',
                                'Value': e.value
                            })
                            
                    # 2. Episode Length
                    if 'rollout/ep_len_mean' in tags:
                        for e in ea.Scalars('rollout/ep_len_mean'):
                            data.append({
                                'Scenario': scenario.capitalize(),
                                'Algorithm': algo_name,
                                'Seed': run_id,
                                'Step': e.step,
                                'Metric': 'Episode Length',
                                'Value': e.value
                            })
                            
                except Exception as e:
                    print(f"Err reading {latest_file}: {e}")
                    
    return pd.DataFrame(data)

def plot_convergence_grid(df, output_dir):
    """Fig 1: Learning Curves for all algorithms"""
    df_rew = df[df['Metric'] == 'Mean Reward'].copy()
    
    # Smoothing for AI methods (Heuristic is constant, but smoothing is harmless)
    df_rew['Value'] = df_rew.groupby(['Scenario', 'Algorithm', 'Seed'])['Value'] \
                            .transform(lambda x: x.rolling(window=20, min_periods=1).mean())

    g = sns.FacetGrid(df_rew, col="Scenario", sharey=False, height=3, aspect=1.3)
    g.map_dataframe(sns.lineplot, x="Step", y="Value", hue="Algorithm", style="Algorithm", 
                    palette=PALETTE, errorbar=('ci', 95), linewidth=1.5)
    g.add_legend()
    g.set_axis_labels("Training Timesteps", "Average Reward")
    g.set_titles("{col_name} Scenario")
    
    g.savefig(os.path.join(output_dir, "Fig1_Convergence.pdf"), bbox_inches='tight')
    print("Generated Fig 1 (Convergence).")

def plot_stability_boxplot(df, output_dir):
    """Fig 2: Boxplot of final performance (Robustness)"""
    # Get last 10% steps
    max_step = df['Step'].max()
    threshold = max_step * 0.9
    df_final = df[(df['Metric'] == 'Mean Reward') & (df['Step'] > threshold)]
    
    plt.figure(figsize=(6, 4))
    sns.boxplot(data=df_final, x="Scenario", y="Value", hue="Algorithm", 
                palette=PALETTE, showfliers=False, width=0.6)
    plt.title("Performance Stability (Final 10% Steps)")
    plt.ylabel("Converged Reward Distribution")
    plt.grid(True, axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, "Fig2_Stability_Boxplot.pdf"), bbox_inches='tight')
    print("Generated Fig 2 (Stability).")

def generate_stats_tables(df, output_dir):
    """Table II & III"""
    max_step = df['Step'].max()
    # Filter final converged data
    df_final = df[(df['Metric'] == 'Mean Reward') & (df['Step'] > max_step * 0.95)]
    
    if df_final.empty:
        print("No converged data found for Tables.")
        return

    # 1. Mean +- Std Table
    stats_df = df_final.groupby(['Scenario', 'Algorithm'])['Value'].agg(['mean', 'std']).reset_index()
    stats_df.to_csv(os.path.join(output_dir, "Table2_Raw.csv"))
    
    print("\n=== TABLE II: OVERALL PERFORMANCE ===")
    print(stats_df)
    
    # 2. T-Test (Comparing CGA vs others in Random Scenario)
    print("\n=== TABLE III: T-TEST (Scenario: Random) ===")
    p_values = []
    target_scen = "Random"
    
    # Get CGA data
    cga_data = df_final[(df_final['Scenario'] == target_scen) & (df_final['Algorithm'] == 'CGA-PPO')]['Value']
    
    if cga_data.empty:
        print("CGA-PPO data missing for T-test.")
        return

    for competitor in ["XYZ-PPO", "Max-Elevation", "Min-Handover"]:
        comp_data = df_final[(df_final['Scenario'] == target_scen) & (df_final['Algorithm'] == competitor)]['Value']
        
        if comp_data.empty: continue
        
        t_stat, p_val = stats.ttest_ind(cga_data, comp_data, equal_var=False)
        res = "Significant" if p_val < 0.05 else "Not Sig"
        
        print(f"CGA vs {competitor}: P-value = {p_val:.5e} ({res})")
        p_values.append({'Competitor': competitor, 'P-value': p_val, 'Result': res})
        
    pd.DataFrame(p_values).to_csv(os.path.join(output_dir, "Table3_P_Values.csv"))

if __name__ == "__main__":
    LOG_ROOT = "./results/logs"
    REPORT_DIR = "./results/report"
    os.makedirs(REPORT_DIR, exist_ok=True)
    
    print("Loading all logs (AI + Heuristics)...")
    df = load_all_data(LOG_ROOT)
    
    if not df.empty:
        print(f"Loaded {len(df)} data points.")
        print("Generating Figures...")
        plot_convergence_grid(df, REPORT_DIR)
        plot_stability_boxplot(df, REPORT_DIR)
        
        print("Generating Tables...")
        generate_stats_tables(df, REPORT_DIR)
        
        print(f"\nReport generated at {REPORT_DIR}")
    else:
        print("No data found! Please run training and heuristics scripts first.")
