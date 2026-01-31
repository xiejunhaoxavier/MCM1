#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MCM 2026 Problem C: Model Evaluation & Visualization
Evaluates and compares Basic Monte Carlo vs. Bayesian-Dirichlet models.

Metrics:
1. Explanation Rate: % of elimination weeks where n_valid_sims > 0
2. Certainty: Average confidence (valid_sims / total_sims) for explained weeks
3. Stability: Average standard deviation of fan share estimates

Author: MCM Team
Date: 2026-01-30
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# =============================================================================
# Configuration
# =============================================================================
BASE_DIR = Path(__file__).parent.parent
RESULTS_DIR = BASE_DIR / "results"
PLOTS_DIR = RESULTS_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

BASIC_RESULTS_PATH = RESULTS_DIR / "full_simulation_basic.csv"
BAYESIAN_RESULTS_PATH = RESULTS_DIR / "full_simulation_bayesian.csv"
REPORT_PATH = RESULTS_DIR / "model_comparison_report.md"

# Plot style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


# =============================================================================
# Data Loading & Preprocessing
# =============================================================================
def load_and_prep_data(filepath: Path, model_name: str) -> pd.DataFrame:
    """
    Load simulation results and prepare for evaluation.
    """
    if not filepath.exists():
        raise FileNotFoundError(f"Results file not found: {filepath}")
    
    df = pd.read_csv(filepath)
    df['model'] = model_name
    
    # Ensure n_valid_sims is numeric, treating NaN as NaN
    df['n_valid_sims'] = pd.to_numeric(df['n_valid_sims'], errors='coerce')
    
    return df


def get_week_classification(group: pd.DataFrame) -> str:
    """
    Classify a (season, week) group based on simulation outcome.
    
    Returns:
        - 'Non-Elimination': No actual elimination occurred (n_valid_sims is NaN)
        - 'Unexplained': Elimination occurred but model found 0 valid sims
        - 'Explained': Elimination occurred and model found valid sims
    """
    # Check if this week had an elimination
    # In data_processing.py, is_eliminated is True for the eliminated contestant
    has_elimination = group['is_eliminated'].any()
    
    if not has_elimination:
        return 'Non-Elimination'
    
    # Check simulation results
    # n_valid_sims should be uniform for the group if it ran
    # If it's NaN, it implies the simulation skipped this week (likely treated as non-elimination)
    valid_sims = group['n_valid_sims'].iloc[0]
    
    if pd.isna(valid_sims):
        return 'Non-Elimination' # Should ideally encompass the check above, but safe to separate
    
    if valid_sims > 0:
        return 'Explained'
    else:
        return 'Unexplained'


def compute_week_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates data to (season, week) level to determine success/failure.
    """
    # Group by season, week
    # We take the first row for week-level metadata, but need to aggregate specific cols
    week_groups = df.groupby(['season', 'week'])
    
    records = []
    
    for (season, week), group in week_groups:
        classification = get_week_classification(group)
        rule_system = group['rule_system'].iloc[0]
        
        # Metrics for this week
        valid_sims = group['n_valid_sims'].iloc[0]
        confidence = group['confidence'].iloc[0] if classification == 'Explained' else 0.0
        
        # Stability: Mean std dev of fan shares across all contestants in this week
        # Only relevant if Explained
        avg_share_std = group['share_std'].mean() if classification == 'Explained' else np.nan
        
        # Entropy (Information/Uncertainty): Mean Shannon entropy
        # Only available for Bayesian Mixture model currently
        avg_share_entropy = np.nan
        if 'share_entropy' in group.columns and classification == 'Explained':
             avg_share_entropy = group['share_entropy'].mean()

        records.append({
            'season': season,
            'week': week,
            'rule_system': rule_system,
            'classification': classification,
            'n_valid_sims': valid_sims,
            'confidence': confidence,
            'avg_share_std': avg_share_std,
            'avg_share_entropy': avg_share_entropy,
            'model': group['model'].iloc[0]
        })
        
    return pd.DataFrame(records)


# =============================================================================
# Visualizations
# =============================================================================
def plot_entropy_heatmap(raw_df: pd.DataFrame, season: int, model_name: str = 'Bayesian'):
    """
    Generate a heatmap of Share Entropy for a specific season.
    X-axis: Week
    Y-axis: Contestant
    Color: Entropy (Darker = Higher Uncertainty/Entropy)
    """
    season_data = raw_df[(raw_df['season'] == season) & (raw_df['model'] == model_name)].copy()
    
    if season_data.empty or 'share_entropy' not in season_data.columns:
        print(f"[INFO] Skipping entropy heatmap for S{season} (No entropy data found)")
        return

    # Pivot: Index=Contestant, Columns=Week, Values=Entropy
    pivot_table = season_data.pivot(index='celebrity_name', columns='week', values='share_entropy')
    
    # Sort Y-axis by average entropy or elimination order? 
    # Elimination order is better to see the "survival" path.
    # We can infer elimination order by who stops appearing.
    # Alternatively, sort by total entropy.
    pivot_table['mean_ent'] = pivot_table.mean(axis=1)
    pivot_table = pivot_table.sort_values('mean_ent', ascending=False) # Most uncertain on top
    pivot_table = pivot_table.drop(columns='mean_ent')
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(pivot_table, cmap="YlGnBu", annot=True, fmt=".2f", cbar_kws={'label': 'Shannon Entropy (nats)'})
    
    plt.title(f'Uncertainty Heatmap: Season {season} ({model_name} Mixture Model)\n'
              f'High Entropy (Dark) = High Uncertainty/Latent Potential\n'
              f'Low Entropy (Light) = Strict Constraints', fontsize=14)
    plt.xlabel('Week Num', fontsize=12)
    plt.ylabel('Contestant', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f"entropy_heatmap_S{season}.png", dpi=300)
    plt.close()


def plot_explanation_rate(perf_df: pd.DataFrame):
    """
    Plot Explanation Rate comparison by Rule System, including Overall.
    """
    # Filter out Non-Elimination weeks
    df = perf_df[perf_df['classification'] != 'Non-Elimination'].copy()
    
    # 1. Calculate Per-Rule System Rates
    summary = df.groupby(['model', 'rule_system', 'classification']).size().unstack(fill_value=0)
    if 'Unexplained' not in summary.columns:
        summary['Unexplained'] = 0
    if 'Explained' not in summary.columns:
        summary['Explained'] = 0
        
    summary['Total'] = summary['Explained'] + summary['Unexplained']
    summary['Explanation Rate'] = summary['Explained'] / summary['Total']
    summary = summary.reset_index()
    
    # 2. Calculate Overall Rates
    overall = df.groupby('model').apply(lambda x: pd.Series({
        'Explained': (x['classification'] == 'Explained').sum(),
        'Total': len(x)
    })).reset_index()
    overall['Explanation Rate'] = overall['Explained'] / overall['Total']
    overall['rule_system'] = 'Overall'
    
    # 3. Combine Data
    plot_data = pd.concat([
        summary[['model', 'rule_system', 'Explanation Rate']], 
        overall[['model', 'rule_system', 'Explanation Rate']]
    ], ignore_index=True)
    
    # Order: Percent -> Rank -> Rank_With_Save -> Overall
    rule_order = sorted(df['rule_system'].unique()) + ['Overall']
    
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(
        data=plot_data,
        x='rule_system', 
        y='Explanation Rate', 
        hue='model',
        order=rule_order,
        palette=['#95a5a6', '#2ecc71'] # Gray for Basic, Green for Bayesian
    )
    
    plt.title('Model Explanation Rate by Era & Overall (Higher is Better)', fontsize=14)
    plt.ylabel('Explanation Rate (Valid Eliminations / Total Eliminations)', fontsize=12)
    plt.xlabel('Rule System', fontsize=12)
    plt.ylim(0, 1.15) # Give space for labels
    
    # Add value labels with correct formatting
    for container in ax.containers:
        # Use lambda to format 0.992 as 99.2%
        labels = [f'{v.get_height():.1%}' for v in container]
        ax.bar_label(container, labels=labels, padding=3, fontsize=10)
        
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "explanation_rate_comparison.png", dpi=300)
    plt.close()


def plot_certainty_distribution(perf_df: pd.DataFrame):
    """
    Plot distribution of Confidence (Certainty) for Explained weeks.
    """
    # Filter Only Explained weeks
    df = perf_df[perf_df['classification'] == 'Explained'].copy()
    
    plt.figure(figsize=(10, 6))
    sns.boxplot(
        data=df,
        x='rule_system',
        y='confidence',
        hue='model',
        palette=['#95a5a6', '#2ecc71']
    )
    
    plt.title('Model Certainty Distribution (Explained Weeks Only)', fontsize=14)
    plt.ylabel('Confidence (Valid Simulations / Total)', fontsize=12)
    plt.xlabel('Rule System', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "certainty_distribution.png", dpi=300)
    plt.close()


def plot_stability_distribution(perf_df: pd.DataFrame):
    """
    Plot distribution of Share Standard Deviation (Stability).
    Lower std dev means higher stability/precision in estimate.
    """
    # Filter Only Explained weeks
    df = perf_df[perf_df['classification'] == 'Explained'].copy()
    
    plt.figure(figsize=(10, 6))
    sns.boxplot(
        data=df,
        x='rule_system',
        y='avg_share_std',
        hue='model',
        palette=['#95a5a6', '#2ecc71']
    )
    
    plt.title('Model Stability: Estimation Uncertainty (Lower is Better)', fontsize=14)
    plt.ylabel('Avg Std Dev of Fan Share Estimate', fontsize=12)
    plt.xlabel('Rule System', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "stability_distribution.png", dpi=300)
    plt.close()


def plot_entropy_distribution(perf_df: pd.DataFrame):
    """
    Plot distribution of Shannon Entropy.
    """
    # Filter Only Explained weeks and Bayesian model
    df = perf_df[(perf_df['classification'] == 'Explained') & (perf_df['model'] == 'Bayesian')].copy()
    
    if df['avg_share_entropy'].isna().all():
        return

    plt.figure(figsize=(10, 6))
    sns.violinplot(
        data=df,
        x='rule_system',
        y='avg_share_entropy',
        color='#2ecc71'
    )
    
    plt.title('Uncertainty Profile using Shannon Entropy (Bayesian Mixture Model)', fontsize=14)
    plt.ylabel('Avg Shannon Entropy (nats)', fontsize=12)
    plt.xlabel('Rule System', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "entropy_distribution.png", dpi=300)
    plt.close()


def plot_fan_trajectory(basic_raw: pd.DataFrame, bayes_raw: pd.DataFrame, 
                       celebrity_name: str, season: int):
    """
    Plot fan share trajectory for a specific celebrity.
    Comparing Basic vs Bayesian estimates with confidence intervals.
    """
    # Filter data
    def get_celeb_data(df):
        mask = (df['season'] == season) & (df['celebrity_name'].str.contains(celebrity_name, case=False, na=False))
        return df[mask].sort_values('week')

    b_data = get_celeb_data(basic_raw)
    y_data = get_celeb_data(bayes_raw)
    
    if len(b_data) == 0 or len(y_data) == 0:
        print(f"[WARN] No data found for {celebrity_name} in Season {season}")
        return

    plt.figure(figsize=(12, 6))
    
    # Plot Basic
    plt.plot(b_data['week'], b_data['estimated_fan_share'], 
             label='Basic Model', color='#95a5a6', linestyle='--', marker='o')
    plt.fill_between(b_data['week'], 
                     b_data['estimated_fan_share'] - b_data['share_std'],
                     b_data['estimated_fan_share'] + b_data['share_std'],
                     color='#95a5a6', alpha=0.2)
    
    # Plot Bayesian
    plt.plot(y_data['week'], y_data['estimated_fan_share'], 
             label='Bayesian Model', color='#2ecc71', linewidth=2, marker='s')
    plt.fill_between(y_data['week'], 
                     y_data['estimated_fan_share'] - y_data['share_std'],
                     y_data['estimated_fan_share'] + y_data['share_std'],
                     color='#2ecc71', alpha=0.3)
    
    plt.title(f'Fan Share Trajectory: {celebrity_name} (Season {season})', fontsize=14)
    plt.xlabel('Week', fontsize=12)
    plt.ylabel('Estimated Fan Vote Share', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    safe_name = celebrity_name.replace(' ', '_')
    plt.savefig(PLOTS_DIR / f"trajectory_{safe_name}_S{season}.png", dpi=300)
    plt.close()


def plot_dual_axis_entropy_share(raw_df: pd.DataFrame, celebrity_name: str, season: int):
    """
    Case Study Plot: Dual-axis chart showing Fan Share (Left) and Entropy (Right).
    Used to visualize the relationship between 'Vote Magnitude' and 'Uncertainty'.
    Only for Bayesian Model.
    """
    # Filter data for Bayesian model
    mask = (raw_df['season'] == season) & \
           (raw_df['celebrity_name'].str.contains(celebrity_name, case=False, na=False)) & \
           (raw_df['model'] == 'Bayesian')
    
    data = raw_df[mask].sort_values('week')
    
    if data.empty:
        print(f"[WARN] No Bayesian data found for {celebrity_name} in Season {season}")
        return
        
    if 'share_entropy' not in data.columns:
        print("[WARN] 'share_entropy' column missing.")
        return

    # Create figure and axis
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # Trace 1: Fan Share (Left Axis)
    color1 = '#2ecc71' # Green
    ax1.set_xlabel('Week', fontsize=12)
    ax1.set_ylabel('Estimated Fan Share (%)', color=color1, fontsize=12)
    l1 = ax1.plot(data['week'], data['estimated_fan_share'], color=color1, 
             marker='o', linewidth=2, label='Fan Share')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(True, alpha=0.3)
    
    # Add fill for std dev
    ax1.fill_between(data['week'], 
                     data['estimated_fan_share'] - data['share_std'],
                     data['estimated_fan_share'] + data['share_std'],
                     color=color1, alpha=0.1)

    # Trace 2: Entropy (Right Axis)
    ax2 = ax1.twinx()
    color2 = '#e74c3c' # Red
    ax2.set_ylabel('Shannon Entropy (nats)\nHigh = Uncertain/Chaotic', color=color2, fontsize=12)
    l2 = ax2.plot(data['week'], data['share_entropy'], color=color2, 
             marker='s', linestyle='--', linewidth=2, label='Uncertainty (Entropy)')
    ax2.tick_params(axis='y', labelcolor=color2)
    ax2.grid(False) # Turn off grid for second axis to avoid clutter
    
    # Combined legend
    lines = l1 + l2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left')

    plt.title(f'Case Study: {celebrity_name} (Season {season})\n'
              f'Relationship between Fan Support and Model Uncertainty', fontsize=14)
    
    safe_name = celebrity_name.replace(' ', '_')
    plt.savefig(PLOTS_DIR / f"case_study_entropy_{safe_name}_S{season}.png", dpi=300)
    plt.close()


# =============================================================================
# Reporting
# =============================================================================
def generate_report(perf_df: pd.DataFrame, explained_diff: pd.DataFrame):
    """
    Generate Markdown report.
    """
    
    # Global Stats
    total_eliminations = len(perf_df[perf_df['classification'] != 'Non-Elimination']) // 2 # div 2 because 2 models
    
    basic_explained = perf_df[(perf_df['model'] == 'Basic') & (perf_df['classification'] == 'Explained')]
    bayes_explained = perf_df[(perf_df['model'] == 'Bayesian') & (perf_df['classification'] == 'Explained')]
    
    basic_rate = len(basic_explained) / total_eliminations if total_eliminations > 0 else 0
    bayes_rate = len(bayes_explained) / total_eliminations if total_eliminations > 0 else 0
    
    recovered_count = len(explained_diff[explained_diff['Basic'] == 'Unexplained'])
    
    with open(REPORT_PATH, 'w') as f:
        f.write("# Model Evaluation Report: Basic vs. Bayesian\n\n")
        f.write("## 1. Executive Summary\n")
        f.write(f"- **Total Elimination Weeks Analyzed**: {total_eliminations}\n")
        f.write(f"- **Basic Model Explanation Rate**: {basic_rate:.1%} ({len(basic_explained)}/{total_eliminations})\n")
        f.write(f"- **Bayesian Model Explanation Rate**: {bayes_rate:.1%} ({len(bayes_explained)}/{total_eliminations})\n")
        f.write(f"- **Net Improvement**: +{(bayes_rate - basic_rate):.1%} ({recovered_count} weeks recovered)\n\n")
        
        f.write("## 2. Key Metrics by Era\n")
        f.write("Evaluation excludes non-elimination weeks (e.g. withdrawals, finals).\n\n")
        
        # Table by Era
        f.write("| Rule System | Basic Rate | Bayesian Rate | Improvement | Avg Certainty (Basic -> Bayes) | Avg Stability (Basic -> Bayes) |\n")
        f.write("|---|---|---|---|---|---|\n")
        
        eras = sorted(perf_df['rule_system'].unique())
        for era in eras:
            # Stats for this era
            era_data = perf_df[perf_df['rule_system'] == era]
            total_basic = len(era_data[(era_data['model'] == 'Basic') & (era_data['classification'] != 'Non-Elimination')])
            
            if total_basic == 0:
                continue
                
            basic_ok_df = era_data[(era_data['model'] == 'Basic') & (era_data['classification'] == 'Explained')]
            bayes_ok_df = era_data[(era_data['model'] == 'Bayesian') & (era_data['classification'] == 'Explained')]
            
            basic_ok = len(basic_ok_df)
            bayes_ok = len(bayes_ok_df)
            
            basic_pct = basic_ok / total_basic
            bayes_pct = bayes_ok / total_basic # Assuming same denominator for fair comparison
            
            # Comparison of secondary metrics
            cert_basic = basic_ok_df['confidence'].mean()
            cert_bayes = bayes_ok_df['confidence'].mean()
            
            stab_basic = basic_ok_df['avg_share_std'].mean()
            stab_bayes = bayes_ok_df['avg_share_std'].mean()
            
            f.write(f"| {era} | {basic_pct:.1%} | {bayes_pct:.1%} | +{(bayes_pct-basic_pct):.1%} | {cert_basic:.3f} -> {cert_bayes:.3f} | {stab_basic:.3f} -> {stab_bayes:.3f} |\n")
        
        f.write("\n\n")
        
        f.write("## 3. Notable Recovered Cases\n")
        f.write("Weeks where Basic Model failed (valid_sims=0) but Bayesian Model succeeded:\n\n")
        f.write("| Season | Week | Result |\n")
        f.write("|---|---|---|\n")
        
        for idx, row in explained_diff[explained_diff['Basic'] == 'Unexplained'].iterrows():
             # Re-fetch classification from Bayesian to be sure
             # Actually explained_diff is intersection of weeks
             f.write(f"| {row['season']} | {row['week']} | Recovered |\n")
            
        f.write("\n## 4. Visualization Index\n")
        f.write("- [Explanation Rate Comparison](plots/explanation_rate_comparison.png)\n")
        f.write("- [Certainty Distribution](plots/certainty_distribution.png)\n")
        f.write("- [Stability Distribution](plots/stability_distribution.png)\n")
        f.write("- [Trajectory: Bobby Bones](plots/trajectory_Bobby_Bones_S27.png)\n")
        f.write("- [Trajectory: Sean Spicer](plots/trajectory_Sean_Spicer_S28.png)\n")
        

# =============================================================================
# Main Pipeline
# =============================================================================
def main():
    print("="*60)
    print("MCM 2026 Problem C - Model Evaluation Pipeline")
    print("="*60)
    
    # 1. Load Data
    print("[INFO] Loading results...")
    try:
        basic_df = load_and_prep_data(BASIC_RESULTS_PATH, "Basic")
        bayes_df = load_and_prep_data(BAYESIAN_RESULTS_PATH, "Bayesian")
        print(f"Loaded: Basic ({len(basic_df)} rows), Bayesian ({len(bayes_df)} rows)")
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        return

    # 2. Compute Metrics Aggregated by Week
    print("[INFO] Computing per-week metrics...")
    basic_perf = compute_week_metrics(basic_df)
    bayes_perf = compute_week_metrics(bayes_df)
    
    perf = pd.concat([basic_perf, bayes_perf], ignore_index=True)
    
    # 3. Analyze Differences
    # Pivot to compare classifications side-by-side
    pivot_perf = perf.pivot(index=['season', 'week'], columns='model', values='classification')
    pivot_perf = pivot_perf.reset_index()
    
    # Identify Recovered Cases: Basic=Unexplained, Bayesian=Explained
    # Also ignore Non-Elminations
    mask_recovered = (pivot_perf['Basic'] == 'Unexplained') & (pivot_perf['Bayesian'] == 'Explained')
    recovered_cases = pivot_perf[mask_recovered]
    
    print(f"[INFO] Found {len(recovered_cases)} recovered weeks (Basic Failed -> Bayesian Success)")
    if len(recovered_cases) > 0:
        print(recovered_cases.head())

    # 4. Generate Visualizations
    print("[INFO] Generating plots...")
    plot_explanation_rate(perf)
    plot_certainty_distribution(perf)
    plot_stability_distribution(perf)
    plot_entropy_distribution(perf)
    
    # Trajectories
    plot_fan_trajectory(basic_df, bayes_df, "Bobby Bones", 27)
    plot_fan_trajectory(basic_df, bayes_df, "Sean Spicer", 28)
    
    # Entropy Heatmaps (Targeted)
    plot_entropy_heatmap(bayes_df, 27) # Bobby Bones Era
    plot_entropy_heatmap(bayes_df, 28) # Sean Spicer Era
    plot_entropy_heatmap(bayes_df, 32) # Xochitl Gomez / Jason Mraz era? (Recent high skill)

    # Duel Axis Case Studies
    plot_dual_axis_entropy_share(bayes_df, "Bobby Bones", 27)
    plot_dual_axis_entropy_share(bayes_df, "Sean Spicer", 28)

    # 5. Generate Report
    print("[INFO] Generating report...")
    generate_report(perf, recovered_cases)
    
    print(f"\n[SUCCESS] Evaluation complete. Report saved to: {REPORT_PATH}")

if __name__ == "__main__":
    main()
