#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MCM 2026 Problem C: Rule Comparison & Counterfactual Analysis
The Multiverse Simulator: Compare 3 elimination rules against historical data.

Core Question: "Does one method favor fan votes more?"
"What if rules were different?"

Universes:
- A: Rank System (Classic) - Seasons 1-2
- B: Percent System (Modern) - Seasons 3-27
- C: Rank + Judges' Save (Hybrid) - Seasons 28+

Author: MCM Team
Date: 2026-01-31
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
from pathlib import Path
import warnings
from typing import Tuple, Dict, List, Optional

warnings.filterwarnings('ignore')

# =============================================================================
# Configuration
# =============================================================================
BASE_DIR = Path(__file__).parent.parent
INPUT_JUDGE_PATH = BASE_DIR / "data_processed" / "dwts_simulation_input.csv"
INPUT_FAN_PATH = BASE_DIR / "results" / "question1" / "full_simulation_bayesian.csv"
OUTPUT_PATH = BASE_DIR / "results" / "question2" / "counterfactual_outcomes.csv"
PLOTS_DIR = BASE_DIR / "results" / "plots" / "question2"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Era definitions for analysis
ERA_MAP = {
    "Rank": list(range(1, 3)),           # Seasons 1-2
    "Percent": list(range(3, 28)),       # Seasons 3-27
    "Rank_With_Save": list(range(28, 35)) # Seasons 28-34
}

# Plot style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


# =============================================================================
# Data Loading & Preparation
# =============================================================================
def load_and_merge_data() -> pd.DataFrame:
    """
    Load judge data and fan estimates, inner join on [season, week, celebrity_name].
    
    Returns:
        Merged DataFrame with all necessary columns.
    """
    print("[INFO] Loading judge data...")
    judge_df = pd.read_csv(INPUT_JUDGE_PATH)
    
    print("[INFO] Loading fan estimates...")
    fan_df = pd.read_csv(INPUT_FAN_PATH)
    
    # Select relevant columns from fan estimates
    fan_cols = ['season', 'week', 'celebrity_name', 'estimated_fan_share', 'share_std']
    fan_subset = fan_df[fan_cols].copy()
    
    # Merge
    merged = pd.merge(
        judge_df,
        fan_subset,
        on=['season', 'week', 'celebrity_name'],
        how='inner'
    )
    
    # Ensure fan share is valid (non-negative, not NaN)
    merged = merged[merged['estimated_fan_share'].notna() & (merged['estimated_fan_share'] >= 0)]
    
    print(f"[INFO] Merged dataset shape: {merged.shape}")
    print(f"[INFO] Seasons: {merged['season'].nunique()}, Weeks: {merged['week'].nunique()}")
    
    return merged


def compute_fan_rank(fan_shares: np.ndarray) -> np.ndarray:
    """
    Compute fan ranks from fan shares using method='average' (ties get average rank).
    
    Args:
        fan_shares: Array of fan shares (higher = better)
        
    Returns:
        Array of ranks (1 = best, higher = worse)
    """
    # Use pandas rank for tie handling
    s = pd.Series(fan_shares)
    # Rank ascending=False: highest share gets rank 1
    ranks = s.rank(method='average', ascending=False).values
    return ranks


def simulate_week_rank(df_week: pd.DataFrame) -> Tuple[str, Dict]:
    """
    System A: Rank System (Classic)
    
    Logic:
    1. Compute fan ranks from estimated_fan_share (method='average')
    2. Total_Rank = Judge_Rank + Fan_Rank
    3. Loser = contestant with MAX Total_Rank (worst)
    4. Tie-breaker: If Total_Rank tied, contestant with worse Fan Rank loses.
    
    Returns:
        loser_name, metadata dict
    """
    if len(df_week) < 2:
        return None, {"error": "Not enough contestants"}
    
    # Compute fan ranks
    fan_shares = df_week['estimated_fan_share'].values
    fan_ranks = compute_fan_rank(fan_shares)
    
    judge_ranks = df_week['judge_rank'].values
    total_ranks = judge_ranks + fan_ranks
    
    # Find contestant(s) with worst total rank
    worst_total = total_ranks.max()
    worst_mask = total_ranks == worst_total
    
    if worst_mask.sum() == 1:
        # Single worst contestant
        loser_idx = np.where(worst_mask)[0][0]
    else:
        # Tie: pick the one with worse fan rank (higher rank number)
        tied_indices = np.where(worst_mask)[0]
        tied_fan_ranks = fan_ranks[tied_indices]
        # Higher fan rank number = worse
        worst_fan = tied_fan_ranks.max()
        # If still tied (unlikely), pick first
        loser_rel_idx = np.where(tied_fan_ranks == worst_fan)[0][0]
        loser_idx = tied_indices[loser_rel_idx]
    
    loser_name = df_week.iloc[loser_idx]['celebrity_name']
    
    metadata = {
        "total_ranks": total_ranks.tolist(),
        "fan_ranks": fan_ranks.tolist(),
        "judge_ranks": judge_ranks.tolist()
    }
    
    return loser_name, metadata


def simulate_week_percent(df_week: pd.DataFrame) -> Tuple[str, Dict]:
    """
    System B: Percent System (Modern)
    
    Logic:
    1. Total_Share = Judge_Share + Estimated_Fan_Share
    2. Loser = contestant with MIN Total_Share
    
    Returns:
        loser_name, metadata dict
    """
    if len(df_week) < 2:
        return None, {"error": "Not enough contestants"}
    
    judge_shares = df_week['judge_share'].values
    fan_shares = df_week['estimated_fan_share'].values
    total_shares = judge_shares + fan_shares
    
    loser_idx = total_shares.argmin()
    loser_name = df_week.iloc[loser_idx]['celebrity_name']
    
    metadata = {
        "total_shares": total_shares.tolist(),
        "judge_shares": judge_shares.tolist(),
        "fan_shares": fan_shares.tolist()
    }
    
    return loser_name, metadata


def simulate_week_rank_save(df_week: pd.DataFrame) -> Tuple[str, Dict]:
    """
    System C: Rank + Judges' Save (Hybrid)
    
    Logic:
    1. Compute Total_Rank (same as System A)
    2. Identify Bottom 2:
       - If >2 people tie at bottom, sort by Fan Rank (worse first) to pick specific Bottom 2
    3. The Save: Compare normalized_score (Judge Score) of Bottom 2
       - Higher Judge Score -> SAVED
       - Lower Judge Score -> ELIMINATED
       - If Judge Scores equal, contestant with worse Fan Rank goes home
    
    Returns:
        loser_name, metadata dict
    """
    if len(df_week) < 2:
        return None, {"error": "Not enough contestants"}
    
    # Step 1: Compute Total_Rank (same as System A)
    fan_shares = df_week['estimated_fan_share'].values
    fan_ranks = compute_fan_rank(fan_shares)
    judge_ranks = df_week['judge_rank'].values
    total_ranks = judge_ranks + fan_ranks
    
    # Step 2: Identify Bottom 2
    # Sort contestants by total_rank (descending: worst first)
    # Use fan_rank as secondary sort (worse fan rank first)
    df_sorted = df_week.copy()
    df_sorted['total_rank'] = total_ranks
    df_sorted['fan_rank'] = fan_ranks
    
    # Sort: total_rank descending, then fan_rank descending (higher rank number = worse)
    df_sorted = df_sorted.sort_values(
        ['total_rank', 'fan_rank'], 
        ascending=[False, False]
    ).reset_index(drop=True)
    
    # Bottom 2 are the first two rows
    bottom2 = df_sorted.iloc[:2]
    
    # Step 3: The Save
    judge_scores = bottom2['normalized_score'].values
    fan_ranks_b2 = bottom2['fan_rank'].values
    
    if judge_scores[0] > judge_scores[1]:
        # Contestant 0 has higher judge score -> saved, contestant 1 eliminated
        loser_idx_in_b2 = 1
    elif judge_scores[1] > judge_scores[0]:
        loser_idx_in_b2 = 0
    else:
        # Judge scores equal, compare fan ranks
        if fan_ranks_b2[0] > fan_ranks_b2[1]:
            # Contestant 0 has worse fan rank
            loser_idx_in_b2 = 0
        else:
            loser_idx_in_b2 = 1
    
    loser_name = bottom2.iloc[loser_idx_in_b2]['celebrity_name']
    
    metadata = {
        "total_ranks": total_ranks.tolist(),
        "fan_ranks": fan_ranks.tolist(),
        "judge_ranks": judge_ranks.tolist(),
        "bottom2_names": bottom2['celebrity_name'].tolist(),
        "bottom2_judge_scores": judge_scores.tolist(),
        "bottom2_fan_ranks": fan_ranks_b2.tolist()
    }
    
    return loser_name, metadata


# =============================================================================
# Simulation Loop with Enhanced Metrics
# =============================================================================
def run_counterfactual_simulation_with_metrics(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run all three rule systems for each (season, week) group.
    Also compute per-contestant-week rankings for each system.
    
    Returns:
        results_df: weekly outcomes
        rank_df: per-contestant-week rankings for each system
    """
    print("[INFO] Running counterfactual simulation with metrics...")
    
    results = []
    rank_records = []
    
    # Group by season and week
    grouped = df.groupby(['season', 'week'])
    
    for (season, week), group in grouped:
        # Skip weeks with insufficient contestants
        if len(group) < 2:
            continue
        
        # Get actual eliminated contestant (if any)
        actual_loser = None
        if group['is_eliminated'].any():
            actual_loser = group[group['is_eliminated']]['celebrity_name'].iloc[0]
        
        # Determine rule system based on season
        if season <= 2:
            rule_system = "Rank"
        elif season <= 27:
            rule_system = "Percent"
        else:
            rule_system = "Rank_With_Save"
        
        # Simulate with each system
        rank_loser, rank_meta = simulate_week_rank(group)
        percent_loser, percent_meta = simulate_week_percent(group)
        save_loser, save_meta = simulate_week_rank_save(group)
        
        # Determine if any reversal occurred
        is_flipped = (rank_loser != percent_loser)
        
        results.append({
            'season': season,
            'week': week,
            'n_contestants': len(group),
            'rule_system': rule_system,
            'actual_loser': actual_loser,
            'rank_loser': rank_loser,
            'percent_loser': percent_loser,
            'save_loser': save_loser,
            'is_flipped': is_flipped
        })
        
        # Compute rankings for each contestant under each system
        # System A: Rank by total_rank (lower is better)
        fan_shares = group['estimated_fan_share'].values
        fan_ranks = compute_fan_rank(fan_shares)
        judge_ranks = group['judge_rank'].values
        total_ranks = judge_ranks + fan_ranks
        # Ranking: 1 = best (lowest total_rank)
        rank_rank = pd.Series(total_ranks).rank(method='average', ascending=True).values
        
        # System B: Rank by total_share (higher is better)
        judge_shares = group['judge_share'].values
        total_shares = judge_shares + fan_shares
        percent_rank = pd.Series(total_shares).rank(method='average', ascending=False).values
        
        # System C: Rank by total_rank (same as System A) but with save adjustment?
        # For ranking correlation, we use total_rank as well (save only affects elimination)
        save_rank = rank_rank.copy()
        
        # Record per contestant - use positional index
        # Reset group index to ensure alignment with arrays
        group_reset = group.reset_index(drop=True)
        for pos_idx, row in group_reset.iterrows():
            rank_records.append({
                'season': season,
                'week': week,
                'celebrity_name': row['celebrity_name'],
                'normalized_score': row['normalized_score'],
                'judge_share': row['judge_share'],
                'estimated_fan_share': row['estimated_fan_share'],
                'rank_rank': rank_rank[pos_idx],
                'percent_rank': percent_rank[pos_idx],
                'save_rank': save_rank[pos_idx],
                'is_eliminated': row['is_eliminated'],
                'is_bottom3_judge': False,  # will be computed later
                'rule_system': rule_system
            })
    
    results_df = pd.DataFrame(results)
    rank_df = pd.DataFrame(rank_records)
    
    # Determine bottom 3 judge scorers per week
    rank_df['is_bottom3_judge'] = False
    for (season, week), group in rank_df.groupby(['season', 'week']):
        # Sort by normalized_score ascending (lower = worse)
        sorted_indices = group['normalized_score'].argsort().values
        k = min(3, len(group))
        bottom_indices = sorted_indices[:k]
        # Get original indices in rank_df
        orig_indices = group.iloc[bottom_indices].index
        rank_df.loc[orig_indices, 'is_bottom3_judge'] = True
    
    print(f"[INFO] Simulated {len(results_df)} weeks")
    print(f"[INFO] Generated {len(rank_df)} contestant-week rankings")
    
    return results_df, rank_df


# =============================================================================
# Evaluation Metrics (Complete)
# =============================================================================
def compute_reversal_rate(results_df: pd.DataFrame) -> float:
    """Proportion of weeks where Rank result != Percent result."""
    if len(results_df) == 0:
        return 0.0
    return results_df['is_flipped'].mean()


def compute_fan_power_index(rank_df: pd.DataFrame) -> Dict[str, float]:
    """
    Compute Spearman correlation between final ranking and fan share.
    
    For each system, compute correlation across all contestant-weeks.
    Also compute by era.
    
    Returns:
        Dict with FPI for each rule system and era
    """
    print("[INFO] Computing Fan Power Index...")
    
    # Overall FPI for each system
    systems = ['rank', 'percent', 'save']
    fpi = {}
    
    for sys in systems:
        rank_col = f'{sys}_rank'
        # Spearman correlation between rank and fan share (higher fan share should correlate with better rank (lower number))
        # Since rank: 1 = best, we expect negative correlation (higher fan share -> lower rank number)
        # Use absolute value for FPI magnitude
        corr, pval = spearmanr(rank_df[rank_col], rank_df['estimated_fan_share'])
        fpi[sys.capitalize()] = abs(corr) if not np.isnan(corr) else 0.0
    
    # FPI by era
    eras = ['Rank', 'Percent', 'Rank_With_Save']
    for era in eras:
        era_mask = rank_df['rule_system'] == era
        if era_mask.sum() > 0:
            for sys in systems:
                rank_col = f'{sys}_rank'
                corr, _ = spearmanr(rank_df.loc[era_mask, rank_col], 
                                    rank_df.loc[era_mask, 'estimated_fan_share'])
                key = f"{sys.capitalize()}_{era}"
                fpi[key] = abs(corr) if not np.isnan(corr) else 0.0
    
    return fpi


def compute_merit_safety_rate(results_df: pd.DataFrame, rank_df: pd.DataFrame) -> Dict[str, float]:
    """
    Compute % of times a "Bottom 3 Judge Scorer" survives under each system.
    
    For each system, check if the loser is among bottom 3 judge scorers.
    Survival rate = (bottom 3 contestants not eliminated) / total bottom 3 contestants.
    
    Returns:
        Dict with survival rates for each system
    """
    print("[INFO] Computing Merit Safety Rate...")
    
    # For each system, compute survival rate
    systems = ['rank', 'percent', 'save']
    safety_rates = {}
    
    for sys in systems:
        loser_col = f'{sys}_loser'
        # Filter weeks where we have a loser
        valid_weeks = results_df[results_df[loser_col].notna()].copy()
        
        bottom3_survived = 0
        bottom3_total = 0
        
        for _, row in valid_weeks.iterrows():
            season = row['season']
            week = row['week']
            loser = row[loser_col]
            
            # Get all contestants in this week that are bottom 3
            week_bottom3 = rank_df[
                (rank_df['season'] == season) & 
                (rank_df['week'] == week) & 
                (rank_df['is_bottom3_judge'])
            ]
            
            bottom3_total += len(week_bottom3)
            
            # Count how many bottom 3 contestants survived (i.e., not the loser)
            survived = week_bottom3[week_bottom3['celebrity_name'] != loser]
            bottom3_survived += len(survived)
        
        safety_rate = bottom3_survived / bottom3_total if bottom3_total > 0 else 0.0
        safety_rates[sys.capitalize()] = safety_rate
    
    return safety_rates


def validate_system_c(results_df: pd.DataFrame) -> float:
    """
    Validate System C against actual history for Seasons 28-34.
    
    Returns:
        Accuracy rate (proportion of weeks where simulated loser == actual loser)
    """
    # Filter for seasons 28-34
    validation_df = results_df[
        (results_df['season'] >= 28) & 
        (results_df['season'] <= 34) &
        (results_df['actual_loser'].notna())
    ].copy()
    
    if len(validation_df) == 0:
        return 0.0
    
    # Compare save_loser vs actual_loser
    matches = validation_df['save_loser'] == validation_df['actual_loser']
    accuracy = matches.mean()
    
    print(f"[VALIDATION] System C accuracy (Seasons 28-34): {accuracy:.2%} ({matches.sum()}/{len(validation_df)})")
    
    return accuracy


# =============================================================================
# Visualization Functions (Complete)
# =============================================================================
def plot_fan_bias_comparison(fpi: Dict[str, float]):
    """
    Create bar chart of Fan Power Index by Era & Rule.
    """
    # Extract data for plotting
    # We'll show overall FPI for each system, and maybe by era
    systems = ['Rank', 'Percent', 'Save']
    overall_values = [fpi.get(sys, 0.0) for sys in systems]
    
    # Era-specific values
    era_values = {}
    for era in ['Rank', 'Percent', 'Rank_With_Save']:
        era_values[era] = [fpi.get(f'{sys}_{era}', 0.0) for sys in ['Rank', 'Percent', 'Save']]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Overall FPI
    ax1 = axes[0]
    x_pos = np.arange(len(systems))
    ax1.bar(x_pos, overall_values, color=['#3498db', '#2ecc71', '#e74c3c'])
    ax1.set_xlabel('Rule System', fontsize=12)
    ax1.set_ylabel('Fan Power Index (|Spearman ρ|)', fontsize=12)
    ax1.set_title('Overall Fan Power Index\nHigher = More Fan Influence', fontsize=14)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(systems)
    ax1.set_ylim(0, 1)
    # Add value labels
    for i, v in enumerate(overall_values):
        ax1.text(i, v + 0.02, f'{v:.3f}', ha='center', fontsize=10)
    
    # Plot 2: FPI by Era
    ax2 = axes[1]
    width = 0.25
    x = np.arange(len(systems))
    for idx, (era, values) in enumerate(era_values.items()):
        offset = (idx - 1) * width
        ax2.bar(x + offset, values, width, label=era, alpha=0.8)
    
    ax2.set_xlabel('Rule System', fontsize=12)
    ax2.set_ylabel('Fan Power Index (|Spearman ρ|)', fontsize=12)
    ax2.set_title('Fan Power Index by Historical Era', fontsize=14)
    ax2.set_xticks(x)
    ax2.set_xticklabels(systems)
    ax2.set_ylim(0, 1)
    ax2.legend(title='Era')
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "q2_fan_bias_comparison.png", dpi=300)
    plt.close()
    
    print("[INFO] Saved fan bias comparison plot")


def plot_merit_safety(safety_rates: Dict[str, float]):
    """
    Create bar chart comparing Merit Safety Rates.
    """
    systems = ['Rank', 'Percent', 'Save']
    values = [safety_rates.get(sys, 0.0) for sys in systems]
    
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x=systems, y=values, palette=['#3498db', '#2ecc71', '#e74c3c'])
    
    plt.title('Merit Safety Rate: Protection for Bottom 3 Judge Scorers', fontsize=14)
    plt.ylabel('Survival Rate (% Bottom 3 Contestants Not Eliminated)', fontsize=12)
    plt.xlabel('Rule System', fontsize=12)
    plt.ylim(0, 1)
    
    # Add value labels
    for i, v in enumerate(values):
        ax.text(i, v + 0.02, f'{v:.2%}', ha='center', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "q2_merit_safety.png", dpi=300)
    plt.close()
    
    print("[INFO] Saved merit safety plot")


def plot_bobby_bones_survival(rank_df: pd.DataFrame, results_df: pd.DataFrame):
    """
    Plot Bobby Bones' virtual rank across systems.
    
    Requirements:
    - Filter Season 27, Bobby Bones
    - Step plot of his "Virtual Rank" (Universe A vs B vs C) over time
    - Horizontal line at "Bottom 2" threshold
    - Highlight week he would be eliminated under System C
    """
    print("[INFO] Creating Bobby Bones survival plot...")
    
    # Filter data for Season 27, Bobby Bones
    bones_data = rank_df[
        (rank_df['season'] == 27) & 
        (rank_df['celebrity_name'].str.contains('Bobby Bones', case=False, na=False))
    ].copy()
    
    if bones_data.empty:
        print("[WARNING] Bobby Bones not found in Season 27")
        return
    
    # Get weekly results for Season 27
    season_results = results_df[results_df['season'] == 27].copy()
    
    # Determine when he would be eliminated under System C
    # Find first week where save_loser == Bobby Bones
    elimination_week = None
    for _, row in season_results.iterrows():
        if row['save_loser'] and 'Bobby Bones' in row['save_loser']:
            elimination_week = row['week']
            break
    
    weeks = bones_data['week'].unique()
    weeks.sort()
    
    # Extract his ranks for each system
    rank_vals = []
    percent_vals = []
    save_vals = []
    
    for week in weeks:
        week_data = bones_data[bones_data['week'] == week]
        if len(week_data) > 0:
            rank_vals.append(week_data['rank_rank'].iloc[0])
            percent_vals.append(week_data['percent_rank'].iloc[0])
            save_vals.append(week_data['save_rank'].iloc[0])
        else:
            rank_vals.append(np.nan)
            percent_vals.append(np.nan)
            save_vals.append(np.nan)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Step plot
    ax.step(weeks, rank_vals, where='mid', label='System A (Rank)', linewidth=2, marker='o')
    ax.step(weeks, percent_vals, where='mid', label='System B (Percent)', linewidth=2, marker='s')
    ax.step(weeks, save_vals, where='mid', label='System C (Rank+Save)', linewidth=2, marker='^')
    
    # Bottom 2 threshold (rank > n_contestants - 1)
    # Need n_contestants per week
    for week in weeks:
        n_contestants = season_results[season_results['week'] == week]['n_contestants'].iloc[0]
        bottom2_threshold = n_contestants - 1  # If rank > this, in bottom 2
        ax.axhline(y=bottom2_threshold + 0.5, xmin=(week-1)/len(weeks), xmax=week/len(weeks), 
                   color='red', linestyle='--', alpha=0.3)
    
    # Highlight elimination week
    if elimination_week:
        ax.axvline(x=elimination_week, color='red', linestyle='-', alpha=0.7, linewidth=2)
        ax.text(elimination_week, max(save_vals) * 0.8, f'Eliminated Week {elimination_week}', 
                rotation=90, ha='right', va='top', fontsize=10, color='red')
    
    ax.set_xlabel('Week', fontsize=12)
    ax.set_ylabel('Virtual Rank (1 = Best)', fontsize=12)
    ax.set_title('Bobby Bones (Season 27) - Virtual Rank Under Different Rules', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Invert y-axis so rank 1 is at top
    ax.invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "q2_bobby_bones_survival.png", dpi=300)
    plt.close()
    
    print(f"[INFO] Saved Bobby Bones survival plot (eliminated week: {elimination_week})")


# =============================================================================
# Main Pipeline
# =============================================================================
def main():
    print("="*70)
    print("MCM 2026 Problem C - Rule Comparison (Multiverse Simulator)")
    print("="*70)
    
    # Step 1: Load and merge data
    df = load_and_merge_data()
    
    # Step 2: Run counterfactual simulation with metrics
    results_df, rank_df = run_counterfactual_simulation_with_metrics(df)
    
    # Step 3: Compute metrics
    reversal_rate = compute_reversal_rate(results_df)
    print(f"\n[METRICS] Reversal Rate (Rank vs Percent): {reversal_rate:.2%}")
    
    fpi = compute_fan_power_index(rank_df)
    safety_rates = compute_merit_safety_rate(results_df, rank_df)
    validation_accuracy = validate_system_c(results_df)
    
    # Print metrics
    print("\n[METRICS] Fan Power Index (|Spearman ρ|):")
    for key, value in fpi.items():
        if '_' not in key:  # Overall metrics
            print(f"  - {key}: {value:.3f}")
    
    print("\n[METRICS] Merit Safety Rate (Bottom 3 Judge Scorers):")
    for sys, rate in safety_rates.items():
        print(f"  - {sys}: {rate:.2%}")
    
    # Step 4: Save results
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(OUTPUT_PATH, index=False)
    print(f"[INFO] Saved counterfactual outcomes to: {OUTPUT_PATH}")
    
    # Save rank data for further analysis
    rank_output = BASE_DIR / "results" / "question2" / "counterfactual_rankings.csv"
    rank_df.to_csv(rank_output, index=False)
    print(f"[INFO] Saved contestant rankings to: {rank_output}")
    
    # Step 5: Generate visualizations
    plot_fan_bias_comparison(fpi)
    plot_merit_safety(safety_rates)
    plot_bobby_bones_survival(rank_df, results_df)
    
    # Step 6: Print summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Total weeks simulated: {len(results_df)}")
    print(f"Reversal Rate (Rank vs Percent): {reversal_rate:.2%}")
    print(f"System C Validation Accuracy (S28-34): {validation_accuracy:.2%}")
    print(f"\nOutput files:")
    print(f"  - Counterfactual outcomes: {OUTPUT_PATH}")
    print(f"  - Contestant rankings: {rank_output}")
    print(f"  - Fan bias plot: {PLOTS_DIR / 'q2_fan_bias_comparison.png'}")
    print(f"  - Merit safety plot: {PLOTS_DIR / 'q2_merit_safety.png'}")
    print(f"  - Bobby Bones plot: {PLOTS_DIR / 'q2_bobby_bones_survival.png'}")
    print("\n" + "="*70)


if __name__ == "__main__":
    main()