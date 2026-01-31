#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MCM 2026 Problem C: Three-Stage Bayesian Diagnostic Analysis
With Distribution A/B/C Testing (Uniform vs Pareto vs Exponential)

Phase 1: Pure Rationality Baseline (lambda=0) - Collect Survival Deficits
Phase 2: Distribution Identification - Visualize Skill-Bias Pattern (FIXED)
Phase 3: Distribution A/B/C Test - Compare Uniform, Pareto, Exponential

Author: MCM Team
Date: 2026-02-01
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
from typing import Tuple, List, Dict, Optional
import warnings

warnings.filterwarnings('ignore')

# =============================================================================
# Configuration
# =============================================================================
BASE_DIR = Path(__file__).parent.parent
INPUT_PATH = BASE_DIR / "data_processed" / "dwts_simulation_input.csv"
PLOTS_DIR = BASE_DIR / "results" / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Simulation parameters
N_SIMULATIONS = 300
RANDOM_SEED = 42

# Bayesian parameters
INITIAL_ALPHA = 1.0
LEARNING_RATE = 0.4
EVIDENCE_BOOST = 5.0
MIN_ALPHA = 0.1
SKILL_IMPACT_FACTOR = 0.3

# Plot style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


# =============================================================================
# WeekState Dataclass (from monte_carlo_bayesian_dirichlet.py)
# =============================================================================
from dataclasses import dataclass

@dataclass
class WeekState:
    """State container for week-to-week prior propagation."""
    alpha: np.ndarray
    contestant_names: List[str]
    n_contestants: int


def align_prior_to_current(
    prev_state: WeekState,
    current_names: List[str]
) -> np.ndarray:
    """
    Align previous week's posterior to current week's contestants.
    EXACT copy from monte_carlo_bayesian_dirichlet.py
    """
    aligned_alpha = []
    
    for name in current_names:
        if name in prev_state.contestant_names:
            # Continuing contestant: inherit accumulated α
            idx = prev_state.contestant_names.index(name)
            aligned_alpha.append(prev_state.alpha[idx])
        else:
            # New contestant (rare): start with default prior
            aligned_alpha.append(INITIAL_ALPHA)
    
    aligned_alpha = np.array(aligned_alpha)
    
    # Scale to maintain relative proportions after elimination
    remaining_ratio = len(current_names) / prev_state.n_contestants
    if remaining_ratio < 1 and remaining_ratio > 0:
        scale_factor = min(1.0 / remaining_ratio, 1.3)
        aligned_alpha = aligned_alpha * scale_factor
    
    aligned_alpha = np.maximum(aligned_alpha, MIN_ALPHA)
    
    return aligned_alpha


def update_prior_with_evidence(
    current_alpha: np.ndarray,
    evidence: np.ndarray,
    learning_rate: float = LEARNING_RATE
) -> np.ndarray:
    """
    Bayesian update: accumulate evidence into prior.
    EXACT copy from monte_carlo_bayesian_dirichlet.py
    """
    evidence_scaled = evidence * EVIDENCE_BOOST
    new_alpha = current_alpha + learning_rate * evidence_scaled
    new_alpha = np.maximum(new_alpha, MIN_ALPHA)
    return new_alpha


# =============================================================================
# Core Simulation Functions
# =============================================================================
def shares_to_ranks(shares: np.ndarray) -> np.ndarray:
    """Convert shares to ranks (1 = highest share = best)."""
    order = np.argsort(-shares, axis=1)
    ranks = np.argsort(order, axis=1) + 1
    return ranks.astype(float)


def generate_mixed_shares(alpha: np.ndarray, n_sims: int, 
                          chaos_weight: float, 
                          distribution: str = 'uniform') -> np.ndarray:
    """
    Generate fan shares using Mixture Model with configurable chaos distribution.
    
    Args:
        distribution: 'uniform', 'pareto', or 'exponential'
    """
    n_chaos = int(n_sims * chaos_weight)
    n_skill = n_sims - n_chaos
    
    # 1. Rational/Skill Component
    samples_skill = np.random.dirichlet(alpha, size=n_skill)
    
    if n_chaos == 0:
        return samples_skill
    
    # 2. Chaos Component - varies by distribution type
    if distribution == 'pareto':
        # Pareto with a=3.0 (milder tail than a=2.0)
        alpha_inverted = 1.0 / (alpha + 0.1)
        pareto_noise = np.random.pareto(a=3.0, size=len(alpha)) + 1
        chaos_alpha = alpha_inverted * pareto_noise
        chaos_alpha = np.maximum(chaos_alpha, MIN_ALPHA)
    elif distribution == 'exponential':
        # Exponential: mild tail, moderate underdog bias
        alpha_inverted = 1.0 / (alpha + 0.1)
        exp_noise = np.random.exponential(scale=1.0, size=len(alpha))
        chaos_alpha = alpha_inverted * (exp_noise + 0.5)
        chaos_alpha = np.maximum(chaos_alpha, MIN_ALPHA)
    else:  # uniform (baseline)
        chaos_alpha = np.ones_like(alpha)
    
    samples_chaos = np.random.dirichlet(chaos_alpha, size=n_chaos)
    
    return np.concatenate([samples_skill, samples_chaos], axis=0)


def apply_rank_rule(judge_ranks: np.ndarray, fan_shares: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """RANK elimination rule (Seasons 1-2)."""
    fan_ranks = shares_to_ranks(fan_shares)
    total_scores = judge_ranks + fan_ranks
    eliminated_idx = np.argmax(total_scores + fan_ranks * 0.001, axis=1)
    return eliminated_idx, total_scores


def apply_percent_rule(judge_shares: np.ndarray, fan_shares: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """PERCENT elimination rule (Seasons 3-27)."""
    total_scores = judge_shares + fan_shares
    eliminated_idx = np.argmin(total_scores, axis=1)
    return eliminated_idx, total_scores


def apply_rank_with_save_rule(judge_ranks: np.ndarray, judge_scores: np.ndarray,
                               fan_shares: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """RANK_WITH_SAVE elimination rule (Seasons 28+)."""
    n_sims, n_contestants = fan_shares.shape
    
    if n_contestants < 2:
        return np.zeros(n_sims, dtype=int), fan_shares
    
    fan_ranks = shares_to_ranks(fan_shares)
    total_scores = judge_ranks + fan_ranks + fan_ranks * 1e-6
    
    if n_contestants == 2:
        if judge_scores[0] < judge_scores[1]:
            return np.zeros(n_sims, dtype=int), total_scores
        elif judge_scores[1] < judge_scores[0]:
            return np.ones(n_sims, dtype=int), total_scores
        else:
            return np.argmax(fan_ranks, axis=1), total_scores
    
    sorted_indices = np.argsort(-total_scores, axis=1)
    bottom2_first = sorted_indices[:, 0]
    bottom2_second = sorted_indices[:, 1]
    
    judge_scores_first = judge_scores[bottom2_first]
    judge_scores_second = judge_scores[bottom2_second]
    
    eliminated_idx = np.where(
        judge_scores_first < judge_scores_second,
        bottom2_first,
        np.where(judge_scores_second < judge_scores_first, bottom2_second, bottom2_first)
    )
    
    return eliminated_idx, total_scores


# =============================================================================
# Phase 1: Pure Rationality Baseline
# =============================================================================
def compute_rational_score(week_data: pd.DataFrame, prev_shares: Optional[np.ndarray] = None) -> np.ndarray:
    """Compute Rational Score = Judge Performance + Momentum"""
    judge_scores = week_data['normalized_score'].values.astype(float)
    
    if prev_shares is not None and len(prev_shares) == len(judge_scores):
        momentum = prev_shares - np.mean(prev_shares)
    else:
        momentum = np.zeros_like(judge_scores)
    
    rational_score = judge_scores + 0.2 * momentum
    return rational_score


def compute_survival_deficit(week_data: pd.DataFrame, 
                              predicted_scores: np.ndarray,
                              rule_system: str) -> pd.DataFrame:
    """Compute Survival Deficit Gap for each surviving contestant."""
    n_contestants = len(week_data)
    contestants = week_data['celebrity_name'].values
    is_eliminated = week_data['is_eliminated'].values
    
    mu = np.mean(predicted_scores)
    sigma = np.std(predicted_scores, ddof=1) if np.std(predicted_scores) > 1e-6 else 1.0
    z_scores = (predicted_scores - mu) / sigma
    
    records = []
    
    if rule_system in ['Rank', 'Rank_With_Save']:
        predicted_ranks = np.argsort(np.argsort(-predicted_scores)) + 1
        
        if rule_system == 'Rank_With_Save':
            safe_threshold_rank = n_contestants - 1
        else:
            safe_threshold_rank = n_contestants
        
        for i, (name, eliminated, z, rank) in enumerate(zip(contestants, is_eliminated, z_scores, predicted_ranks)):
            if eliminated:
                continue
            
            gap = rank - (safe_threshold_rank - 1)
            predicted_elimination = (rank >= safe_threshold_rank)
            
            records.append({
                'contestant': name,
                'Rational_Score_Z': z,
                'Predicted_Rank': rank,
                'Safe_Threshold': safe_threshold_rank,
                'Predicted_Elimination': predicted_elimination,
                'Survival_Deficit_Gap': max(0, gap),
                'Era': 'Rank' if rule_system == 'Rank' else 'Rank_With_Save',
                'Gap_Type': 'rank'  # For independent normalization
            })
    
    else:  # Percent Era (S3-27)
        eliminated_scores = predicted_scores[is_eliminated]
        if len(eliminated_scores) > 0:
            eliminated_score = eliminated_scores[0]
        else:
            eliminated_score = np.min(predicted_scores)
        
        for i, (name, eliminated, z, score) in enumerate(zip(contestants, is_eliminated, z_scores, predicted_scores)):
            if eliminated:
                continue
            
            gap = eliminated_score - score
            predicted_elimination = (score < eliminated_score)
            
            records.append({
                'contestant': name,
                'Rational_Score_Z': z,
                'Predicted_Score': score,
                'Eliminated_Score': eliminated_score,
                'Predicted_Elimination': predicted_elimination,
                'Survival_Deficit_Gap': max(0, gap),
                'Era': 'Percent',
                'Gap_Type': 'percent'  # For independent normalization
            })
    
    return pd.DataFrame(records)


def run_diagnostics(df: pd.DataFrame) -> pd.DataFrame:
    """Phase 1: Run Pure Rationality baseline and collect Survival Deficits."""
    print("\n" + "="*70)
    print("   PHASE 1: Pure Rationality Baseline (lambda=0)")
    print("="*70)
    
    np.random.seed(RANDOM_SEED)
    
    all_diagnostics = []
    seasons = sorted(df['season'].unique())
    
    for season in tqdm(seasons, desc="Processing seasons"):
        season_data = df[df['season'] == season].copy()
        rule_system = season_data['rule_system'].iloc[0]
        weeks = sorted(season_data['week'].unique())
        
        prev_shares = None
        
        for week in weeks:
            week_data = season_data[season_data['week'] == week].copy()
            week_data = week_data.sort_values('celebrity_name').reset_index(drop=True)
            
            if not week_data['is_eliminated'].any():
                continue
            
            rational_scores = compute_rational_score(week_data, prev_shares)
            week_diag = compute_survival_deficit(week_data, rational_scores, rule_system)
            
            if len(week_diag) > 0:
                week_diag['season'] = season
                week_diag['week'] = week
                all_diagnostics.append(week_diag)
            
            prev_shares = week_data['normalized_score'].values
    
    df_diagnostics = pd.concat(all_diagnostics, ignore_index=True)
    
    print(f"\n   Total survivor records: {len(df_diagnostics)}")
    era_counts = df_diagnostics.groupby('Era').size()
    print(f"   Records by Era: {era_counts.to_dict()}")
    
    anomalies = df_diagnostics[df_diagnostics['Survival_Deficit_Gap'] > 0]
    print(f"   Anomalies (Gap > 0): {len(anomalies)}")
    print(f"   Anomalies by Era: {anomalies.groupby('Era').size().to_dict()}")
    
    return df_diagnostics


# =============================================================================
# Phase 2: Distribution Identification (FIXED: Independent Normalization)
# =============================================================================
def plot_skill_bias_scatter(df_diagnostics: pd.DataFrame) -> None:
    """Phase 2: Scatter plot with INDEPENDENT normalization for Rank vs Percent gaps."""
    print("\n" + "="*70)
    print("   PHASE 2: Distribution Identification (FIXED)")
    print("="*70)
    
    anomalies = df_diagnostics[df_diagnostics['Survival_Deficit_Gap'] > 0].copy()
    
    if len(anomalies) == 0:
        print("   [WARNING] No anomalies found.")
        return
    
    # INDEPENDENT NORMALIZATION by Gap Type
    rank_mask = anomalies['Gap_Type'] == 'rank'
    percent_mask = anomalies['Gap_Type'] == 'percent'
    
    # Normalize Rank gaps independently
    rank_gaps = anomalies.loc[rank_mask, 'Survival_Deficit_Gap']
    if len(rank_gaps) > 0 and rank_gaps.max() > 0:
        anomalies.loc[rank_mask, 'Normalized_Gap'] = rank_gaps / rank_gaps.max()
    else:
        anomalies.loc[rank_mask, 'Normalized_Gap'] = 0.5
    
    # Normalize Percent gaps independently
    percent_gaps = anomalies.loc[percent_mask, 'Survival_Deficit_Gap']
    if len(percent_gaps) > 0 and percent_gaps.max() > 0:
        anomalies.loc[percent_mask, 'Normalized_Gap'] = percent_gaps / percent_gaps.max()
    else:
        anomalies.loc[percent_mask, 'Normalized_Gap'] = 0.5
    
    # DEBUG OUTPUT
    n_rank = rank_mask.sum()
    n_percent = percent_mask.sum()
    print(f"   DEBUG: Plotting {n_rank} Rank-Era points and {n_percent} Percent-Era points.")
    
    if n_percent == 0:
        print("   [ERROR] Percent-Era points still missing!")
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    era_colors = {'Rank': '#e74c3c', 'Percent': '#3498db', 'Rank_With_Save': '#2ecc71'}
    era_order = ['Rank', 'Percent', 'Rank_With_Save']
    
    for era in era_order:
        era_data = anomalies[anomalies['Era'] == era]
        if len(era_data) > 0:
            ax.scatter(era_data['Rational_Score_Z'], era_data['Normalized_Gap'],
                       alpha=0.6, s=80, label=f'{era} Era (n={len(era_data)})',
                       color=era_colors.get(era, '#95a5a6'), edgecolors='white', linewidth=0.5)
    
    # Trend line
    x = anomalies['Rational_Score_Z'].values
    y = anomalies['Normalized_Gap'].values
    
    try:
        coefs = np.polyfit(x, y, 2)
        poly = np.poly1d(coefs)
        x_smooth = np.linspace(x.min(), x.max(), 100)
        ax.plot(x_smooth, poly(x_smooth), 'k--', linewidth=2, alpha=0.7, label='Trend')
    except:
        pass
    
    ax.set_xlabel('Rational Score (Z-score)\n<-- Weaker | Stronger -->', fontsize=12)
    ax.set_ylabel('Normalized Survival Gap (0-1)', fontsize=12)
    ax.set_title('Phase 2: Skill-Bias Scatter (Independent Normalization)\nAll Eras Visible', fontsize=14)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.clf()  # Clear buffer
    
    # Re-create and save fresh
    fig, ax = plt.subplots(figsize=(12, 8))
    for era in era_order:
        era_data = anomalies[anomalies['Era'] == era]
        if len(era_data) > 0:
            ax.scatter(era_data['Rational_Score_Z'], era_data['Normalized_Gap'],
                       alpha=0.6, s=80, label=f'{era} Era (n={len(era_data)})',
                       color=era_colors.get(era, '#95a5a6'), edgecolors='white', linewidth=0.5)
    
    try:
        coefs = np.polyfit(x, y, 2)
        poly = np.poly1d(coefs)
        x_smooth = np.linspace(x.min(), x.max(), 100)
        ax.plot(x_smooth, poly(x_smooth), 'k--', linewidth=2, alpha=0.7, label='Trend')
    except:
        pass
    
    ax.set_xlabel('Rational Score (Z-score)\n<-- Weaker | Stronger -->', fontsize=12)
    ax.set_ylabel('Normalized Survival Gap (0-1)', fontsize=12)
    ax.set_title('Phase 2: Skill-Bias Scatter (Independent Normalization)\nAll Eras Visible', fontsize=14)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = PLOTS_DIR / "skill_bias_scatter_v2_FIXED.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close('all')
    
    print(f"   Saved: {plot_path}")
    
    correlation = anomalies['Rational_Score_Z'].corr(anomalies['Normalized_Gap'])
    print(f"   Correlation: {correlation:.3f}")


# =============================================================================
# Phase 3: Distribution A/B/C Testing
# =============================================================================
def simulate_week_with_lambda(week_data: pd.DataFrame, 
                               chaos_weight: float,
                               prev_alpha: np.ndarray,
                               distribution: str = 'uniform') -> Tuple[np.ndarray, np.ndarray]:
    """Simulate a single week with given chaos weight and distribution."""
    n_contestants = len(week_data)
    rule_system = week_data['rule_system'].iloc[0]
    
    judge_scores = week_data['normalized_score'].values.astype(float)
    
    if len(judge_scores) > 1:
        mu = np.mean(judge_scores)
        sigma = np.std(judge_scores, ddof=1) if np.std(judge_scores) > 1e-6 else 1.0
        z_scores = (judge_scores - mu) / sigma
        skill_multipliers = 1.0 + SKILL_IMPACT_FACTOR * z_scores
        skill_multipliers = np.clip(skill_multipliers, 0.5, 2.0)
        alpha = prev_alpha * skill_multipliers if prev_alpha is not None else INITIAL_ALPHA * skill_multipliers
    else:
        alpha = prev_alpha if prev_alpha is not None else np.full(n_contestants, INITIAL_ALPHA)
    
    alpha = np.maximum(alpha, MIN_ALPHA)
    
    fan_shares = generate_mixed_shares(alpha, N_SIMULATIONS, chaos_weight, distribution)
    
    judge_shares = week_data['judge_share'].values
    judge_ranks = week_data['judge_rank'].values
    judge_scores_norm = week_data['normalized_score'].values
    
    if rule_system == 'Rank':
        eliminated_idx, _ = apply_rank_rule(judge_ranks, fan_shares)
    elif rule_system == 'Percent':
        eliminated_idx, _ = apply_percent_rule(judge_shares, fan_shares)
    else:
        eliminated_idx, _ = apply_rank_with_save_rule(judge_ranks, judge_scores_norm, fan_shares)
    
    valid_shares = fan_shares.mean(axis=0)
    new_alpha = alpha + LEARNING_RATE * valid_shares * EVIDENCE_BOOST
    new_alpha = np.maximum(new_alpha, MIN_ALPHA)
    
    return eliminated_idx, new_alpha


def compute_metrics_for_distribution(df: pd.DataFrame, chaos_weight: float, 
                                      distribution: str = 'uniform') -> Dict[str, float]:
    """
    Compute ALL metrics for a distribution:
    - Top-1 Recall: exact elimination prediction
    - Overall Accuracy: correct survival/elimination for all contestants
    - Explanation Rate: % of weeks where at least 1 simulation matches actual elimination
    - Certainty: average confidence (n_valid/N_SIMULATIONS) for explained weeks
    - Stability: average std dev of fan share estimates
    """
    np.random.seed(RANDOM_SEED)
    
    top1_correct = 0
    total_correct_predictions = 0
    total_contestants = 0
    total_eliminations = 0
    
    # New metrics
    explained_weeks = 0
    unexplained_weeks = 0
    certainty_sum = 0.0
    stability_values = []
    entropy_values = []  # Shannon Entropy
    
    seasons = sorted(df['season'].unique())
    
    for season in seasons:
        season_data = df[df['season'] == season].copy()
        weeks = sorted(season_data['week'].unique())
        
        prev_state: Optional[WeekState] = None  # Use WeekState instead of raw alpha
        
        for week in weeks:
            week_data = season_data[season_data['week'] == week].copy()
            week_data = week_data.sort_values('celebrity_name').reset_index(drop=True)
            
            n_contestants = len(week_data)
            contestant_names = week_data['celebrity_name'].tolist()
            
            if not week_data['is_eliminated'].any():
                # Non-elimination week: still update state
                if prev_state is None:
                    alpha = np.full(n_contestants, INITIAL_ALPHA)
                else:
                    alpha = align_prior_to_current(prev_state, contestant_names)
                prev_state = WeekState(alpha=alpha, contestant_names=contestant_names, n_contestants=n_contestants)
                continue
            
            actual_eliminated_mask = week_data['is_eliminated'].values
            actual_eliminated_idx = np.where(actual_eliminated_mask)[0][0]
            
            # ─────────────────────────────────────────────────────────────────
            # STEP 1: Construct Prior (EXACT same as monte_carlo_bayesian_dirichlet.py)
            # ─────────────────────────────────────────────────────────────────
            if prev_state is None:
                alpha = np.full(n_contestants, INITIAL_ALPHA)
            else:
                alpha = align_prior_to_current(prev_state, contestant_names)
            
            # ─────────────────────────────────────────────────────────────────
            # OPTIMIZATION: Incorporate Judge Skill into Prior
            # ─────────────────────────────────────────────────────────────────
            judge_scores = week_data['normalized_score'].values.astype(float)
            
            if len(judge_scores) > 1:
                mu = np.mean(judge_scores)
                sigma = np.std(judge_scores, ddof=1)
                if sigma > 1e-6:
                    z_scores = (judge_scores - mu) / sigma
                else:
                    z_scores = np.zeros_like(judge_scores)
                
                skill_multipliers = 1.0 + SKILL_IMPACT_FACTOR * z_scores
                skill_multipliers = np.clip(skill_multipliers, 0.5, 2.0)
                alpha = alpha * skill_multipliers
                alpha = np.maximum(alpha, MIN_ALPHA)
            
            # ─────────────────────────────────────────────────────────────────
            # STEP 2: Generate Samples with Chaos Mixture
            # ─────────────────────────────────────────────────────────────────
            fan_shares = generate_mixed_shares(alpha, N_SIMULATIONS, chaos_weight, distribution)
            
            # ─────────────────────────────────────────────────────────────────
            # STEP 3: Apply Elimination Rules
            # ─────────────────────────────────────────────────────────────────
            rule_system = week_data['rule_system'].iloc[0]
            judge_shares = week_data['judge_share'].values
            judge_ranks = week_data['judge_rank'].values
            judge_scores_norm = week_data['normalized_score'].values
            
            if rule_system == 'Rank':
                simulated_eliminated, _ = apply_rank_rule(judge_ranks, fan_shares)
            elif rule_system == 'Percent':
                simulated_eliminated, _ = apply_percent_rule(judge_shares, fan_shares)
            else:
                simulated_eliminated, _ = apply_rank_with_save_rule(judge_ranks, judge_scores_norm, fan_shares)
            
            prediction_counts = np.bincount(simulated_eliminated, minlength=n_contestants)
            predicted_eliminated_idx = np.argmax(prediction_counts)
            
            # Top-1 Recall
            if predicted_eliminated_idx == actual_eliminated_idx:
                top1_correct += 1
            
            # Overall Accuracy
            for i in range(n_contestants):
                actual_survived = not actual_eliminated_mask[i]
                predicted_survived = (i != predicted_eliminated_idx)
                if actual_survived == predicted_survived:
                    total_correct_predictions += 1
            
            # Explanation Rate: check if n_valid > 0
            n_valid = prediction_counts[actual_eliminated_idx]
            if n_valid > 0:
                explained_weeks += 1
                # Certainty: confidence for explained weeks
                certainty_sum += n_valid / N_SIMULATIONS
            else:
                unexplained_weeks += 1
            
            # Stability: compute share std across simulations
            share_estimates = prediction_counts / N_SIMULATIONS
            share_std = np.std(share_estimates)
            stability_values.append(share_std)
            
            # Shannon Entropy: H = -sum(p * log(p))
            probs = share_estimates[share_estimates > 0]
            entropy = -np.sum(probs * np.log(probs + 1e-10))
            entropy_values.append(entropy)
            
            total_contestants += n_contestants
            total_eliminations += 1
            
            # ─────────────────────────────────────────────────────────────────
            # STEP 4: Update Prior with Evidence (EXACT same as monte_carlo)
            # ─────────────────────────────────────────────────────────────────
            valid_shares = fan_shares.mean(axis=0)
            new_alpha = update_prior_with_evidence(alpha, valid_shares)
            
            # Create new state for survivors
            surviving_mask = ~actual_eliminated_mask
            if surviving_mask.sum() > 0:
                surviving_names = [contestant_names[i] for i in range(n_contestants) if surviving_mask[i]]
                surviving_alpha = new_alpha[surviving_mask]
                prev_state = WeekState(alpha=surviving_alpha, contestant_names=surviving_names, n_contestants=len(surviving_names))
            else:
                prev_state = None
    
    # Compute final metrics
    explanation_rate = explained_weeks / total_eliminations if total_eliminations > 0 else 0
    certainty = certainty_sum / explained_weeks if explained_weeks > 0 else 0
    stability = np.mean(stability_values) if stability_values else 0
    entropy_avg = np.mean(entropy_values) if entropy_values else 0
    
    return {
        'top1_recall': top1_correct / total_eliminations if total_eliminations > 0 else 0,
        'accuracy': total_correct_predictions / total_contestants if total_contestants > 0 else 0,
        'explanation_rate': explanation_rate,
        'certainty': certainty,
        'stability': stability,
        'entropy': entropy_avg  # Shannon Entropy
    }


def run_distribution_abc_test(df: pd.DataFrame,
                               lambda_range: Tuple[float, float] = (0.0, 0.15),
                               lambda_step: float = 0.001) -> pd.DataFrame:  # Changed to 0.001
    """
    Phase 3: A/B/C Test comparing Uniform, Pareto, Exponential distributions.
    """
    print("\n" + "="*70)
    print("   PHASE 3: Distribution A/B/C Test")
    print("="*70)
    
    distributions = ['uniform', 'pareto', 'exponential']
    lambda_values = np.arange(lambda_range[0], lambda_range[1] + lambda_step, lambda_step)
    
    all_results = []
    best_results = {}
    
    for dist in distributions:
        print(f"\n   Testing: {dist.upper()}")
        dist_results = []
        
        for lam in tqdm(lambda_values, desc=f"  {dist}"):
            metrics = compute_metrics_for_distribution(df, lam, dist)
            # Compute Information Ratio: IR = Explanation Rate / (1 - Certainty)
            ir = metrics['explanation_rate'] / (1 - metrics['certainty']) if metrics['certainty'] < 1 else float('inf')
            dist_results.append({
                'distribution': dist,
                'lambda': lam,
                'top1_recall': metrics['top1_recall'],
                'accuracy': metrics['accuracy'],
                'explanation_rate': metrics['explanation_rate'],
                'certainty': metrics['certainty'],
                'stability': metrics['stability'],
                'entropy': metrics['entropy'],  # Shannon Entropy
                'ir': ir  # Information Ratio
            })
        
        dist_df = pd.DataFrame(dist_results)
        all_results.append(dist_df)
        
        # Find best lambda for this distribution (based on INFORMATION RATIO)
        best_idx = dist_df['ir'].idxmax()
        best_lam = dist_df.loc[best_idx, 'lambda']
        best_ir = dist_df.loc[best_idx, 'ir']
        
        # Get ALL metrics at best lambda
        best_metrics = compute_metrics_for_distribution(df, best_lam, dist)
        ir_val = best_metrics['explanation_rate'] / (1 - best_metrics['certainty']) if best_metrics['certainty'] < 1 else float('inf')
        best_results[dist] = {
            'best_lambda': best_lam,
            'top1_recall': best_metrics['top1_recall'],
            'accuracy': best_metrics['accuracy'],
            'explanation_rate': best_metrics['explanation_rate'],
            'certainty': best_metrics['certainty'],
            'stability': best_metrics['stability'],
            'entropy': best_metrics['entropy'],  # Shannon Entropy
            'ir': ir_val  # Information Ratio
        }
    
    results_df = pd.concat(all_results, ignore_index=True)
    
    # Print FULL comparison table with ALL 7 metrics
    print("\n" + "="*120)
    print("   DISTRIBUTION COMPARISON TABLE (ALL METRICS @ BEST IR)")
    print("="*120)
    print(f"   {'Dist':<12} {'Lambda':<8} {'IR':<7} {'Expl.Rate':<10} {'Certainty':<10} {'Accuracy':<10} {'Stability':<10} {'Entropy':<8}")
    print("   " + "-"*95)
    
    for dist in distributions:
        r = best_results[dist]
        print(f"   {dist.upper():<12} {r['best_lambda']:<8.3f} {r['ir']:<7.3f} {r['explanation_rate']:<10.1%} "
              f"{r['certainty']:<10.1%} {r['accuracy']:<10.1%} {r['stability']:<10.4f} {r['entropy']:<8.4f}")
    
    # Auto-select winner based on INFORMATION RATIO
    print("\n   " + "-"*55)
    print("   AUTO-SELECTION (Maximum Information Ratio):")
    
    winner = None
    winner_ir = 0
    
    for dist in distributions:
        r = best_results[dist]
        if r['ir'] > winner_ir:
            winner = dist
            winner_ir = r['ir']
    
    print(f"\n   >>> WINNER: {winner.upper()} <<<")
    print(f"       Lambda: {best_results[winner]['best_lambda']:.3f}")
    print(f"       Information Ratio: {best_results[winner]['ir']:.3f}")
    print(f"       Explanation Rate: {best_results[winner]['explanation_rate']:.1%}")
    print(f"       Certainty: {best_results[winner]['certainty']:.1%}")
    print("="*70)
    
    # Extra: Output UNIFORM at Lambda=0.05 for comparison
    print("\n" + "="*70)
    print("   REFERENCE: UNIFORM @ Lambda=0.050")
    print("="*70)
    uniform_050 = compute_metrics_for_distribution(df, 0.05, 'uniform')
    ir_050 = uniform_050['explanation_rate'] / (1 - uniform_050['certainty']) if uniform_050['certainty'] < 1 else float('inf')
    print(f"   Lambda:          0.050")
    print(f"   Information Ratio: {ir_050:.3f}")
    print(f"   Explanation Rate: {uniform_050['explanation_rate']:.1%}")
    print(f"   Certainty:       {uniform_050['certainty']:.1%}")
    print(f"   Accuracy:        {uniform_050['accuracy']:.1%}")
    print(f"   Stability:       {uniform_050['stability']:.4f}")
    print(f"   Shannon Entropy: {uniform_050['entropy']:.4f}")
    print("="*70)
    
    # Plot comparison
    plot_distribution_comparison(results_df, best_results, winner)
    
    return results_df, best_results, winner


def plot_distribution_comparison(results_df: pd.DataFrame, 
                                  best_results: Dict, 
                                  winner: str) -> None:
    """Plot optimization curves for all distributions on one chart."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    colors = {'uniform': '#3498db', 'pareto': '#e74c3c', 'exponential': '#2ecc71'}
    
    # Left plot: Top-1 Recall
    ax1 = axes[0]
    for dist in ['uniform', 'pareto', 'exponential']:
        dist_data = results_df[results_df['distribution'] == dist]
        label = f'{dist.upper()}'
        if dist == winner:
            label += ' (WINNER)'
        ax1.plot(dist_data['lambda'], dist_data['top1_recall'], 
                 'o-', linewidth=2, markersize=4, color=colors[dist], label=label)
        
        # Mark best point
        best_lam = best_results[dist]['best_lambda']
        best_val = best_results[dist]['top1_recall']
        ax1.scatter([best_lam], [best_val], s=150, color=colors[dist], 
                    edgecolors='black', linewidth=2, zorder=5)
    
    ax1.set_xlabel('Chaos Weight (lambda)', fontsize=12)
    ax1.set_ylabel('Top-1 Recall (Strict)', fontsize=12)
    ax1.set_title('Top-1 Recall Comparison', fontsize=14)
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    
    # Right plot: Overall Accuracy
    ax2 = axes[1]
    ax2.axhline(y=0.88, color='gray', linestyle='--', linewidth=2, alpha=0.7, label='88% Threshold')
    
    for dist in ['uniform', 'pareto', 'exponential']:
        dist_data = results_df[results_df['distribution'] == dist]
        label = f'{dist.upper()}'
        if dist == winner:
            label += ' (WINNER)'
        ax2.plot(dist_data['lambda'], dist_data['accuracy'], 
                 'o-', linewidth=2, markersize=4, color=colors[dist], label=label)
        
        best_lam = best_results[dist]['best_lambda']
        best_val = best_results[dist]['accuracy']
        ax2.scatter([best_lam], [best_val], s=150, color=colors[dist], 
                    edgecolors='black', linewidth=2, zorder=5)
    
    ax2.set_xlabel('Chaos Weight (lambda)', fontsize=12)
    ax2.set_ylabel('Overall Accuracy', fontsize=12)
    ax2.set_title('Overall Accuracy Comparison', fontsize=14)
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    ax2.set_ylim(0.80, 1.0)
    
    plt.suptitle(f'Distribution A/B/C Test - Winner: {winner.upper()}', fontsize=16, y=1.02)
    plt.tight_layout()
    
    plot_path = PLOTS_DIR / "distribution_comparison.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close('all')
    
    print(f"   Saved: {plot_path}")


def plot_explanation_certainty_curves(results_df: pd.DataFrame, 
                                       best_results: Dict, 
                                       winner: str) -> None:
    """
    Plot two separate charts:
    1. Explanation Rate vs Lambda (for all 3 distributions)
    2. Certainty vs Lambda (for all 3 distributions)
    """
    colors = {'uniform': '#3498db', 'pareto': '#e74c3c', 'exponential': '#2ecc71'}
    
    # ==========================================================================
    # Plot 1: Explanation Rate vs Lambda
    # ==========================================================================
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    
    for dist in ['uniform', 'pareto', 'exponential']:
        dist_data = results_df[results_df['distribution'] == dist]
        label = f'{dist.upper()}'
        if dist == winner:
            label += ' (WINNER)'
        ax1.plot(dist_data['lambda'], dist_data['explanation_rate'], 
                 'o-', linewidth=2, markersize=5, color=colors[dist], label=label)
        
        # Mark best point
        best_lam = best_results[dist]['best_lambda']
        best_val = best_results[dist]['explanation_rate']
        ax1.scatter([best_lam], [best_val], s=150, color=colors[dist], 
                    edgecolors='black', linewidth=2, zorder=5)
    
    ax1.axhline(y=0.95, color='gray', linestyle='--', linewidth=2, alpha=0.7, label='95% Threshold')
    ax1.set_xlabel('Chaos Weight (Lambda)', fontsize=12)
    ax1.set_ylabel('Explanation Rate', fontsize=12)
    ax1.set_title('Explanation Rate vs Lambda\n(Higher = Better, Goal: >95%)', fontsize=14)
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    ax1.set_ylim(0.85, 1.02)
    
    plt.tight_layout()
    plot_path1 = PLOTS_DIR / "explanation_rate_vs_lambda.png"
    plt.savefig(plot_path1, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   Saved: {plot_path1}")
    
    # ==========================================================================
    # Plot 2: Certainty vs Lambda
    # ==========================================================================
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    
    for dist in ['uniform', 'pareto', 'exponential']:
        dist_data = results_df[results_df['distribution'] == dist]
        label = f'{dist.upper()}'
        if dist == winner:
            label += ' (WINNER)'
        ax2.plot(dist_data['lambda'], dist_data['certainty'], 
                 'o-', linewidth=2, markersize=5, color=colors[dist], label=label)
        
        # Mark best point
        best_lam = best_results[dist]['best_lambda']
        best_val = best_results[dist]['certainty']
        ax2.scatter([best_lam], [best_val], s=150, color=colors[dist], 
                    edgecolors='black', linewidth=2, zorder=5)
    
    ax2.set_xlabel('Chaos Weight (Lambda)', fontsize=12)
    ax2.set_ylabel('Certainty (Confidence)', fontsize=12)
    ax2.set_title('Certainty vs Lambda\n(Higher = More Confident Predictions)', fontsize=14)
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    ax2.set_ylim(0.20, 0.50)
    
    plt.tight_layout()
    plot_path2 = PLOTS_DIR / "certainty_vs_lambda.png"
    plt.savefig(plot_path2, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   Saved: {plot_path2}")


def plot_information_ratio(results_df: pd.DataFrame, 
                            best_results: Dict, 
                            winner: str) -> None:
    """
    Plot Information Ratio (IR) vs Lambda for all 3 distributions.
    IR = Explanation Rate / (1 - Certainty)
    Higher IR = better balance of coverage and confidence.
    """
    colors = {'uniform': '#3498db', 'pareto': '#e74c3c', 'exponential': '#2ecc71'}
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Find global maximum IR point
    global_best_ir = 0
    global_best_dist = None
    global_best_lam = None
    
    for dist in ['uniform', 'pareto', 'exponential']:
        dist_data = results_df[results_df['distribution'] == dist]
        label = f'{dist.upper()}'
        if dist == winner:
            label += ' (WINNER)'
        
        ax.plot(dist_data['lambda'], dist_data['ir'], 
                'o-', linewidth=2.5, markersize=6, color=colors[dist], label=label)
        
        # Mark best point for this distribution
        best_lam = best_results[dist]['best_lambda']
        best_ir = best_results[dist]['ir']
        ax.scatter([best_lam], [best_ir], s=200, color=colors[dist], 
                   edgecolors='black', linewidth=2.5, zorder=5)
        
        # Track global best
        if best_ir > global_best_ir:
            global_best_ir = best_ir
            global_best_dist = dist
            global_best_lam = best_lam
    
    # Annotate global maximum
    ax.annotate(f'MAX: {global_best_dist.upper()}\nλ={global_best_lam:.3f}\nIR={global_best_ir:.3f}',
                xy=(global_best_lam, global_best_ir),
                xytext=(global_best_lam + 0.02, global_best_ir + 0.03),
                fontsize=11, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5),
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
    
    ax.set_xlabel('Chaos Weight (Lambda)', fontsize=13)
    ax.set_ylabel('Information Ratio (IR)', fontsize=13)
    ax.set_title('Information Ratio vs Lambda\nIR = Explanation Rate / (1 - Certainty)\nHigher = Better Balance of Coverage and Confidence', 
                 fontsize=14)
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = PLOTS_DIR / "information_ratio_vs_lambda.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   Saved: {plot_path}")


# =============================================================================
# Main Pipeline
# =============================================================================
def main():
    print("="*70)
    print("   MCM 2026: Bayesian Diagnostic + Distribution A/B/C Test")
    print("="*70)
    
    print(f"\n[INFO] Loading data from: {INPUT_PATH}")
    df = pd.read_csv(INPUT_PATH)
    print(f"   Rows: {len(df)}, Seasons: {df['season'].nunique()}")
    
    # Phase 1
    df_diagnostics = run_diagnostics(df)
    
    # Phase 2 (FIXED)
    plot_skill_bias_scatter(df_diagnostics)
    
    # Phase 3: A/B/C Test
    results_df, best_results, winner = run_distribution_abc_test(df)
    
    # Phase 4: Plot Explanation Rate and Certainty curves
    print("\n[INFO] Generating diagnostic plots...")
    plot_explanation_certainty_curves(results_df, best_results, winner)
    
    # Phase 5: Plot Information Ratio curve
    plot_information_ratio(results_df, best_results, winner)
    
    # Final Summary
    print("\n" + "="*70)
    print("   FINAL RECOMMENDATION (Based on Information Ratio)")
    print("="*70)
    print(f"\n   Distribution: {winner.upper()}")
    print(f"   Lambda: {best_results[winner]['best_lambda']:.3f}")
    print(f"   Information Ratio: {best_results[winner]['ir']:.3f}")
    print(f"\n   Update monte_carlo_bayesian_dirichlet.py:")
    print(f"      CHAOS_FACTOR = {best_results[winner]['best_lambda']:.3f}")
    print(f"      CHAOS_DISTRIBUTION = '{winner}'")
    print("="*70)
    
    return winner, best_results


if __name__ == "__main__":
    main()


