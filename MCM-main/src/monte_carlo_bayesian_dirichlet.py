#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MCM 2026 Problem C: Data With The Stars
Bayesian-Dirichlet Monte Carlo Simulation
- Bayesian Updating with Industry-Based Asymmetric Dirichlet Prior

══════════════════════════════════════════════════════════════════════════════
Bayesian-Dirichlet Optimization: Industry-Informed Two-Class Dirichlet Prior
══════════════════════════════════════════════════════════════════════════════

核心问题：
---------
原始模型使用对称 Dirichlet 先验 Dir(α, α, ..., α)，假设所有选手的"粉丝爆发概率"
相同。然而，现实中不同行业的选手具有截然不同的粉丝基础：

- Sean Spicer (S28, Politician): 评委分数垫底，但政治粉丝投票极高，存活多周
- Reality Stars: 自带社交媒体粉丝群体，投票动员能力强
- 传统 Actor/Athlete: 粉丝基础相对普通

这种"起跑线不同"的现象导致对称先验无法解释某些"异常"淘汰/存活案例。

方案 A：两类先验策略
-------------------
将选手分为两类：

1. **"自带粉丝群体"类 (Built-in Fanbase)**:
   - Reality Star / TV Personality
   - Social Media Personality  
   - Politician
   - Radio Personality
   → 使用较高的 α_high = 1.2（更可能成为"大头"）

2. **"普通选手"类 (Regular Contestants)**:
   - Actor/Actress, Athlete, Singer, Model, etc.
   → 使用标准的 α_low = 0.8（稀疏解倾向）

数学直觉：
---------
- α < 1: 分布呈"U型"，倾向于极端值（少数人拿大头）
- α = 1: 均匀分布
- α > 1: 分布呈"倒U型"，倾向于集中（大家份额接近）

通过给"自带粉丝"选手更高的 α，使其在先验分布中更可能获得较高份额。

Author: MCM Team
Date: 2026-01-30
"""

import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import Tuple, Optional, Dict, List
from dataclasses import dataclass
import warnings
from scipy.stats import entropy

warnings.filterwarnings('ignore')

# =============================================================================
# Configuration
# =============================================================================
BASE_DIR = Path(__file__).parent.parent
RAW_DATA_PATH = BASE_DIR / "data_raw" / "2026_MCM_Problem_C_Data.csv"
INPUT_PATH = BASE_DIR / "data_processed" / "dwts_simulation_input.csv"
OUTPUT_PATH = BASE_DIR / "results" / "full_simulation_bayesian.csv"

# Simulation parameters
N_SIMULATIONS = 10000
RANDOM_SEED = 42

# ┌─────────────────────────────────────────────────────────────────────────────┐
# │           方案 A: 混合贝叶斯模型 (Mixture Bayesian Model)                     │
# │           Pure Skill + Momentum + Chaos (Pareto/Uniform Tail)               │
# └─────────────────────────────────────────────────────────────────────────────┘

# 移除行业先验，避免循环论证。
# 初始 α 对所有选手一视同仁。
INITIAL_ALPHA = 1.0    # 初始无信息先验 (Uniform / Slight Sparsity)

# 时间动态参数
LEARNING_RATE = 0.4
EVIDENCE_BOOST = 5.0
MIN_ALPHA = 0.1
SKILL_IMPACT_FACTOR = 0.3  # 增加实力权重

# 混合模型参数 (Mixture Model)
# 允许一定比例的样本来自"混乱分布" (Uniform)，涵盖非理性投票/长尾事件。
CHAOS_FACTOR = 0.05  # 5% 的可能性是完全随机投票 (The "Chaos" Component)


# =============================================================================
# Data Classes
# =============================================================================
@dataclass
class WeekState:
    """State information passed between weeks"""
    alpha: np.ndarray
    contestant_names: List[str]
    # contestant_industries removed
    
    @property
    def alpha_sum(self) -> float:
        return self.alpha.sum()
    
    @property
    def n_contestants(self) -> int:
        return len(self.contestant_names)


# =============================================================================
# Core Bayesian Functions
# =============================================================================
def initialize_priors(contestant_names: List[str]) -> np.ndarray:
    """
    Initialize Dirichlet prior (Uniform).
    
    Args:
        contestant_names: List of contestant names
        
    Returns:
        alpha_array
    """
    n = len(contestant_names)
    return np.full(n, INITIAL_ALPHA)


def align_prior_to_current(
    prev_state: WeekState,
    current_names: List[str]
) -> np.ndarray:
    """
    Align previous week's posterior to current week's contestants.
    
    Args:
        prev_state: Previous week's state
        current_names: Current week's contestant names
        
    Returns:
        aligned_alpha
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
    # Since alpha sum roughly correlates to information quantity (pseudo-counts),
    # removing a loser reduces total information. We might want to slightly
    # scale up survivors to reflect "reallocated attention", but keeping
    # alpha stable (inertia) is the primary goal.
    # Here we don't aggressively rescale because evidence accumulates over time.
    # However, to avoid alpha vanishing or exploding irrelevant to N, we can check mean.
    
    # Optional: decay old information slightly? 
    # Current logic: alpha grows with evidence. 
    # Let's keep the simple logic from before: scale if drop is significant?
    # Previous logic was: scale_factor = min(1.0 / remaining_ratio, 1.3)
    # This keeps total alpha sum roughly constant.
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
    
    Update rule: α_new = α_old + η · evidence_scaled
    """
    evidence_scaled = evidence * EVIDENCE_BOOST
    new_alpha = current_alpha + learning_rate * evidence_scaled
    new_alpha = np.maximum(new_alpha, MIN_ALPHA)
    return new_alpha


# =============================================================================
# Simulation Functions
# =============================================================================
def generate_fan_shares(alpha: np.ndarray, n_sims: int = N_SIMULATIONS) -> np.ndarray:
    """
    Generate fan share samples using a Mixture Model.
    
    Model: (1 - lambda) * Dirichlet(alpha_skill) + lambda * Dirichlet(alpha_uniform)
    """
    n_chaos = int(n_sims * CHAOS_FACTOR)
    n_skill = n_sims - n_chaos
    
    # 1. Rational/Skill Component (History + Judge Scores)
    samples_skill = np.random.dirichlet(alpha, size=n_skill)
    
    # 2. Chaos/Irrational Component (Uniform/Random)
    # Represents viral moments, protest votes, or complete noise.
    alpha_chaos = np.ones_like(alpha) # Uniform prior
    samples_chaos = np.random.dirichlet(alpha_chaos, size=n_chaos)
    
    # Combine
    return np.concatenate([samples_skill, samples_chaos], axis=0)


def shares_to_ranks(shares: np.ndarray) -> np.ndarray:
    """Convert shares to ranks (1 = highest)."""
    order = np.argsort(-shares, axis=1)
    ranks = np.argsort(order, axis=1) + 1
    return ranks.astype(float)


def apply_rank_rule(judge_ranks: np.ndarray, fan_shares: np.ndarray) -> np.ndarray:
    """RANK elimination rule (Seasons 1-2)."""
    fan_ranks = shares_to_ranks(fan_shares)
    total_scores = judge_ranks + fan_ranks
    return np.argmax(total_scores + fan_ranks * 0.001, axis=1)


def apply_percent_rule(judge_shares: np.ndarray, fan_shares: np.ndarray) -> np.ndarray:
    """PERCENT elimination rule (Seasons 3-27)."""
    return np.argmin(judge_shares + fan_shares, axis=1)


def apply_rank_with_save_rule(
    judge_ranks: np.ndarray,
    judge_scores: np.ndarray,
    fan_shares: np.ndarray
) -> np.ndarray:
    """RANK_WITH_SAVE elimination rule (Seasons 28+)."""
    n_sims, n_contestants = fan_shares.shape
    
    if n_contestants < 2:
        return np.zeros(n_sims, dtype=int)
    
    fan_ranks = shares_to_ranks(fan_shares)
    total_scores = judge_ranks + fan_ranks + fan_ranks * 1e-6
    
    if n_contestants == 2:
        if judge_scores[0] < judge_scores[1]:
            return np.zeros(n_sims, dtype=int)
        elif judge_scores[1] < judge_scores[0]:
            return np.ones(n_sims, dtype=int)
        else:
            return np.argmax(fan_ranks, axis=1)
    
    sorted_indices = np.argsort(-total_scores, axis=1)
    bottom2_first = sorted_indices[:, 0]
    bottom2_second = sorted_indices[:, 1]
    
    judge_scores_first = judge_scores[bottom2_first]
    judge_scores_second = judge_scores[bottom2_second]
    
    return np.where(
        judge_scores_first < judge_scores_second,
        bottom2_first,
        np.where(judge_scores_second < judge_scores_first, bottom2_second, bottom2_first)
    )


def simulate_week_with_priors(
    week_data: pd.DataFrame,
    prev_state: Optional[WeekState],
    rule_system: str
) -> Tuple[pd.DataFrame, Optional[WeekState]]:
    """
    Simulate single week with Bayesian Dirichlet prior (Skill + Momentum only).
    """
    contestant_names = week_data['celebrity_name'].tolist()
    n_contestants = len(contestant_names)
    
    # ─────────────────────────────────────────────────────────────────────────
    # STEP 1: Construct Prior
    # ─────────────────────────────────────────────────────────────────────────
    if prev_state is None:
        # First week: use uniform prior
        alpha = initialize_priors(contestant_names)
    else:
        # Subsequent weeks: inherit + align
        alpha = align_prior_to_current(prev_state, contestant_names)
    
    # ─────────────────────────────────────────────────────────────────────────
    # OPTIMIZATION: Incorporate Judge Skill (Performance) into Prior
    # ─────────────────────────────────────────────────────────────────────────
    # Adjust prior based on this week's judge performance (Z-score approach)
    # Theory: Better dancers (higher judge scores) naturally attract more bandwagon fans
    
    # Get normalized scores (0.0 to 1.0) for current contestants
    current_judge_scores = week_data['normalized_score'].values.astype(float)
    
    if len(current_judge_scores) > 1:
        # Calculate Z-score
        mu = np.mean(current_judge_scores)
        sigma = np.std(current_judge_scores, ddof=1) # Sample std dev
        
        if sigma > 1e-6:
            z_scores = (current_judge_scores - mu) / sigma
        else:
            z_scores = np.zeros_like(current_judge_scores)
            
        # Apply adjustment: alpha_new = alpha * (1 + lambda * z_score)
        # We clamp the multiplier to [0.5, 2.0] to prevent extreme distortions
        # If z=2 (very good), factor = 1 + 0.3*2 = 1.6
        # If z=-2 (very bad), factor = 1 + 0.3*(-2) = 0.4
        skill_multipliers = 1.0 + SKILL_IMPACT_FACTOR * z_scores
        skill_multipliers = np.clip(skill_multipliers, 0.5, 2.0)
        
        alpha = alpha * skill_multipliers
        # Ensure alpha doesn't drop below minimum
        alpha = np.maximum(alpha, MIN_ALPHA)

    prior_strength = alpha.sum()
    
    # ─────────────────────────────────────────────────────────────────────────
    # STEP 2: Generate Samples
    # ─────────────────────────────────────────────────────────────────────────
    fan_share_samples = generate_fan_shares(alpha, N_SIMULATIONS)
    
    # ─────────────────────────────────────────────────────────────────────────
    # STEP 3: Apply Elimination Rules
    # ─────────────────────────────────────────────────────────────────────────
    judge_shares = week_data['judge_share'].values
    judge_ranks = week_data['judge_rank'].values
    judge_scores = week_data['normalized_score'].values
    
    if rule_system == 'Rank':
        simulated_eliminated = apply_rank_rule(
            np.tile(judge_ranks, (N_SIMULATIONS, 1)), fan_share_samples
        )
    elif rule_system == 'Percent':
        simulated_eliminated = apply_percent_rule(
            np.tile(judge_shares, (N_SIMULATIONS, 1)), fan_share_samples
        )
    else:  # Rank_With_Save
        simulated_eliminated = apply_rank_with_save_rule(
            np.tile(judge_ranks, (N_SIMULATIONS, 1)), judge_scores, fan_share_samples
        )
    
    # ─────────────────────────────────────────────────────────────────────────
    # STEP 4: Filter Valid Simulations
    # ─────────────────────────────────────────────────────────────────────────
    actual_eliminated_mask = week_data['is_eliminated'].values
    
    if actual_eliminated_mask.sum() == 0:
        # Non-elimination week (e.g. withdrawal)
        new_state = WeekState(alpha=alpha, contestant_names=contestant_names)
        results = week_data.copy()
        results['estimated_fan_share'] = alpha / alpha.sum()
        results['share_std'] = np.nan
        results['confidence'] = np.nan
        results['n_valid_sims'] = np.nan  # Mark as skipped for reporting consistency
        results['prior_strength'] = prior_strength
        results['posterior_strength'] = prior_strength
        return results, new_state
    
    actual_eliminated_idx = np.where(actual_eliminated_mask)[0][0]
    valid_mask = simulated_eliminated == actual_eliminated_idx
    n_valid = valid_mask.sum()
    
    # ─────────────────────────────────────────────────────────────────────────
    # STEP 5: Estimate Fan Shares
    # ─────────────────────────────────────────────────────────────────────────
    if n_valid > 0:
        valid_samples = fan_share_samples[valid_mask]
        estimated_shares = valid_samples.mean(axis=0)
        share_stds = valid_samples.std(axis=0)
        
        # Calculate Shannon Entropy for each contestant (Uncertainty Metric)
        # Higher entropy = Model is less sure about the specific vote share
        share_entropies = []
        for i in range(n_contestants):
             # Discretize into bins to calculate entropy of the distribution
             counts, _ = np.histogram(valid_samples[:, i], bins=50, range=(0, 1))
             # Normalizing to probabilities is handled by scipy.stats.entropy if standard input
             probs = counts / counts.sum()
             ent = entropy(probs) # Base e (nats)
             share_entropies.append(ent)
        share_entropies = np.array(share_entropies)
        
        confidence = n_valid / N_SIMULATIONS
        new_alpha = update_prior_with_evidence(alpha, estimated_shares)
    else:
        # Fallback to prior
        estimated_shares = alpha / alpha.sum()
        share_stds = np.full(n_contestants, np.nan)
        share_entropies = np.full(n_contestants, np.nan)
        confidence = 0.0
        new_alpha = alpha
    
    posterior_strength = new_alpha.sum()
    
    # ─────────────────────────────────────────────────────────────────────────
    # STEP 6: Build Results
    # ─────────────────────────────────────────────────────────────────────────
    results = week_data.copy()
    results['estimated_fan_share'] = estimated_shares
    results['share_std'] = share_stds
    results['share_entropy'] = share_entropies
    results['confidence'] = confidence
    results['n_valid_sims'] = n_valid
    results['prior_strength'] = prior_strength
    results['posterior_strength'] = posterior_strength
    
    new_state = WeekState(
        alpha=new_alpha,
        contestant_names=contestant_names
    )
    
    return results, new_state


def run_simulation_bayesian(df: pd.DataFrame) -> pd.DataFrame:
    """Run full simulation with Bayesian prior (Skill-only)."""
    np.random.seed(RANDOM_SEED)
    
    # 移除行业信息加载
    # print("[INFO] Loading industry mapping...")
    # industry_map = load_industry_mapping()
    
    all_results = []
    seasons = df['season'].unique()
    
    for season in tqdm(seasons, desc="Processing seasons"):
        season_data = df[df['season'] == season].copy()
        
        # Check rule system (handle case where rule_system col might be missing)
        if 'rule_system' in season_data.columns:
            rule_system = season_data['rule_system'].iloc[0]
        else:
            # Fallback based on season number if needed
            if season <= 2: rule_system = 'Rank'
            elif season <= 27: rule_system = 'Percent'
            else: rule_system = 'Rank_With_Save'
            
        # Ensure week_num or week is used
        week_col = 'week_num' if 'week_num' in season_data.columns else 'week'
        weeks = sorted(season_data[week_col].unique())
        
        prev_state = None
        
        for week in weeks:
            week_data = season_data[season_data[week_col] == week].copy()
            week_data = week_data.sort_values('celebrity_name').reset_index(drop=True)
            
            results, prev_state = simulate_week_with_priors(
                week_data, prev_state, rule_system
            )
            
            # Ensure season/week columns persist
            results['season'] = season
            results['week'] = week
            all_results.append(results)
    
    return pd.concat(all_results, ignore_index=True)


# =============================================================================
# Pressure Test: Sean Spicer Case
# =============================================================================
def pressure_test_sean_spicer(df: pd.DataFrame, results: pd.DataFrame):
    """
    Pressure test: Check if the model can now explain Sean Spicer's survival.
    
    Sean Spicer (S28) is the canonical "unexplainable" case:
    - Lowest judge scores
    - Survived many weeks due to political fanbase
    """
    print("\n" + "="*70)
    print("   PRESSURE TEST: Sean Spicer (S28)")
    print("="*70)
    
    # Find Sean Spicer's data
    spicer_results = results[results['celebrity_name'].str.contains('Spicer', case=False, na=False)]
    
    if len(spicer_results) == 0:
        print("   [WARNING] Sean Spicer not found in results")
        return
    
    print(f"\n   Sean Spicer's simulation results:")
    print("-" * 70)
    
    for _, row in spicer_results.iterrows():
        conf_val = row['confidence'] if pd.notna(row['confidence']) else 0
        n_val = int(row['n_valid_sims']) if pd.notna(row['n_valid_sims']) else 0
        print(f"   Week {row['week']:2d}: n_valid={n_val:5d}, "
              f"conf={conf_val:.4f}, "
              f"fan_share={row['estimated_fan_share']:.4f}")
    
    # Summary
    valid_weeks = spicer_results[spicer_results['n_valid_sims'] > 0]
    total_weeks = len(spicer_results)
    explainable_weeks = len(valid_weeks)
    
    print(f"\n   Summary: {explainable_weeks}/{total_weeks} weeks explained (n_valid > 0)")
    
    if explainable_weeks > 0:
        print(f"   [OK] Skill+Momentum model explains some Spicer survivals")
    else:
        print(f"   [!!] Still cannot explain Spicer's survival (Needs Industry logic?)")


# =============================================================================
# Main
# =============================================================================
def main():
    print("="*70)
    print("   MCM 2026 Problem C: Bayesian-Dirichlet Monte Carlo")
    print("   Mixture Bayesian Model: Skill + Momentum + Chaos")
    print("="*70)
    
    print(f"\n[CONFIG] Hyperparameters:")
    print(f"   initial_alpha (uniform)       = {INITIAL_ALPHA}")
    print(f"   eta (LEARNING_RATE)           = {LEARNING_RATE}")
    print(f"   skill_impact (lambda)         = {SKILL_IMPACT_FACTOR}")
    print(f"   chaos_factor (mixture)        = {CHAOS_FACTOR}")
    print(f"   N_SIMULATIONS                 = {N_SIMULATIONS}")
    
    # Load data
    print(f"\n[INFO] Loading data from: {INPUT_PATH}")
    df = pd.read_csv(INPUT_PATH)
    print(f"   Rows: {len(df)}, Seasons: {df['season'].nunique()}")
    
    # Run simulation
    print(f"\n[INFO] Running simulation with Skill-Based prior...")
    results = run_simulation_bayesian(df)
    
    # Results already contain all original columns
    final_results = results.copy()
    
    # Reorder columns: original columns first, then estimates
    # (industry_class removed, added share_entropy)
    estimate_cols = ['estimated_fan_share', 'share_std', 'share_entropy', 'confidence', 'n_valid_sims',
                     'prior_strength', 'posterior_strength']
    other_cols = [c for c in final_results.columns if c not in estimate_cols]
    final_results = final_results[other_cols + estimate_cols]
    
    # Save
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    final_results.to_csv(OUTPUT_PATH, index=False)
    print(f"\n[INFO] Full simulation results saved to: {OUTPUT_PATH}")
    print(f"   Rows: {len(final_results)}, Columns: {len(final_results.columns)}")
    
    # Summary
    print("\n" + "="*70)
    print("   RESULTS SUMMARY")
    print("="*70)
    
    valid_results = final_results[final_results['n_valid_sims'] > 0]
    
    print(f"\n   Total contestant-weeks: {len(final_results)}")
    print(f"   Valid simulations: {len(valid_results)} ({100*len(valid_results)/len(final_results):.1f}%)")
    
    print(f"\n   Prior Strength Evolution (alpha_sum):")
    strength_by_week = final_results.groupby('week')['prior_strength'].mean()
    for week in [1, 3, 5, 7, 9]:
        if week in strength_by_week.index:
            print(f"      Week {week}: {strength_by_week[week]:.2f}")
    
    # Pressure test
    # (Optional: Spicer might be harder to explain without industry prior, 
    #  but if he had momentum or other factors, it might show)
    pressure_test_sean_spicer(df, final_results)
    
    print("\n" + "="*70)
    print("   Simulation Complete!")
    print("="*70)
    
    return final_results


if __name__ == "__main__":
    main()
