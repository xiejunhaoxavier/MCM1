#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MCM 2026 Problem C: Data With The Stars
Monte Carlo Simulation - Fan Vote Share Estimation (Inverse Solver)

This script uses a Generate-Filter-Aggregate approach to estimate
the unknown fan vote shares that are consistent with historical
elimination outcomes.

Mathematical Logic:
- Generate random fan vote distributions using Dirichlet(alpha=0.8)
- Apply the appropriate elimination rule for each era
- Filter simulations that match actual elimination outcomes
- Aggregate valid simulations to estimate fan shares

Author: MCM Team
Date: 2026-01-30
"""

import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import Tuple, Optional
import warnings

warnings.filterwarnings('ignore')

# =============================================================================
# Configuration
# =============================================================================
INPUT_PATH = Path(__file__).parent.parent / "data_processed" / "dwts_simulation_input.csv"
OUTPUT_PATH = Path(__file__).parent.parent / "results" / "full_simulation_basic.csv"

# Simulation parameters
N_SIMULATIONS = 10000
DIRICHLET_ALPHA = 0.8  # alpha < 1 simulates "Star Power" (Zipfian distribution)

# Random seed for reproducibility
RANDOM_SEED = 42


# =============================================================================
# Core Simulation Functions
# =============================================================================
def generate_fan_shares(n_sims: int, n_contestants: int, alpha: float = 0.8) -> np.ndarray:
    """
    Generate random fan vote shares using Dirichlet distribution.
    
    The Dirichlet distribution with alpha < 1 creates a "rich get richer"
    effect where few contestants get most of the votes (realistic Star Power).
    
    Args:
        n_sims: Number of simulation trials
        n_contestants: Number of active contestants
        alpha: Dirichlet concentration parameter (< 1 for Star Power)
        
    Returns:
        Array of shape (n_sims, n_contestants) where each row sums to 1.0
    """
    # Create alpha vector (same concentration for all contestants)
    alpha_vector = np.full(n_contestants, alpha)
    
    # Generate Dirichlet samples
    # Shape: (n_sims, n_contestants)
    fan_shares = np.random.dirichlet(alpha_vector, size=n_sims)
    
    return fan_shares


def shares_to_ranks(shares: np.ndarray) -> np.ndarray:
    """
    Convert share values to ranks (1 = highest share = best).
    Uses average method for ties.
    
    Vectorized implementation for efficiency.
    
    Args:
        shares: Array of shape (n_sims, n_contestants)
        
    Returns:
        Ranks array of same shape (1 = best)
    """
    n_sims, n_contestants = shares.shape
    
    # argsort twice gives ranks (0-indexed)
    # Negate to get descending order (highest share = rank 0)
    order = np.argsort(-shares, axis=1)
    ranks = np.argsort(order, axis=1) + 1  # Convert to 1-indexed
    
    # Note: This simple approach doesn't handle ties with average method
    # For Monte Carlo with continuous Dirichlet samples, ties are extremely rare
    # so we use this faster approach
    
    return ranks.astype(float)


def apply_rank_rule(judge_ranks: np.ndarray, fan_shares: np.ndarray) -> np.ndarray:
    """
    Apply RANK elimination rule (Seasons 1-2).
    
    Total_Score = Judge_Rank + Fan_Rank
    Elimination: Contestant with HIGHEST Total Score (worst combined rank)
    Tie-Breaker: Worse Fan Rank (lower votes) is eliminated
    
    Args:
        judge_ranks: Array of shape (n_contestants,) with judge rankings
        fan_shares: Array of shape (n_sims, n_contestants) with fan shares
        
    Returns:
        Array of shape (n_sims,) with index of eliminated contestant per simulation
    """
    n_sims, n_contestants = fan_shares.shape
    
    # Convert fan shares to ranks
    fan_ranks = shares_to_ranks(fan_shares)
    
    # Total score = Judge Rank + Fan Rank (lower is better)
    # Broadcast judge_ranks across all simulations
    total_scores = judge_ranks + fan_ranks  # Shape: (n_sims, n_contestants)
    
    # Find contestant(s) with highest total score (worst rank sum)
    max_scores = np.max(total_scores, axis=1, keepdims=True)
    is_worst = (total_scores == max_scores)  # Boolean mask for worst performers
    
    # Tie-breaker: Among tied contestants, eliminate the one with worse fan rank
    # (higher fan rank number = worse = lower votes)
    # Create a composite score: total_score + small tie-breaker
    tie_breaker = fan_ranks * 0.001  # Small value to break ties
    composite_score = total_scores + tie_breaker
    
    # Eliminated contestant is the one with highest composite score
    eliminated_idx = np.argmax(composite_score, axis=1)
    
    return eliminated_idx


def apply_percent_rule(judge_shares: np.ndarray, fan_shares: np.ndarray) -> np.ndarray:
    """
    Apply PERCENT elimination rule (Seasons 3-27).
    
    Total_Score = Judge_Share + Fan_Share
    Elimination: Contestant with LOWEST Total Score
    
    Args:
        judge_shares: Array of shape (n_contestants,) with judge score shares
        fan_shares: Array of shape (n_sims, n_contestants) with fan shares
        
    Returns:
        Array of shape (n_sims,) with index of eliminated contestant per simulation
    """
    # Total score = Judge Share + Fan Share (higher is better)
    # Broadcast judge_shares across all simulations
    total_scores = judge_shares + fan_shares  # Shape: (n_sims, n_contestants)
    
    # Eliminated contestant is the one with lowest total score
    eliminated_idx = np.argmin(total_scores, axis=1)
    
    return eliminated_idx


def apply_rank_with_save_rule(judge_ranks: np.ndarray, judge_scores: np.ndarray, 
                               fan_shares: np.ndarray) -> np.ndarray:
    """
    Apply RANK_WITH_SAVE elimination rule (Seasons 28+).
    
    1. Calculate Total_Score = Judge_Rank + Fan_Rank (same as Rank rule)
    2. Identify Bottom 2 contestants (highest total scores)
    3. Judges' Save: Between Bottom 2, the one with HIGHER judge score is SAVED
    
    Args:
        judge_ranks: Array of shape (n_contestants,) with judge rankings
        judge_scores: Array of shape (n_contestants,) with normalized judge scores
        fan_shares: Array of shape (n_sims, n_contestants) with fan shares
        
    Returns:
        Array of shape (n_sims,) with index of eliminated contestant per simulation
    """
    n_sims, n_contestants = fan_shares.shape
    
    if n_contestants < 2:
        # Can't have bottom 2 with less than 2 contestants
        return np.zeros(n_sims, dtype=int)
    
    # Convert fan shares to ranks
    fan_ranks = shares_to_ranks(fan_shares)
    
    # Total score = Judge Rank + Fan Rank
    total_scores = judge_ranks + fan_ranks  # Shape: (n_sims, n_contestants)
    
    # Find bottom 2 (highest total scores)
    # Use argpartition for efficiency
    if n_contestants == 2:
        # Both are bottom 2
        bottom2_indices = np.tile(np.array([0, 1]), (n_sims, 1))
    else:
        # Get indices of 2 highest total scores per simulation
        # argpartition puts the k largest at the end
        partition_idx = np.argpartition(total_scores, -2, axis=1)[:, -2:]
        bottom2_indices = partition_idx
    
    eliminated_idx = np.zeros(n_sims, dtype=int)
    
    for sim in range(n_sims):
        b2_idx = bottom2_indices[sim]
        
        # Get judge scores for bottom 2
        b2_judge_scores = judge_scores[b2_idx]
        
        # The one with LOWER judge score is eliminated (other is saved)
        if b2_judge_scores[0] < b2_judge_scores[1]:
            eliminated_idx[sim] = b2_idx[0]
        elif b2_judge_scores[1] < b2_judge_scores[0]:
            eliminated_idx[sim] = b2_idx[1]
        else:
            # Tie in judge scores - use fan rank as tie-breaker
            b2_fan_ranks = fan_ranks[sim, b2_idx]
            if b2_fan_ranks[0] > b2_fan_ranks[1]:
                eliminated_idx[sim] = b2_idx[0]
            else:
                eliminated_idx[sim] = b2_idx[1]
    
    return eliminated_idx


def apply_rank_with_save_rule_vectorized(judge_ranks: np.ndarray, judge_scores: np.ndarray, 
                                          fan_shares: np.ndarray) -> np.ndarray:
    """
    Vectorized version of RANK_WITH_SAVE elimination rule.
    
    Args:
        judge_ranks: Array of shape (n_contestants,) with judge rankings
        judge_scores: Array of shape (n_contestants,) with normalized judge scores
        fan_shares: Array of shape (n_sims, n_contestants) with fan shares
        
    Returns:
        Array of shape (n_sims,) with index of eliminated contestant per simulation
    """
    n_sims, n_contestants = fan_shares.shape
    
    if n_contestants < 2:
        return np.zeros(n_sims, dtype=int)
    
    # Convert fan shares to ranks
    fan_ranks = shares_to_ranks(fan_shares)
    
    # Total score = Judge Rank + Fan Rank
    total_scores = judge_ranks + fan_ranks
    
    # Add small noise based on fan rank for deterministic tie-breaking
    noise = fan_ranks * 1e-6
    total_scores_noisy = total_scores + noise
    
    if n_contestants == 2:
        # Simple case: only 2 contestants
        # The one with lower judge score is eliminated
        if judge_scores[0] < judge_scores[1]:
            return np.zeros(n_sims, dtype=int)
        elif judge_scores[1] < judge_scores[0]:
            return np.ones(n_sims, dtype=int)
        else:
            # Tie: worse fan rank eliminated
            return np.argmax(fan_ranks, axis=1)
    
    # Find indices of bottom 2 (2 highest total scores)
    # argsort descending to get worst performers first
    sorted_indices = np.argsort(-total_scores_noisy, axis=1)
    bottom2_first = sorted_indices[:, 0]   # Worst performer
    bottom2_second = sorted_indices[:, 1]  # Second worst
    
    # Get judge scores for bottom 2
    judge_scores_first = judge_scores[bottom2_first]
    judge_scores_second = judge_scores[bottom2_second]
    
    # Decision: Lower judge score gets eliminated
    # If first has lower score, eliminate first; else eliminate second
    eliminated_idx = np.where(
        judge_scores_first < judge_scores_second,
        bottom2_first,
        np.where(
            judge_scores_second < judge_scores_first,
            bottom2_second,
            # Tie in judge scores: use fan rank (already incorporated in noise)
            bottom2_first  # The noisy sorting already broke ties by fan rank
        )
    )
    
    return eliminated_idx


# =============================================================================
# Simulation Engine
# =============================================================================
def simulate_week(week_data: pd.DataFrame, n_sims: int = N_SIMULATIONS) -> Optional[pd.DataFrame]:
    """
    Run Monte Carlo simulation for a single week.
    
    Args:
        week_data: DataFrame with all contestants for one (season, week)
        n_sims: Number of simulation trials
        
    Returns:
        DataFrame with estimated fan shares, or None if no elimination
    """
    season = week_data['season'].iloc[0]
    week = week_data['week'].iloc[0]
    rule_system = week_data['rule_system'].iloc[0]
    
    # Get actual elimination info
    actual_eliminated_mask = week_data['is_eliminated'].values
    
    # Check if there's an actual elimination this week
    if not actual_eliminated_mask.any():
        # No elimination this week (e.g., finals, non-elimination week)
        # Return Dirichlet mean as prior estimate
        n_contestants = len(week_data)
        prior_mean = 1.0 / n_contestants  # Dirichlet mean with equal alpha
        
        result = week_data.copy()
        result['estimated_fan_share'] = prior_mean
        result['share_std'] = np.nan
        result['confidence'] = np.nan
        result['n_valid_sims'] = np.nan  # Use NaN to indicate "skipped/non-applicable"
        
        return result
    
    # Get contestant info in consistent order
    contestants = week_data['celebrity_name'].values
    n_contestants = len(contestants)
    
    # Find actual eliminated contestant index
    actual_eliminated_idx = np.where(actual_eliminated_mask)[0][0]
    
    # Extract judge data
    judge_shares = week_data['judge_share'].values.astype(float)
    judge_ranks = week_data['judge_rank'].values.astype(float)
    judge_scores = week_data['normalized_score'].values.astype(float)
    
    # Generate fan share samples
    fan_shares = generate_fan_shares(n_sims, n_contestants, alpha=DIRICHLET_ALPHA)
    
    # Apply elimination rule based on era
    if rule_system == "Rank":
        simulated_eliminated = apply_rank_rule(judge_ranks, fan_shares)
    elif rule_system == "Percent":
        simulated_eliminated = apply_percent_rule(judge_shares, fan_shares)
    elif rule_system == "Rank_With_Save":
        simulated_eliminated = apply_rank_with_save_rule_vectorized(
            judge_ranks, judge_scores, fan_shares
        )
    else:
        # Unknown rule, use Percent as default
        simulated_eliminated = apply_percent_rule(judge_shares, fan_shares)
    
    # Filter: Keep simulations that match actual outcome
    valid_mask = (simulated_eliminated == actual_eliminated_idx)
    n_valid = valid_mask.sum()
    
    if n_valid == 0:
        # No valid simulations found - model couldn't explain this outcome
        # Return prior with low confidence
        prior_mean = 1.0 / n_contestants
        
        result = week_data.copy()
        result['estimated_fan_share'] = prior_mean
        result['share_std'] = np.nan
        result['confidence'] = 0.0
        result['n_valid_sims'] = 0
        
        return result
    
    # Aggregate valid simulations
    valid_fan_shares = fan_shares[valid_mask]
    
    estimated_shares = valid_fan_shares.mean(axis=0)
    share_stds = valid_fan_shares.std(axis=0)
    confidence = n_valid / n_sims
    
    # Build result DataFrame - keep all original columns
    result = week_data.copy()
    result['estimated_fan_share'] = estimated_shares
    result['share_std'] = share_stds
    result['confidence'] = confidence
    result['n_valid_sims'] = n_valid
    
    return result


def run_simulation_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run Monte Carlo simulation for all (season, week) groups.
    
    Args:
        df: Input DataFrame from data processing
        
    Returns:
        DataFrame with fan share estimates for all contestants
    """
    print(f"\n{'='*60}")
    print("MONTE CARLO SIMULATION - FAN VOTE ESTIMATION")
    print(f"{'='*60}")
    print(f"Configuration:")
    print(f"  - N_SIMULATIONS: {N_SIMULATIONS:,}")
    print(f"  - DIRICHLET_ALPHA: {DIRICHLET_ALPHA}")
    print(f"  - Random Seed: {RANDOM_SEED}")
    
    # Set random seed
    np.random.seed(RANDOM_SEED)
    
    # Group by (season, week)
    groups = df.groupby(['season', 'week'])
    n_groups = len(groups)
    
    print(f"\nProcessing {n_groups} (season, week) groups...")
    
    results = []
    
    for (season, week), group_data in tqdm(groups, desc="Simulating", total=n_groups):
        week_result = simulate_week(group_data.reset_index(drop=True))
        if week_result is not None:
            results.append(week_result)
    
    # Combine all results
    result_df = pd.concat(results, ignore_index=True)
    
    return result_df


# =============================================================================
# Analysis & Output
# =============================================================================
def analyze_results(df: pd.DataFrame) -> None:
    """
    Print analysis summary of simulation results.
    
    Args:
        df: Results DataFrame
    """
    print(f"\n{'-'*60}")
    print("SIMULATION RESULTS ANALYSIS")
    print(f"{'-'*60}")
    
    # Overall statistics
    print(f"\nTotal rows: {len(df)}")
    
    # Confidence analysis
    valid_confidence = df[df['confidence'].notna()]
    if len(valid_confidence) > 0:
        print(f"\nConfidence Statistics (where elimination occurred):")
        print(f"  - Mean confidence: {valid_confidence['confidence'].mean():.4f}")
        print(f"  - Median confidence: {valid_confidence['confidence'].median():.4f}")
        print(f"  - Min confidence: {valid_confidence['confidence'].min():.4f}")
        print(f"  - Max confidence: {valid_confidence['confidence'].max():.4f}")
    
    # Zero confidence cases (model failures)
    # Only count rows where confidence is 0 AND simulation was actually attempted (n_valid_sims is not NaN)
    zero_conf = df[(df['confidence'] == 0) & (df['n_valid_sims'].notna()) & (df['n_valid_sims'] != 0)]
    # Or simpler: if n_valid_sims is 0, confidence is 0. If n_valid_sims is NaN, confidence is NaN.
    # We want cases where n_valid_sims == 0 (explicit failure)
    
    explicit_failures = df[df['n_valid_sims'] == 0]
    
    if len(explicit_failures) > 0:
        print(f"\n⚠️  Model couldn't explain {len(explicit_failures)} contestant-weeks (Explicit Failures)")
        # Show some examples
        examples = explicit_failures.groupby(['season', 'week']).first().head(3)
        print("  Sample unexplainable cases:")
        for (s, w), row in examples.iterrows():
            print(f"    - Season {s}, Week {w}")
    else:
        print("\n✓ Model successfully explained ALL elimination events (100% Coverage)")
    
    # Fan share distribution
    print(f"\nEstimated Fan Share Statistics:")
    print(f"  - Mean: {df['estimated_fan_share'].mean():.4f}")
    print(f"  - Std: {df['estimated_fan_share'].std():.4f}")
    print(f"  - Min: {df['estimated_fan_share'].min():.4f}")
    print(f"  - Max: {df['estimated_fan_share'].max():.4f}")
    
    # Rule system breakdown
    print(f"\nResults by Rule System:")
    for rule in ['Rank', 'Percent', 'Rank_With_Save']:
        rule_data = df[df['season'].apply(lambda s: get_rule_for_season(s)) == rule]
        if len(rule_data) > 0:
            valid = rule_data[rule_data['confidence'].notna() & (rule_data['confidence'] > 0)]
            if len(valid) > 0:
                print(f"  {rule}:")
                print(f"    - Records: {len(rule_data)}")
                print(f"    - Mean Confidence: {valid['confidence'].mean():.4f}")


def get_rule_for_season(season: int) -> str:
    """Get rule system for a season."""
    if season <= 2:
        return "Rank"
    elif season <= 27:
        return "Percent"
    else:
        return "Rank_With_Save"


def save_results(df: pd.DataFrame, output_path: Path) -> None:
    """
    Save full simulation results to CSV.
    
    Args:
        df: Results DataFrame with all original columns + estimates
        output_path: Path for output file
    """
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Keep all columns, reorder to put estimates at the end
    estimate_cols = ['estimated_fan_share', 'share_std', 'confidence', 'n_valid_sims']
    other_cols = [c for c in df.columns if c not in estimate_cols]
    output_cols = other_cols + estimate_cols
    
    output_df = df[output_cols].copy()
    
    # Round for readability
    output_df['estimated_fan_share'] = output_df['estimated_fan_share'].round(6)
    output_df['share_std'] = output_df['share_std'].round(6)
    output_df['confidence'] = output_df['confidence'].round(6)
    
    output_df.to_csv(output_path, index=False)
    print(f"\n✓ Full simulation results saved to: {output_path}")
    print(f"  Total rows: {len(output_df)}")
    print(f"  Columns: {len(output_cols)}")


def display_sample_results(df: pd.DataFrame) -> None:
    """
    Display sample results from different eras.
    
    Args:
        df: Results DataFrame
    """
    print(f"\n{'-'*60}")
    print("SAMPLE RESULTS")
    print(f"{'-'*60}")
    
    display_cols = ['season', 'week', 'celebrity_name', 
                    'estimated_fan_share', 'confidence']
    
    # Season 1 (Rank Era)
    s1 = df[df['season'] == 1].head(5)
    if len(s1) > 0:
        print("\n[SAMPLE] Season 1 (Rank Era):")
        print(s1[display_cols].to_string(index=False))
    
    # Season 10 (Percent Era)
    s10 = df[df['season'] == 10].head(5)
    if len(s10) > 0:
        print("\n[SAMPLE] Season 10 (Percent Era):")
        print(s10[display_cols].to_string(index=False))
    
    # Season 30 (Rank_With_Save Era)
    s30 = df[df['season'] == 30].head(5)
    if len(s30) > 0:
        print("\n[SAMPLE] Season 30 (Rank_With_Save Era):")
        print(s30[display_cols].to_string(index=False))


# =============================================================================
# Main Pipeline
# =============================================================================
def main():
    """
    Main Monte Carlo simulation pipeline.
    """
    print("="*60)
    print("MCM 2026 Problem C - Monte Carlo Simulation Pipeline")
    print("="*60)
    
    # Load processed data
    print(f"\n[INFO] Loading data from: {INPUT_PATH}")
    df = pd.read_csv(INPUT_PATH)
    print(f"[INFO] Loaded {len(df)} records")
    
    # Run simulation
    results_df = run_simulation_pipeline(df)
    
    # Analyze and display results
    analyze_results(results_df)
    display_sample_results(results_df)
    
    # Save results
    save_results(results_df, OUTPUT_PATH)
    
    print(f"\n{'='*60}")
    print("Simulation Complete!")
    print("="*60)
    
    return results_df


if __name__ == "__main__":
    main()
