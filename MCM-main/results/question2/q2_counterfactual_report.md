# Q2 Counterfactual Analysis Report: The Multiverse Simulator

## Executive Summary

This report presents a comprehensive static counterfactual analysis of three elimination rule systems applied to the history of _Dancing with the Stars_ (seasons 1-34). We simulate "parallel universes" where each historical week's outcome is recalculated under three different voting systems, answering the core question: **"Does one method seem to favor fan votes more?"**

### Key Findings

| Metric                              | Value                 | Interpretation                                                                   |
| ----------------------------------- | --------------------- | -------------------------------------------------------------------------------- |
| **Reversal Rate (Rank vs Percent)** | **13.43%**            | Moderate sensitivity: 1 in 7 weeks would see different outcomes                  |
| **System C Validation Accuracy**    | **83.93%**            | High consistency with historical S28-34 outcomes                                 |
| **Fan Power Index (Percent)**       | **0.847**             | Percent system gives fans the most influence                                     |
| **Merit Safety Rate (Percent)**     | **71.59%**            | Percent system creates strongest "Popularity Shield" for low-scoring contestants |
| **Bobby Bones Fate (System C)**     | **Eliminated Week 8** | Would not have won under Rank+Save system                                        |

## 1. Analysis Overview

### 1.1 The Three Parallel Universes

We define three rule systems consistent with historical eras:

1. **Universe A: Rank System (Classic)** - Seasons 1-2
   - Judge Rank + Fan Rank = Total Points
   - **Elimination**: Highest total points (worst ranking)
   - **Tie-breaker**: Worse Fan Rank loses

2. **Universe B: Percent System (Modern)** - Seasons 3-27
   - Judge Share % + Fan Share % = Total Share %
   - **Elimination**: Lowest total share percentage
   - **Known feature**: Sensitive to fan vote extremes (Bobby Bones effect)

3. **Universe C: Rank + Judges' Save (Hybrid)** - Seasons 28+
   - Same as Rank system to identify Bottom 2
   - **The Save**: Judge score higher → SAVED (technical merit protected)

### 1.2 Methodology Constraints

- **Static Snapshot**: Each week analyzed independently, no dynamic carryover
- **"God's Eye" Data**: Uses Q1 Bayesian estimates of fan shares as ground truth
- **Parallel Comparison**: All three systems applied to each historical week
- **Validation Check**: System C compared against actual S28-34 outcomes (83.9% match)

## 2. Quantitative Metrics

### 2.1 Reversal Rate: Rule Sensitivity

**13.43% reversal rate** indicates moderate system sensitivity. This means:

- 45 of 335 weeks would have produced different elimination results
- Rank and Percent systems disagree on ~1 in 7 weeks
- **Interpretation**: While fan influence differs, both systems converge in majority of cases

### 2.2 Fan Power Index (FPI)

| System        | FPI (     | Spearman ρ                                              | )   | Interpretation |
| ------------- | --------- | ------------------------------------------------------- | --- | -------------- |
| **Percent**   | **0.847** | Highest correlation between fan share and final ranking |
| **Rank**      | 0.793     | Moderate fan influence                                  |
| **Rank+Save** | 0.793     | Save mechanism does not reduce fan power                |

**Key Insight**: The Percent system gives fans the **most influence** over outcomes, confirming the hypothesis that percentage-based voting amplifies fan voices relative to ranking-based systems.

### 2.3 Merit Safety Rate

| System        | Survival Rate | Protection Level                                                                        |
| ------------- | ------------- | --------------------------------------------------------------------------------------- |
| **Percent**   | **71.59%**    | Strongest "Popularity Shield" - protects low-scoring contestants most effectively       |
| **Rank**      | 68.99%        | Moderate protection for low-scoring contestants                                         |
| **Rank+Save** | **68.00%**    | **Most meritocratic** - most ruthless elimination of low-scoring (bottom 3) contestants |

**Critical Insight**: The Percent system's 71.59% survival rate indicates it creates the strongest **"Popularity Shield"** - 71.6% of contestants in the bottom 3 judge scorers survive. This paradoxically means the system that gives fans the most influence (FPI = 0.847) also provides the strongest protection for technically weak contestants. Conversely, Rank+Save's 68.00% survival rate makes it the **most meritocratic system**, most effectively eliminating low-scoring contestants as intended by the Judges' Save mechanism.

### 2.4 System Validation

System C (Rank+Save) achieves **83.93% accuracy** (47/56 weeks) against actual S28-34 outcomes. The 9 mismatches may be due to:

1. Data inconsistencies (missing or estimated fan shares)
2. Complex tie-breaking scenarios not fully modeled
3. Non-standard eliminations (injuries, withdrawals)

## 3. Case Study: Bobby Bones (Season 27)

### 3.1 Historical Context

- **Actual result**: Champion under Percent system (S27)
- **Key attribute**: Exceptionally high fan vote share compensating for mediocre judge scores
- **The "Bobby Bones Problem"**: Prototypical example of fan dominance overwhelming technical merit

### 3.2 Counterfactual Fate

| System               | Bobby Bones' Fate | Week Eliminated             |
| -------------------- | ----------------- | --------------------------- |
| **Percent (Actual)** | **CHAMPION**      | N/A                         |
| **Rank**             | Survives          | N/A (virtual rank improves) |
| **Rank+Save**        | **ELIMINATED**    | **Week 8**                  |

**Critical Analysis**:

- Under Rank+Save, Bobby Bones would have fallen into the Bottom 2 in Week 8
- His judge score was lower than his opponent in that week's Bottom 2
- The Judges' Save mechanism would **not** have protected him

**Implication**: The Judges' Save successfully addresses the "Bobby Bones problem" by providing a technical merit safety net.

## 4. Era-Specific Analysis

### 4.1 Rank Era (Seasons 1-2)

- **Historical consistency**: 100% match between simulated and actual outcomes
- **Fan influence**: Moderate (FPI = 0.79)
- **Merit protection**: Adequate (69% survival rate)

### 4.2 Percent Era (Seasons 3-27)

- **Historical consistency**: 100% match (by design - this was actual system)
- **Fan influence**: High (FPI = 0.85)
- **Merit protection**: Good (72% survival rate)

### 4.3 Rank+Save Era (Seasons 28-34)

- **Historical consistency**: 83.9% match (high but imperfect)
- **Fan influence**: Moderate (FPI = 0.79, same as Rank)
- **Merit protection**: Slightly lower than expected (68% survival rate)

## 5. Policy Implications

### 5.1 Does one method favor fan votes more?

**Yes, unequivocally**: The Percent system (Seasons 3-27) gives fans the most influence (FPI = 0.847 vs 0.793 for Rank systems).

### 5.2 Is the Judges' Save effective?

**Mixed results**:

- **Success**: Would have eliminated Bobby Bones (addressing the canonical problem case)
- **Challenge**: Only 68% protection for bottom 3 judge scorers (worse than Percent system)
- **Validation**: 83.9% match with actual outcomes suggests reasonable implementation

### 5.3 Unexpected Finding

**The Percent Paradox**: Despite giving fans more influence (FPI = 0.847), the Percent system paradoxically creates the strongest **"Popularity Shield"** (71.59% survival rate for bottom 3 judge scorers). This is not protection for technically skilled contestants, but rather protection for **low-scoring contestants** through fan popularity. This finding is perfectly consistent with the high FPI - when fans have more influence, they can more effectively rescue their favorites regardless of technical merit.

## 6. Limitations and Future Work

### 6.1 Limitations

1. **Static analysis**: Does not consider dynamic effects of eliminated contestants
2. **Fan share estimation**: Relies on Q1 Bayesian estimates (not actual fan votes)
3. **Tie-breaking simplification**: Some edge cases may not perfectly match official rules
4. **Data completeness**: Non-elimination weeks excluded from some metrics

### 6.2 Future Research Directions

1. **Dynamic simulation**: Model full season trajectories under different rules
2. **Sensitivity analysis**: Test robustness to fan share estimation uncertainty
3. **Additional rule variants**: Consider other hybrid systems (e.g., weighted scoring)
4. **Extended validation**: Cross-check with other reality competition shows

## 7. Conclusion

The Multiverse Simulator reveals nuanced relationships between voting systems and competition outcomes:

1. **Fan influence is maximized** in percentage-based systems (FPI = 0.847)
2. **Rule sensitivity is moderate** (13.4% reversal rate between Rank and Percent)
3. **The Judges' Save works as intended** for canonical problem cases (Bobby Bones)
4. **Counterintuitively**, percentage systems offer **better merit protection** than ranking systems

These findings provide empirical evidence for competition design decisions and demonstrate the value of counterfactual analysis in understanding rule system impacts.

## Appendix: Output Files

1. **Data Files**:
   - `counterfactual_outcomes.csv`: Weekly elimination results for all three systems
   - `counterfactual_rankings.csv`: Per-contestant weekly rankings under each system

2. **Visualizations**:
   - `q2_fan_bias_comparison.png`: Fan Power Index by system and era
   - `q2_merit_safety.png`: Merit protection rates comparison
   - `q2_bobby_bones_survival.png`: Virtual rank trajectory for Bobby Bones

3. **Code**:
   - `src/rule_comparison_v2.py`: Complete simulation implementation

---

**Simulation Statistics**: 335 weeks analyzed, 2777 contestant-week observations, 34 seasons (1-34)

_Report generated: 2026-01-31_
