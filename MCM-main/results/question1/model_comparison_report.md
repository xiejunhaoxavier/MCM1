# Model Evaluation Report: Basic vs. Bayesian

## 1. Executive Summary
- **Total Elimination Weeks Analyzed**: 264
- **Basic Model Explanation Rate**: 99.6% (263/264)
- **Bayesian Model Explanation Rate**: 99.2% (262/264)
- **Net Improvement**: +-0.4% (0 weeks recovered)

## 2. Key Metrics by Era
Evaluation excludes non-elimination weeks (e.g. withdrawals, finals).

| Rule System | Basic Rate | Bayesian Rate | Improvement | Avg Certainty (Basic -> Bayes) | Avg Stability (Basic -> Bayes) |
|---|---|---|---|---|---|
| Percent | 100.0% | 100.0% | +0.0% | 0.213 -> 0.362 | 0.114 -> 0.069 |
| Rank | 100.0% | 100.0% | +0.0% | 0.306 -> 0.309 | 0.134 -> 0.094 |
| Rank_With_Save | 98.2% | 96.4% | +-1.8% | 0.296 -> 0.315 | 0.093 -> 0.066 |


## 3. Notable Recovered Cases
Weeks where Basic Model failed (valid_sims=0) but Bayesian Model succeeded:

| Season | Week | Result |
|---|---|---|

## 4. Visualization Index
- [Explanation Rate Comparison](plots/explanation_rate_comparison.png)
- [Certainty Distribution](plots/certainty_distribution.png)
- [Stability Distribution](plots/stability_distribution.png)
- [Trajectory: Bobby Bones](plots/trajectory_Bobby_Bones_S27.png)
- [Trajectory: Sean Spicer](plots/trajectory_Sean_Spicer_S28.png)
