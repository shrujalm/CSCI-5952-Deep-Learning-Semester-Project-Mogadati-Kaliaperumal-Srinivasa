# Predicting NBA Championship Contenders from Team and Rotation Statistics

## Abstract

This report evaluates whether NBA postseason outcomes can be predicted from regular-season team efficiency, contextual roster features, and the top eight players in each team's rotation. Using the 2003-04 through 2023-24 seasons (21 seasons; 629 team-seasons), we compare four models under leave-one-season-out cross-validation: logistic regression, random forest, a multilayer perceptron baseline, and an attention-based roster model. In this run, Random Forest achieves the best overall accuracy, while MLP Baseline achieves the highest macro F1. The main empirical challenge remains severe class imbalance, especially for the Champion class.

## Research Question

Can a model that explicitly learns which rotation players matter most outperform simpler baselines when forecasting a team's eventual playoff depth from regular-season statistics?

## Label Definition

The task is a six-class classification problem:

| Class | Outcome |
| --- | --- |
| 0 | Missed Playoffs |
| 1 | First Round Exit |
| 2 | Second Round Exit |
| 3 | Conference Finals |
| 4 | Finals Loss |
| 5 | Champion |

## Data Provenance

- Seasons covered: 2003-04 through 2023-24
- Primary data source: `nba_api` endpoints `LeagueDashTeamStats`, `TeamEstimatedMetrics`, and `LeagueDashPlayerStats`
- Curated postseason labels: local historical dictionary extracted from the project notebook and now versioned in code
- Unit of analysis: one team-season
- Sample count: 629 team-seasons

Class distribution:

| Class | Count | Share |
| --- | --- | --- |
| 0 | 293 | 46.6% |
| 1 | 168 | 26.7% |
| 2 | 84 | 13.4% |
| 3 | 42 | 6.7% |
| 4 | 21 | 3.3% |
| 5 | 21 | 3.3% |

## Feature Construction

The final feature matrix contains:

- Team-level regular-season performance variables: win percentage, scoring, rebounding, playmaking, shot-making, plus/minus, wins/losses, and advanced efficiency metrics.
- Contextual variables: conference-strength proxy and roster continuity.
- Player-level variables for the top eight rotation players by minutes: box-score production, shooting efficiency, minutes, and an approximate PER-style impact summary.

## Experimental Protocol

- Evaluation: leave-one-season-out cross-validation
- Neural-model optimizer: Adam
- MLP epochs: 150
- Attention-model epochs: 200
- Learning rate: 0.001
- Random seed: 42
- Hardware used in this run: CPU execution within the local workspace

## Main Results

| Model | Accuracy (OOF) | Accuracy (Fold Mean +/- SD) | Macro F1 (OOF) | Top-2 Accuracy (OOF) |
| --- | --- | --- | --- | --- |
| Logistic Regression | 60.9% | 60.9% +/- 7.6% | 38.0% | 81.4% |
| Random Forest | 68.0% | 68.0% +/- 6.1% | 36.2% | 84.3% |
| MLP Baseline | 62.8% | 62.8% +/- 6.6% | 40.8% | 81.7% |
| Attention Model | 65.3% | 65.4% +/- 7.6% | 39.3% | 82.5% |

The table reports out-of-fold metrics across all team-seasons plus fold-level mean and standard deviation for accuracy. Random forest is the most accurate model in this run, the MLP baseline has the strongest macro F1, and the attention model sits between them while offering direct player-importance explanations.

## Per-Class Analysis

Champion-class performance is the hardest part of the problem because there is only one champion per season. In the attention model, Champion precision is 0.24, recall is 0.29, and F1 is 0.26. For the strongest baseline (MLP Baseline), Champion precision is 0.35, recall is 0.29, and F1 is 0.32.

Detailed per-class tables are saved at:

- `results/research_study/attention_classification_report.csv`
- `results/research_study/best_baseline_classification_report.csv`

## Visual Diagnostics

### Confusion Matrices

![Confusion matrices](figures/research_confusion_matrices.png)

### Attention Weight Profiles

![Attention weights](figures/research_attention_weights.png)

### t-SNE Projection of Team-Season Features

![t-SNE projection](figures/research_tsne.png)

## Error Analysis and Interpretation

- The easiest class remains `Missed Playoffs`, which dominates the dataset and is easier to separate from the deeper-round classes.
- The Champion and Finals-Loss classes remain sparse and frequently get confused with conference-final or second-round teams.
- The attention plots show that the model does not weight every roster slot equally, which supports the original project motivation even when aggregate predictive gains over the best baseline are modest.

## Fairness and Market-Size Check

We performed a coarse market-size audit using the attention model's predictions.

- Large-market average absolute error: 0.533
- Small-market average absolute error: 0.475
- Error gap: 0.058

This is not a comprehensive fairness analysis, but it offers a basic sanity check that prediction error is not grossly asymmetric across a simple market-size partition.

## Threats to Validity

1. The dataset is small for a six-class forecasting problem, especially at the champion tail.
2. Roster continuity and conference strength are useful but still fairly coarse contextual features.
3. The study uses full regular-season data rather than a strict All-Star-break snapshot, so the results should be interpreted as season-level forecasting rather than a pure mid-season forecast.
4. Hyperparameter search was intentionally lightweight to preserve reproducibility and avoid overfitting the small dataset.

## Conclusion

The project now has a real-data experimental pipeline and a reproducible report artifact. The empirical results show that the task is learnable to a degree, but class imbalance and limited sample size remain the main barriers to robust champion prediction. The attention model contributes interpretability and a principled way to aggregate roster information, while the strongest baseline still provides a high bar that any more complex architecture must beat consistently.

## Reproducibility Appendix

Command used to generate this report:

```bash
python run_research_study.py --epochs-mlp 150 --epochs-attention 200 --lr 0.001 --seed 42
```

Key artifact paths:

- Report: `docs/final_report.md`
- Metrics JSON: `results/research_study/summary_metrics.json`
- Model comparison CSV: `results/research_study/model_comparison.csv`
- Fold metrics CSV: `results/research_study/fold_metrics.csv`
- Predictions CSV: `results/research_study/predictions.csv`
- Figures: `docs/figures`
