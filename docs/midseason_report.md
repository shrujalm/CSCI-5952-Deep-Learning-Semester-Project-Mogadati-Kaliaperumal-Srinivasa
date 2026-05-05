# Mid-Season NBA Championship Prediction from Team and Rotation Statistics

## Abstract

This report evaluates whether NBA postseason outcomes can be predicted from All-Star-break proxy / mid-season team efficiency, contextual roster features, and the top eight players in each team's rotation. Using the 2003-04 through 2023-24 seasons (21 seasons; 629 team-seasons), we compare four models under leave-one-season-out cross-validation: logistic regression, random forest, a multilayer perceptron baseline, and an attention-based roster model. In this run, Random Forest achieves the best overall accuracy, while Attention Model achieves the highest macro F1. The main empirical challenge remains severe class imbalance, especially for the Champion class.

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

- Forecasting setting: Mid-season prediction using stats available around the All-Star break
- Seasons covered: 2003-04 through 2023-24
- Primary data source: `nba_api` endpoints `LeagueDashTeamStats` for base team stats, `LeagueDashTeamStats` with `MeasureType=Advanced` for advanced team stats, and `LeagueDashPlayerStats`, all filtered with `DateTo`/`date_to_nullable` cutoffs
- Curated postseason labels: local historical dictionary extracted from the project notebook and now versioned in code
- Unit of analysis: one team-season
- Sample count: 629 team-seasons

Additional data notes:

- Features use only regular-season statistics through season-specific cutoffs in `MIDSEASON_CUTOFF_DATES`.
- Labels still use the final playoff outcome: 0 = Missed Playoffs through 5 = Champion.
- The processed artifact is `data/processed/features_midseason.csv`.


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

The mid-season feature matrix matches the full-season schema exactly and contains:

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
| Logistic Regression | 54.8% | 54.8% +/- 7.7% | 32.4% | 74.1% |
| Random Forest | 60.4% | 60.4% +/- 4.2% | 28.6% | 81.9% |
| MLP Baseline | 57.9% | 57.9% +/- 8.1% | 32.6% | 77.6% |
| Attention Model | 60.4% | 60.4% +/- 9.4% | 37.1% | 80.1% |

The table reports out-of-fold metrics across all team-seasons plus fold-level mean and standard deviation for accuracy. This setting is intentionally harder than the full-season experiment because every feature is limited to information available around the mid-season cutoff.

### Real Results Artifact Summary

The metrics above are copied from generated artifacts in `results/midseason_study`, not estimated or mocked. `predictions.csv` contains 629 out-of-fold team-season predictions plus the header row, matching the 629 team-seasons described in the data section. The main model ranking comes from `model_comparison.csv` and `summary_metrics.json`:

- Best OOF accuracy: Random Forest at 0.604134.
- Best OOF macro F1: Attention Model at 0.370895.
- Best OOF top-2 accuracy: Random Forest at 0.818760.
- Attention model OOF metrics: accuracy 0.604134, macro F1 0.370895, top-2 accuracy 0.801272.
- Best single-season fold accuracy observed: Attention Model at 0.833333 in 2016-17.

## Per-Class Analysis

Champion-class performance is the hardest part of the problem because there is only one champion per season. In the attention model, Champion precision is 0.19, recall is 0.24, and F1 is 0.21. For the strongest baseline (MLP Baseline), Champion precision is 0.12, recall is 0.10, and F1 is 0.11.

Detailed per-class tables are saved at:

- `results/midseason_study/attention_classification_report.csv`
- `results/midseason_study/best_baseline_classification_report.csv`

## Visual Diagnostics

### Confusion Matrices

![Confusion matrices](figures_midseason/midseason_confusion_matrices.png)

### Attention Weight Profiles

![Attention weights](figures_midseason/midseason_attention_weights.png)

### t-SNE Projection of Team-Season Features

![t-SNE projection](figures_midseason/midseason_tsne.png)

## Error Analysis and Interpretation

- The easiest class remains `Missed Playoffs`, which dominates the dataset and is easier to separate from the deeper-round classes.
- The Champion and Finals-Loss classes remain sparse and frequently get confused with conference-final or second-round teams.
- The attention plots show that the model does not weight every roster slot equally, which supports the original project motivation even when aggregate predictive gains over the best baseline are modest.

## Fairness and Market-Size Check

We performed a coarse market-size audit using the attention model's predictions.

- Large-market average absolute error: 0.710
- Small-market average absolute error: 0.556
- Error gap: 0.153

This is not a comprehensive fairness analysis, but it offers a basic sanity check that prediction error is not grossly asymmetric across a simple market-size partition.

## Threats to Validity

1. The cutoff dictionary approximates each season's All-Star Sunday or a nearby mid-February date, so it should be interpreted as an All-Star-break proxy rather than an exact betting-market timestamp.
2. The 2020-21 season uses March 7, 2021 because the NBA All-Star Game moved later on the shortened COVID calendar.
3. Advanced mid-season columns are mapped from OFF_RATING, DEF_RATING, NET_RATING, and PACE to the full-season E_* schema; if an endpoint omits one, the pipeline fills that feature with 0.0.
4. The label remains the final postseason result, so trades, injuries, and form changes after the cutoff can legitimately change the outcome.
5. The champion and finals-loss classes remain very sparse for a six-class problem.

## Conclusion

The project now has a real-data experimental pipeline and a reproducible report artifact. The empirical results show that the task is learnable to a degree, but class imbalance and limited sample size remain the main barriers to robust champion prediction. The attention model contributes interpretability and a principled way to aggregate roster information, while the strongest baseline still provides a high bar that any more complex architecture must beat consistently.

## Reproducibility Appendix

Command used to generate this report:

```bash
python run_midseason_study.py --epochs-mlp 150 --epochs-attention 200 --lr 0.001 --seed 42
```

Key artifact paths:

- Report: `docs/midseason_report.md`
- Metrics JSON: `results/midseason_study/summary_metrics.json`
- Model comparison CSV: `results/midseason_study/model_comparison.csv`
- Fold metrics CSV: `results/midseason_study/fold_metrics.csv`
- Predictions CSV: `results/midseason_study/predictions.csv`
- Figures: `docs/figures_midseason`
