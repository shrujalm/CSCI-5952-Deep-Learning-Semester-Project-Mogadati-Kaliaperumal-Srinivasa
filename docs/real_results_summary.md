# Real Results Summary

This document collects the real generated results for the NBA Championship Outcome Prediction project. All numbers below come from the committed CSV/JSON artifacts under `results/research_study/` and `results/midseason_study/`. No mock data or hand-substituted results are used.

## Dataset Artifacts

| Setting | Feature File | Rows | Columns | Seasons | Labels |
| --- | --- | ---: | ---: | --- | --- |
| Full regular season | `data/processed/features_research.csv` | 629 | 112 | 2003-04 through 2023-24 | Final playoff outcome |
| Mid-season / All-Star proxy | `data/processed/features_midseason.csv` | 629 | 112 | 2003-04 through 2023-24 | Final playoff outcome |

The full-season and mid-season processed files have identical schemas. Both contain the metadata columns `SEASON`, `TEAM`, `TEAM_ID`, `PLAYOFF_RESULT`, and `PLAYOFF_LABEL`, followed by the same team/context features and `P1_` through `P8_` rotation-player features.

Class distribution for both settings:

| Class | Outcome | Count | Share |
| ---: | --- | ---: | ---: |
| 0 | Missed Playoffs | 293 | 46.6% |
| 1 | First Round Exit | 168 | 26.7% |
| 2 | Second Round Exit | 84 | 13.4% |
| 3 | Conference Finals | 42 | 6.7% |
| 4 | Finals Loss | 21 | 3.3% |
| 5 | Champion | 21 | 3.3% |

## Full Regular-Season Experiment

Source artifacts:

- `results/research_study/model_comparison.csv`
- `results/research_study/summary_metrics.json`
- `results/research_study/fold_metrics.csv`
- `results/research_study/predictions.csv`
- `results/research_study/attention_classification_report.csv`
- `results/research_study/best_baseline_classification_report.csv`
- `results/research_study/market_fairness.json`

### Model Comparison

| Model | Accuracy (OOF) | Fold Accuracy Mean | Fold Accuracy SD | Macro F1 (OOF) | Fold Macro F1 Mean | Top-2 Accuracy (OOF) | Fold Top-2 Mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.608903 | 0.608976 | 0.076018 | 0.380101 | 0.357038 | 0.813990 | 0.814176 |
| Random Forest | 0.680445 | 0.680405 | 0.060663 | 0.362342 | 0.349697 | 0.842607 | 0.842693 |
| MLP Baseline | 0.627981 | 0.627805 | 0.066489 | 0.407933 | 0.384171 | 0.817170 | 0.817241 |
| Attention Model | 0.653418 | 0.653585 | 0.076073 | 0.393419 | 0.380243 | 0.825119 | 0.825178 |

Best full-season results:

- Best exact accuracy: Random Forest, 0.680445.
- Best macro F1: MLP Baseline, 0.407933.
- Best top-2 accuracy: Random Forest, 0.842607.
- Attention Model: 0.653418 accuracy, 0.393419 macro F1, 0.825119 top-2 accuracy.

### Full-Season Fold Extremes

| Model | Lowest Accuracy Fold | Highest Accuracy Fold |
| --- | --- | --- |
| Logistic Regression | 2014-15, 0.500000 accuracy, 0.224442 macro F1, 0.733333 top-2 | 2017-18, 0.766667 accuracy, 0.504098 macro F1, 0.833333 top-2 |
| Random Forest | 2012-13, 0.566667 accuracy, 0.231481 macro F1, 0.766667 top-2 | 2016-17, 0.800000 accuracy, 0.620811 macro F1, 0.933333 top-2 |
| MLP Baseline | 2003-04, 0.517241 accuracy, 0.276786 macro F1, 0.862069 top-2 | 2016-17, 0.766667 accuracy, 0.682066 macro F1, 0.933333 top-2 |
| Attention Model | 2022-23, 0.500000 accuracy, 0.212698 macro F1, 0.766667 top-2 | 2016-17, 0.800000 accuracy, 0.680556 macro F1, 0.933333 top-2 |

Complete fold-level results for every season and model are in `results/research_study/fold_metrics.csv`.

### Attention Model Per-Class Results

| Class | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| Missed Playoffs | 0.908 | 0.904 | 0.906 | 293 |
| First Round Exit | 0.631 | 0.601 | 0.616 | 168 |
| Second Round Exit | 0.372 | 0.345 | 0.358 | 84 |
| Conference Finals | 0.204 | 0.238 | 0.220 | 42 |
| Finals Loss | 0.000 | 0.000 | 0.000 | 21 |
| Champion | 0.240 | 0.286 | 0.261 | 21 |
| Macro Average | 0.392 | 0.396 | 0.393 | 629 |
| Weighted Average | 0.663 | 0.653 | 0.658 | 629 |

### Best Baseline Per-Class Results

The best baseline by macro F1 is the MLP Baseline.

| Class | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| Missed Playoffs | 0.874 | 0.874 | 0.874 | 293 |
| First Round Exit | 0.533 | 0.536 | 0.534 | 168 |
| Second Round Exit | 0.315 | 0.345 | 0.330 | 84 |
| Conference Finals | 0.293 | 0.286 | 0.289 | 42 |
| Finals Loss | 0.118 | 0.095 | 0.105 | 21 |
| Champion | 0.353 | 0.286 | 0.316 | 21 |
| Macro Average | 0.414 | 0.404 | 0.408 | 629 |
| Weighted Average | 0.627 | 0.628 | 0.627 | 629 |

### Full-Season Market-Size Audit

| Metric | Value |
| --- | ---: |
| Large-market actual mean label | 1.257143 |
| Small-market actual mean label | 0.923628 |
| Large-market predicted mean label | 1.304762 |
| Small-market predicted mean label | 0.988067 |
| Large-market average absolute error | 0.533333 |
| Small-market average absolute error | 0.474940 |
| Error gap | 0.058393 |

## Mid-Season Experiment

Source artifacts:

- `results/midseason_study/model_comparison.csv`
- `results/midseason_study/summary_metrics.json`
- `results/midseason_study/fold_metrics.csv`
- `results/midseason_study/predictions.csv`
- `results/midseason_study/attention_classification_report.csv`
- `results/midseason_study/best_baseline_classification_report.csv`
- `results/midseason_study/market_fairness.json`

The mid-season experiment uses only statistics available around season-specific All-Star-break proxy dates, while keeping the final playoff outcome label unchanged.

### Model Comparison

| Model | Accuracy (OOF) | Fold Accuracy Mean | Fold Accuracy SD | Macro F1 (OOF) | Fold Macro F1 Mean | Top-2 Accuracy (OOF) | Fold Top-2 Mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.548490 | 0.548440 | 0.076952 | 0.323555 | 0.318307 | 0.740859 | 0.740887 |
| Random Forest | 0.604134 | 0.604160 | 0.042450 | 0.285779 | 0.275847 | 0.818760 | 0.818719 |
| MLP Baseline | 0.578696 | 0.578654 | 0.081483 | 0.326271 | 0.312399 | 0.775835 | 0.775862 |
| Attention Model | 0.604134 | 0.604269 | 0.093969 | 0.370895 | 0.356962 | 0.801272 | 0.801314 |

Best mid-season results:

- Best exact accuracy: Random Forest and Attention Model tie at 0.604134.
- Best macro F1: Attention Model, 0.370895.
- Best top-2 accuracy: Random Forest, 0.818760.
- Attention Model: 0.604134 accuracy, 0.370895 macro F1, 0.801272 top-2 accuracy.

### Mid-Season Fold Extremes

| Model | Lowest Accuracy Fold | Highest Accuracy Fold |
| --- | --- | --- |
| Logistic Regression | 2022-23, 0.400000 accuracy, 0.282323 macro F1, 0.533333 top-2 | 2016-17, 0.766667 accuracy, 0.718838 macro F1, 0.933333 top-2 |
| Random Forest | 2004-05, 0.500000 accuracy, 0.198413 macro F1, 0.733333 top-2 | 2012-13, 0.666667 accuracy, 0.271605 macro F1, 0.800000 top-2 |
| MLP Baseline | 2004-05, 0.433333 accuracy, 0.210826 macro F1, 0.633333 top-2 | 2016-17, 0.733333 accuracy, 0.687007 macro F1, 0.933333 top-2 |
| Attention Model | 2014-15, 0.433333 accuracy, 0.221429 macro F1, 0.766667 top-2 | 2016-17, 0.833333 accuracy, 0.885536 macro F1, 1.000000 top-2 |

Complete fold-level results for every season and model are in `results/midseason_study/fold_metrics.csv`.

### Attention Model Per-Class Results

| Class | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| Missed Playoffs | 0.869 | 0.857 | 0.863 | 293 |
| First Round Exit | 0.518 | 0.506 | 0.512 | 168 |
| Second Round Exit | 0.317 | 0.310 | 0.313 | 84 |
| Conference Finals | 0.286 | 0.286 | 0.286 | 42 |
| Finals Loss | 0.040 | 0.048 | 0.043 | 21 |
| Champion | 0.185 | 0.238 | 0.208 | 21 |
| Macro Average | 0.369 | 0.374 | 0.371 | 629 |
| Weighted Average | 0.612 | 0.604 | 0.608 | 629 |

### Best Baseline Per-Class Results

The best baseline by macro F1 is the MLP Baseline.

| Class | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| Missed Playoffs | 0.841 | 0.829 | 0.835 | 293 |
| First Round Exit | 0.459 | 0.470 | 0.465 | 168 |
| Second Round Exit | 0.333 | 0.405 | 0.366 | 84 |
| Conference Finals | 0.143 | 0.119 | 0.130 | 42 |
| Finals Loss | 0.071 | 0.048 | 0.057 | 21 |
| Champion | 0.118 | 0.095 | 0.105 | 21 |
| Macro Average | 0.328 | 0.328 | 0.326 | 629 |
| Weighted Average | 0.575 | 0.579 | 0.576 | 629 |

### Mid-Season Market-Size Audit

| Metric | Value |
| --- | ---: |
| Large-market actual mean label | 1.257143 |
| Small-market actual mean label | 0.923628 |
| Large-market predicted mean label | 1.261905 |
| Small-market predicted mean label | 1.011933 |
| Large-market average absolute error | 0.709524 |
| Small-market average absolute error | 0.556086 |
| Error gap | 0.153438 |

## Full Season vs. Mid-Season Comparison

| Model | Full Accuracy | Mid Accuracy | Accuracy Change | Full Macro F1 | Mid Macro F1 | Macro F1 Change | Full Top-2 | Mid Top-2 | Top-2 Change |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.608903 | 0.548490 | -0.060413 | 0.380101 | 0.323555 | -0.056545 | 0.813990 | 0.740859 | -0.073132 |
| Random Forest | 0.680445 | 0.604134 | -0.076312 | 0.362342 | 0.285779 | -0.076563 | 0.842607 | 0.818760 | -0.023847 |
| MLP Baseline | 0.627981 | 0.578696 | -0.049285 | 0.407933 | 0.326271 | -0.081662 | 0.817170 | 0.775835 | -0.041335 |
| Attention Model | 0.653418 | 0.604134 | -0.049285 | 0.393419 | 0.370895 | -0.022524 | 0.825119 | 0.801272 | -0.023847 |

All models lose accuracy in the mid-season setting, which is expected because the features contain less information. The Attention Model has the smallest macro-F1 drop among the four models, while Random Forest remains strongest on top-2 accuracy.

## Complete Row-Level Results

The row-level out-of-fold predictions are not duplicated in this Markdown file because each predictions file contains 629 team-season rows. The complete prediction tables are:

- Full regular season: `results/research_study/predictions.csv`
- Mid-season: `results/midseason_study/predictions.csv`

Each predictions file includes `SEASON`, `TEAM`, `PLAYOFF_RESULT`, and one prediction column per model.

## Figure Artifacts

Full regular-season figures:

- `docs/figures/research_confusion_matrices.png`
- `docs/figures/research_attention_weights.png`
- `docs/figures/research_tsne.png`
- `docs/figures/research_tsne_cluster_analysis.png`
- `docs/figures/finals_team_prediction_heatmap.png`
- `docs/figures/finals_case_study_predictions.png`
- `docs/figures/finals_detection_by_model.png`
- `docs/figures/full_season_results_random_forest_attention.png`

Mid-season figures:

- `docs/figures_midseason/midseason_confusion_matrices.png`
- `docs/figures_midseason/midseason_attention_weights.png`
- `docs/figures_midseason/midseason_tsne.png`

Additional presentation figures can be regenerated with:

```powershell
.\.venv\Scripts\python.exe scripts\generate_additional_figures.py
```
