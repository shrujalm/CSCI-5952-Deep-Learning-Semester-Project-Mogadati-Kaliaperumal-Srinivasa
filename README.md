# NBA Championship Outcome Prediction with Deep Learning

Our project studies whether regular-season NBA data can help predict how far a team will go in the postseason. We use team performance, roster context, and rotation-level player statistics to classify each team-season into one of six playoff-depth outcomes.

We built this repository as a reproducible deep learning project, not just a notebook. It includes data processing, feature engineering, classical machine learning baselines, neural models, attention-based interpretability, stored results, figures, and report files.

## Project Snapshot

| Area | Details |
| --- | --- |
| Objective | Predict a team's eventual postseason outcome from regular-season data |
| Task | 6-class classification: Missed Playoffs, First Round, Second Round, Conference Finals, Finals Loss, Champion |
| Coverage | 2003-04 through 2023-24 |
| Sample size | 629 team-seasons across 21 seasons |
| Inputs | Team box-score stats, advanced efficiency metrics, roster continuity, conference context, top-8 rotation player features |
| Models | Logistic Regression, Random Forest, MLP Baseline, Attention Model |
| Evaluation | Leave-one-season-out cross-validation |
| Best accuracy | Random Forest: 68.0% |
| Best macro F1 | MLP Baseline: 40.8% |
| Interpretable deep model | Attention Model: 65.3% accuracy, 39.3% macro F1 |

## Executive Summary

We wanted to know whether a roster-aware attention model could improve playoff-depth prediction, especially by learning which rotation players mattered most for each team. To test that, we collected real NBA data, engineered team and player features, and evaluated every model with leave-one-season-out cross-validation.

Our results were more realistic than dramatic. The task is learnable, but it is hard. Random Forest gave us the highest overall accuracy, the MLP baseline had the best macro F1, and our Attention Model stayed competitive while also giving us player-slot attention weights. That tradeoff became one of the main points of our project: deep learning helped with interpretation, but it did not automatically beat the strongest baseline.

## What This Repo Shows

- We define a clear sports analytics research question and evaluate it with measurable results.
- We move the project beyond a single notebook by organizing the code as a reusable Python package.
- We build structured features from team stats, advanced metrics, roster continuity, and top-eight player rotations.
- We compare classical models and deep learning models instead of assuming the neural model should win.
- We include attention diagnostics, error analysis, and a simple market-size fairness audit.
- We commit figures, metrics, predictions, and reports so someone can review our work without rerunning the full study first.

## Research Question

Can a model that learns which rotation players matter most outperform simpler baselines when predicting a team's final playoff depth from regular-season statistics?

## Dataset and Label Design

Each row in our dataset is one NBA team in one season. The label is the team's final postseason result for that season.

### Label Space

| Class | Outcome |
| --- | --- |
| 0 | Missed Playoffs |
| 1 | First Round Exit |
| 2 | Second Round Exit |
| 3 | Conference Finals |
| 4 | Finals Loss |
| 5 | Champion |

### Data Sources

- `nba_api` regular-season team statistics through `LeagueDashTeamStats`
- Advanced team metrics through `TeamEstimatedMetrics`
- Player statistics through `LeagueDashPlayerStats`
- Curated historical playoff labels stored in the codebase

### Engineered Features

- Team-level variables such as win percentage, scoring, rebounding, assists, shooting efficiency, plus/minus, wins, and losses
- Advanced efficiency metrics such as offensive rating, defensive rating, and net rating
- Context features for conference strength and roster continuity
- Rotation-level player features for the top eight players by minutes, including production, efficiency, minutes, and an approximate PER-style impact score

## Modeling Strategy

We compare four models:

1. Logistic Regression
2. Random Forest
3. MLP Baseline
4. Attention Model

The Attention Model is the main deep learning contribution in our project. Instead of treating all player inputs as one flat block, it learns a weight for each top-eight rotation slot and uses those weights to build a team-level representation. This lets us inspect which parts of the rotation the model emphasized. Even when it does not beat Random Forest on accuracy, it still gives us a useful view into how the model is using roster structure.

## Evaluation Protocol

We use leave-one-season-out cross-validation. In each fold, we hold out one full NBA season for testing and train on all remaining seasons. This setup is stricter than a random row split because teams from the same season can share league-wide context, schedule effects, and other hidden patterns. Holding out full seasons gives us a better test of how the model might behave on a future season.

We generate several outputs:

- Out-of-fold accuracy
- Macro F1
- Top-2 accuracy
- Per-class classification reports
- Confusion matrices
- Attention-weight diagnostics
- t-SNE visualizations of team-season features
- A simple market-size error audit

## Main Results

### Full-Season Results

| Model | Accuracy (OOF) | Accuracy (Fold Mean +/- SD) | Macro F1 (OOF) | Top-2 Accuracy (OOF) |
| --- | --- | --- | --- | --- |
| Logistic Regression | 60.9% | 60.9% +/- 7.6% | 38.0% | 81.4% |
| Random Forest | 68.0% | 68.0% +/- 6.1% | 36.2% | 84.3% |
| MLP Baseline | 62.8% | 62.8% +/- 6.6% | 40.8% | 81.7% |
| Attention Model | 65.3% | 65.4% +/- 7.6% | 39.3% | 82.5% |

### Key Findings

- Random Forest had the strongest overall accuracy.
- The MLP baseline had the strongest macro F1, which means it handled the minority classes a little better overall.
- Our Attention Model was close to the best models while also producing interpretable player-slot weights.
- Champion and Finals Loss were the hardest labels because each appears only once per season.
- Top-2 accuracy was above 81% for every model, so the models often ranked the right outcome near the top even when the exact class was missed.

### Mid-Season Results

We also ran the same modeling setup using only statistics available around the All-Star break or mid-season cutoff. As expected, performance dropped because the models could not see the rest of the regular season. The results still showed useful signal.

| Model | Accuracy (OOF) | Accuracy (Fold Mean +/- SD) | Macro F1 (OOF) | Top-2 Accuracy (OOF) |
| --- | --- | --- | --- | --- |
| Logistic Regression | 54.8% | 54.8% +/- 7.7% | 32.4% | 74.1% |
| Random Forest | 60.4% | 60.4% +/- 4.2% | 28.6% | 81.9% |
| MLP Baseline | 57.9% | 57.9% +/- 8.1% | 32.6% | 77.6% |
| Attention Model | 60.4% | 60.4% +/- 9.4% | 37.1% | 80.1% |

The Attention Model had the best mid-season macro F1 at 37.1%, while Random Forest tied it for best exact accuracy at 60.4% and had the best top-2 accuracy at 81.9%. The attention model also had the smallest macro-F1 drop from full-season to mid-season, falling from 39.3% to 37.1%.

### Attention Model Per-Class Results

The full-season Attention Model performed very well on missed-playoff teams and reasonably on first-round exits, but it struggled with the deepest playoff classes. This is where the class imbalance shows up most clearly.

| Class | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| Missed Playoffs | 0.908 | 0.904 | 0.906 | 293 |
| First Round Exit | 0.631 | 0.601 | 0.616 | 168 |
| Second Round Exit | 0.372 | 0.345 | 0.358 | 84 |
| Conference Finals | 0.204 | 0.238 | 0.220 | 42 |
| Finals Loss | 0.000 | 0.000 | 0.000 | 21 |
| Champion | 0.240 | 0.286 | 0.261 | 21 |

The MLP baseline, which had the best full-season macro F1, did slightly better on the rare Finals classes: F1 0.105 for Finals Loss and F1 0.316 for Champion. That difference is one reason we report macro F1 instead of only reporting accuracy.

### Market-Size Audit Results

We included a simple fairness-style audit by grouping teams into large-market and small-market buckets, then comparing average absolute prediction error. This is only a sanity check, not a complete fairness study.

| Setting | Large-Market Error | Small-Market Error | Error Gap |
| --- | ---: | ---: | ---: |
| Full season | 0.533 | 0.475 | 0.058 |
| Mid-season | 0.710 | 0.556 | 0.153 |

The gap was small in the full-season setting and larger in the mid-season setting. We read this cautiously because market size is only a rough proxy, and playoff success itself is not evenly distributed across markets.

## Interpretation and Research Value

Our results show that regular-season and rotation-level data contain real signal, but the problem is still noisy. A team can look strong in the regular season and still run into injuries, bad matchups, roster changes, or playoff-specific adjustments. That makes exact postseason-depth prediction difficult, especially for the rare Finals and Champion classes.

The attention model gave our project a useful interpretability layer. It did not prove causal player value, and we do not treat the weights as player rankings. Still, the weights helped us see how the model distributed focus across the top-eight rotation slots. In a project like this, that matters because we care not only about whether the model is right, but also about what kind of basketball information it appears to use.

## Repository Structure

```text
.
|-- data/
|   |-- processed/
|   |   `-- features_research.csv
|   `-- raw/
|       |-- player_stats_all_seasons.csv
|       |-- team_advanced_stats_all_seasons.csv
|       `-- team_stats_all_seasons.csv
|-- docs/
|   |-- figures/
|   |-- final_report.md
|   `-- nba_championship_prediction_proposal.pdf
|-- notebooks/
|   `-- championship_prediction_experiments.ipynb
|-- results/
|   `-- research_study/
|-- src/
|   `-- nba_championship_prediction/
|       |-- cli.py
|       |-- data_pipeline.py
|       |-- historical_labels.py
|       |-- interpretability.py
|       |-- modeling.py
|       |-- research_study.py
|       `-- training.py
|-- requirements.txt
|-- run_pipeline.py
`-- run_research_study.py
```

### Important Modules

- `data_pipeline.py`: data ingestion, caching, feature engineering, and dataset preparation
- `historical_labels.py`: curated postseason label definitions
- `modeling.py`: neural model definitions, including the attention-based architecture
- `training.py`: training utilities and cross-validation helpers
- `interpretability.py`: attention and embedding visualizations
- `research_study.py`: full experiment runner for metrics, figures, and evaluation artifacts

## Quick Start

### One-Click Demo

Windows PowerShell:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\run_project.ps1 demo
```

Presentation showcase with narration, testing, evaluation, and prediction examples:

```powershell
.\run_project.ps1 showcase
```

One-command version for execution-policy-restricted Windows terminals:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\run_project.ps1 showcase
```

Manual step-through mode:

```powershell
.\run_project.ps1 showcase -Pause
```

No-delay run:

```powershell
.\run_project.ps1 showcase -NoPause
```

Syntax check for execution-policy-restricted hosts:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -Command "Get-Command .\run_project.ps1 -Syntax"
```

### 1. Install Dependencies

```bash
python -m pip install -r requirements.txt
```

### 2. Run the Package Pipeline

This command runs a lightweight end-to-end version of the package workflow:

```bash
python run_pipeline.py --mode full --seasons 5 --epochs 10
```

### 3. Reproduce the Full Research Study

This command regenerates the main research metrics and artifacts:

```bash
python run_research_study.py --epochs-mlp 150 --epochs-attention 200 --lr 0.001 --seed 42
```

## Mid-Season Prediction Experiment

Our project includes two forecasting settings.

1. Full regular-season prediction uses complete regular-season team and rotation-player statistics.
2. Mid-season prediction uses only statistics available around the All-Star break or mid-season cutoff, then predicts the final playoff outcome.

Commands:

```powershell
.\run_project.ps1 midseason
.\run_project.ps1 midseason-final
```

The mid-season task is harder because the model cannot see late-season changes, trades, injuries, final standings, or playoff seeding effects. The full mid-season result table is included in the results section above.

## Precomputed Artifacts

We include the main outputs from completed real-data runs so reviewers can inspect our results right away.

- Final report: `docs/final_report.md`
- Figures: `docs/figures/`
- Structured metrics and predictions: `results/research_study/`
- Processed feature matrix: `data/processed/features_research.csv`
- Mid-season feature matrix: `data/processed/features_midseason.csv`
- Mid-season report: `docs/midseason_report.md`
- Mid-season figures: `docs/figures_midseason/`
- Mid-season metrics and predictions: `results/midseason_study/`
- Cached raw source tables: `data/raw/`

Additional presentation figures can be regenerated from the committed real-data artifacts:

```powershell
.\.venv\Scripts\python.exe scripts\generate_additional_figures.py
```

## Reproducibility Notes

- The committed report and metrics come from the real historical dataset.
- The package includes a mock-data fallback for local testing when the NBA data path is unavailable.
- For research reproduction, use the real-data path and committed caches rather than the mock fallback.
- The main written version of our study is `docs/final_report.md`.

## Limitations

Our project has several limitations.

First, the class distribution is very imbalanced. Almost half of the dataset is made up of teams that missed the playoffs, while each season has only one champion and one Finals loser. That makes rare-class prediction difficult even when the overall accuracy looks solid.

Second, the dataset is small for a six-class deep learning problem. We have 629 team-seasons across 21 seasons, which is useful for a course project but still limited compared with the amount of data neural models usually need.

Third, our features do not include every basketball factor that affects playoff outcomes. Injuries, matchups, coaching adjustments, trades, rest patterns, and playoff-series context can all change a team's path. We capture team strength and rotation structure, but not the full story.

Fourth, the full-season setup uses complete regular-season statistics, so those results should be read as season-level forecasting rather than true early prediction. The mid-season experiment is closer to an early forecasting task, but it naturally loses information.

Finally, we kept hyperparameter tuning modest. That made the study easier to reproduce and reduced the risk of overfitting, but a larger version of this project could explore more tuning, richer player embeddings, calibrated probabilities, or matchup-aware playoff simulations.

## Recommended Reading Order

If you are reviewing the project quickly:

1. Start with this `README.md` for the overview.
2. Read `docs/final_report.md` for the full research narrative.
3. Inspect `results/research_study/model_comparison.csv` and `results/research_study/summary_metrics.json` for the quantitative outputs.
4. Review `src/nba_championship_prediction/` for the engineering implementation.

## Summary

We built this project around a direct basketball question: how much can regular-season team and rotation data tell us about playoff success? Our results show that the signal is real, but not clean enough to make champion prediction easy. Random Forest was the strongest accuracy baseline, the MLP baseline was best on macro F1, and our Attention Model gave us competitive performance with a clearer view of how the model used the rotation.

That is the main takeaway from our project. Deep learning was useful, especially for interpretation, but the simpler models still mattered. The final system is not a perfect playoff predictor. It is a reproducible study that shows the strengths, limits, and tradeoffs of using machine learning for NBA postseason-depth forecasting.
