# NBA Championship Outcome Prediction with Deep Learning

Research-grade machine learning study on forecasting NBA playoff depth from regular-season team performance, roster context, and rotation-level player statistics.

This repository packages the project as both a reproducible research artifact and a professional end-to-end machine learning case study. It combines data collection, feature engineering, classical baselines, deep learning models, interpretability analysis, and report generation in a structured `src/` package.

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

The project investigates whether an attention-based model that explicitly learns which rotation players matter most can improve postseason forecasting relative to simpler baselines. The study uses real NBA data, engineered team and player features, and a season-aware validation protocol designed to better reflect out-of-sample generalization.

The main result is nuanced and realistic: the problem is learnable, but difficult. Random Forest delivers the strongest overall accuracy, the MLP baseline achieves the best macro F1, and the attention model remains competitive while providing interpretable player-importance weights. That tradeoff between predictive performance and interpretability is a central takeaway of the project.

## Why This Project Is Strong Portfolio Material

- It addresses a clear research question with a measurable evaluation protocol.
- It moves beyond a notebook-only workflow into a reusable Python package with CLI entry points.
- It demonstrates practical feature engineering on structured sports analytics data.
- It compares classical ML and deep learning models instead of assuming a neural approach is automatically best.
- It includes interpretability outputs, error analysis, and a lightweight fairness sanity check.
- It ships with precomputed artifacts, figures, and a written final report, making the work easy to review.

## Research Question

Can a model that explicitly learns which rotation players matter most outperform simpler baselines when forecasting a team's eventual playoff depth from regular-season statistics?

## Dataset and Label Design

The unit of analysis is one team-season. Each row represents a single NBA team in a single season and is labeled with its final playoff outcome.

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

- `nba_api` regular-season team statistics via `LeagueDashTeamStats`
- advanced team metrics via `TeamEstimatedMetrics`
- player statistics via `LeagueDashPlayerStats`
- curated historical playoff labels versioned in code

### Engineered Features

- Team-level performance variables such as win percentage, scoring, rebounding, playmaking, shooting efficiency, plus/minus, and wins/losses
- Advanced efficiency metrics including offensive, defensive, and net rating
- Context features including conference-strength proxy and roster continuity
- Rotation-level player features for the top eight players by minutes, including production, shooting efficiency, minutes, and an approximate PER-style impact score

## Modeling Strategy

The study compares four approaches:

1. Logistic Regression
2. Random Forest
3. MLP Baseline
4. Attention Model

The attention architecture is the most distinctive part of the project. Instead of flattening all player inputs equally, it learns how much weight to assign to each rotation slot when forming a team-level postseason prediction. This makes the model useful even when it does not dominate the best baseline on every metric, because it offers a direct view into which parts of the rotation appear most influential.

## Evaluation Protocol

The main experiment uses leave-one-season-out cross-validation. For each fold, one full NBA season is held out for testing and the model is trained on all remaining seasons. This is a stronger evaluation design than random row-level splitting because it reduces leakage across seasons and better reflects real forecasting behavior.

Additional outputs include:

- out-of-fold accuracy
- macro F1
- top-2 accuracy
- per-class classification reports
- confusion matrices
- attention-weight diagnostics
- t-SNE visualization of team-season feature structure
- a simple market-size error audit

## Main Results

| Model | Accuracy (OOF) | Accuracy (Fold Mean +/- SD) | Macro F1 (OOF) | Top-2 Accuracy (OOF) |
| --- | --- | --- | --- | --- |
| Logistic Regression | 60.9% | 60.9% +/- 7.6% | 38.0% | 81.4% |
| Random Forest | 68.0% | 68.0% +/- 6.1% | 36.2% | 84.3% |
| MLP Baseline | 62.8% | 62.8% +/- 6.6% | 40.8% | 81.7% |
| Attention Model | 65.3% | 65.4% +/- 7.6% | 39.3% | 82.5% |

### Key Findings

- Random Forest is the strongest model on overall accuracy.
- MLP Baseline is the strongest model on macro F1, suggesting better balance across minority classes.
- The Attention Model remains competitive while adding interpretability through learned player-importance weights.
- Champion and Finals-Loss prediction remain the hardest cases because those classes are extremely sparse.
- Top-2 accuracy exceeds 81% for every model, indicating the models often rank plausible postseason outcomes well even when the exact class is missed.

## Interpretation and Research Value

This project produces a realistic conclusion rather than an inflated one: deep learning is useful here, but not automatically superior. The attention model adds analytical value by exposing how different rotation slots contribute to predictions, while the best baseline remains a strong benchmark that more complex architectures must beat consistently.

That makes the repository valuable from both a research and recruiting perspective. It shows the ability to:

- define a nontrivial prediction problem clearly
- build a reproducible feature pipeline
- compare multiple modeling paradigms fairly
- evaluate models with appropriate validation design
- communicate limitations honestly
- package the work in a way that another reviewer can run and inspect

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
- `modeling.py`: neural model definitions including the attention-based architecture
- `training.py`: training utilities and cross-validation helpers
- `interpretability.py`: attention and embedding visualizations
- `research_study.py`: full experiment runner that generates the report, figures, and evaluation artifacts

## Quick Start

## One-Click Demo

Windows PowerShell:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\run_project.ps1 demo
```

Presentation showcase with built-in narration, testing, evaluation, and prediction examples:

```powershell
.\run_project.ps1 showcase
```

One-command version for execution-policy-restricted Windows terminals:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\run_project.ps1 showcase
```

Optional manual step-through mode if you want to pause between sections:

```powershell
.\run_project.ps1 showcase -Pause
```

Optional no-delay run:

```powershell
.\run_project.ps1 showcase -NoPause
```

Optional syntax check on execution-policy-restricted hosts:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -Command "Get-Command .\run_project.ps1 -Syntax"
```

### 1. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 2. Run the package pipeline

This is the lightweight end-to-end package entry point:

```bash
python run_pipeline.py --mode full --seasons 5 --epochs 10
```

### 3. Reproduce the full research study

This command regenerates the report and research artifacts:

```bash
python run_research_study.py --epochs-mlp 150 --epochs-attention 200 --lr 0.001 --seed 42
```

## Mid-Season Prediction Experiment

This project includes two forecasting settings:

1. Full regular-season prediction:
   Uses full regular-season team and rotation-player statistics.

2. Mid-season prediction:
   Uses only statistics available around the All-Star break / mid-season cutoff to predict each team's final playoff outcome.

Commands:

```powershell
.\run_project.ps1 midseason
.\run_project.ps1 midseason-final
```

## Precomputed Artifacts

The repository already includes the main outputs from a completed real-data run, so reviewers do not need to rerun the full study to inspect the work.

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

- The committed report and metrics were generated from the real historical dataset.
- The package includes a fallback mock-data path for local testing when the NBA data path is unavailable.
- For research reproduction, use the real-data path and the committed caches rather than the mock fallback.
- The canonical narrative write-up lives in `docs/final_report.md`.

## Limitations

1. The class distribution is highly imbalanced, with only one champion and one finals loser per season.
2. The sample size is still modest for a six-class forecasting problem.
3. The current framing uses full regular-season statistics, so results should be interpreted as season-level forecasting rather than a strict mid-season prediction task.
4. Contextual features are useful but still relatively simple compared with a production sports analytics system.
5. Hyperparameter search was intentionally lightweight to keep the study reproducible and avoid overfitting.

## Recommended Reading Order

If you are reviewing the project quickly:

1. Start with this `README.md` for the overview.
2. Read `docs/final_report.md` for the full research narrative.
3. Inspect `results/research_study/model_comparison.csv` and `results/research_study/summary_metrics.json` for the quantitative outputs.
4. Review `src/nba_championship_prediction/` for the engineering implementation.

## Summary

This repository presents a polished end-to-end machine learning project built around a concrete sports analytics question: how well can regular-season team and rotation data forecast playoff success? The answer is encouraging but appropriately cautious. The predictive signal is real, the attention model is interpretable and competitive, and the strongest baseline remains a meaningful benchmark. Together, those elements make the project both academically credible and professionally compelling.
