# NBA Championship Prediction with Deep Learning

This repository studies whether NBA postseason outcomes can be predicted from regular-season team statistics, contextual roster features, and the top eight players in each team's rotation. The codebase has been reorganized into a coherent `src/` package, the historical playoff labels are now versioned in code, and the repo includes a research-grade results report generated from a real multi-season experiment.

## Current Status

- Real-data experiment completed on 21 seasons from 2003-04 through 2023-24.
- Curated playoff labels are bundled in code.
- A full report is available at `docs/final_report.md`.
- Structured outputs are available under `results/research_study/`.
- The processed feature matrix is saved at `data/processed/features_research.csv`.

## High-Level Findings

From the current leave-one-season-out cross-validation run over 629 team-seasons:

- Random Forest achieved the best overall accuracy: 68.0%
- MLP Baseline achieved the best macro F1: 40.8%
- Attention Model achieved 65.3% accuracy and 39.3% macro F1
- Top-2 accuracy was above 81% for every model and peaked at 84.3% for Random Forest

The attention model remains valuable because it is competitive while also exposing player-importance weights that support interpretability.

## Repository Layout

```text
.
|-- data/
|   `-- processed/
|       `-- features_research.csv
|-- docs/
|   |-- final_report.md
|   |-- figures/
|   `-- nba_championship_prediction_proposal.pdf
|-- notebooks/
|   `-- championship_prediction_experiments.ipynb
|-- results/
|   `-- research_study/
|-- src/
|   `-- nba_championship_prediction/
|       |-- __init__.py
|       |-- cli.py
|       |-- data_pipeline.py
|       |-- historical_labels.py
|       |-- interpretability.py
|       |-- modeling.py
|       |-- research_study.py
|       `-- training.py
|-- .gitignore
|-- requirements.txt
|-- run_pipeline.py
`-- run_research_study.py
```

## Research Workflow

### Data and labels

The repo now combines:

- `nba_api` regular-season team and player statistics
- advanced team efficiency metrics from `TeamEstimatedMetrics`
- curated postseason outcomes for every season from 2003-04 through 2023-24
- contextual features including conference strength and roster continuity

### Models

The study compares four models:

1. Logistic Regression
2. Random Forest
3. MLP Baseline
4. Attention Model

### Evaluation

The main experiment uses leave-one-season-out cross-validation, which tests each model on one held-out season at a time and aggregates the out-of-fold predictions across the full historical sample.

## Running the Code

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Run the lightweight package pipeline:

```bash
python run_pipeline.py --mode full --seasons 5 --epochs 10
```

Run the full research study and regenerate the report:

```bash
python run_research_study.py --epochs-mlp 150 --epochs-attention 200 --lr 0.001 --seed 42
```

## Main Artifacts

- Report: `docs/final_report.md`
- Figures: `docs/figures/`
- Metrics and tables: `results/research_study/`
- Processed dataset: `data/processed/features_research.csv`


## Recommendation

Use `README.md` as the repo overview and `docs/final_report.md` as the canonical research narrative. The notebook should now be treated as exploratory support material rather than the primary project deliverable.
