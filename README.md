# NBA Championship Prediction Using Deep Learning

Implementation of the project proposal for CSCI 5931 Deep Learning, Spring 2026.

**Team Members:** Shrujal Mogadati, Pranav Kaliaperumal, Disha Srinivasa

---

## Project Overview

This project predicts NBA championship contenders from mid-season (All-Star break) team snapshots using deep learning. Given team and player statistics at the halfway point of the season (~50 games), the model classifies each team into one of six playoff outcomes:

| Class | Outcome |
|-------|---------|
| 0 | Missed Playoffs |
| 1 | First Round Exit |
| 2 | Second Round Exit |
| 3 | Conference Finals |
| 4 | Finals Appearance |
| 5 | Champion |

---

## Repository Structure

```
.
├── models.py              # Neural network architectures (AttentionModel, MLPBaseline)
├── train.py               # Training pipeline with cross-validation
├── data_collection.py     # Data collection from nba_api
├── interpretability.py    # Attention visualization and t-SNE analysis
├── main.py                # Main entry point for full pipeline
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

---

## Model Architecture

### AttentionModel (Main Model)

As described in the proposal, the model has three main components:

1. **Player Embedding Network**: Shared subnetwork that compresses each player's stats into a compact embedding
2. **Attention Pooling**: Learns which players matter most for championship contention
3. **Classifier**: Combines team stats with the attention-pooled player summary for final prediction

```
Player 1 stats → [Embedding Net] → emb1 \
Player 2 stats → [Embedding Net] → emb2  |→ [Attention] → team_player_summary
...                                        |                        |
Player 8 stats → [Embedding Net] → emb8 /                         v
                                                                   
Team stats ------------------------------------------------> [Combine] → [FC Layers] → Prediction
```

### MLPBaseline (Comparison Baseline)

A plain neural network without player embeddings, used as a baseline for comparison.

---

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd nba-championship-prediction

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

- torch >= 2.0.0
- numpy >= 1.24.0
- pandas >= 2.0.0
- scikit-learn >= 1.3.0
- matplotlib >= 3.7.0
- nba_api >= 1.4.0 (optional, for real data collection)

---

## Usage

### Full Pipeline

Run the complete pipeline (data collection → training → evaluation → interpretability):

```bash
python main.py --mode full --seasons 20 --epochs 50
```

### Individual Steps

**1. Data Collection only:**
```bash
python main.py --mode data --seasons 20
```

**2. Training only:**
```bash
python main.py --mode train --epochs 50 --lr 0.001
```

**3. Cross-Validation Evaluation:**
```bash
python main.py --mode evaluate
```

**4. Interpretability Analysis:**
```bash
python main.py --mode interpret
```

### Command-Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--mode` | `full` | Pipeline mode: full, data, train, evaluate, interpret |
| `--seasons` | 20 | Number of seasons to include |
| `--epochs` | 50 | Training epochs |
| `--batch_size` | 32 | Batch size |
| `--lr` | 0.001 | Learning rate |
| `--embedding_size` | 32 | Player embedding dimension |
| `--hidden_size` | 128 | Hidden layer size |
| `--dropout` | 0.3 | Dropout rate |
| `--output_dir` | `outputs` | Output directory |

---

## Features

### Team-Level Features (~15)

- Wins, losses, win percentage
- Offensive/defensive/net rating
- Pace
- Effective field goal percentage (eFG%)
- True shooting percentage (TS%)
- Offensive/defensive rebound percentage
- Assist ratio
- Turnover percentage
- Strength of schedule

### Player-Level Features (~10 per player)

- Points, rebounds, assists
- Field goal percentage (FG%)
- Three-point percentage (3P%)
- Free throw percentage (FT%)
- Minutes played
- Plus/minus
- Player efficiency (fantasy points)
- Effective field goal percentage

### Contextual Features

- Roster continuity
- Conference strength

---

## Training

### Loss Function

Cross-entropy loss with class weights to handle class imbalance (only 1 champion per year vs many teams missing playoffs).

### Evaluation

**Leave-one-season-out cross-validation**: The model is trained on all seasons except one, then tested on the held-out season. This ensures the model never sees "future" data.

### Metrics

- **Accuracy**: Overall classification accuracy
- **F1 Score**: Weighted F1 score for imbalanced classes
- **Top-2 Accuracy**: Did the model rank the actual champion in its top 2 predictions?

---

## Interpretability

### Attention Weight Visualization

Shows which player positions the model considers most important for each playoff outcome. For example, does the model learn that "star players" matter more for championship contention?

### t-SNE Embedding Visualization

Projects the learned team embeddings into 2D space to visualize whether championship teams cluster together.

### Champion Clustering Analysis

Quantifies whether champions form a distinct cluster in the embedding space by comparing:
- Average distance between champions
- Average distance from champions to non-champions

---

## Expected Results

Based on related work:
- **Accuracy**: 60-70% (6-class classification is challenging)
- **Top-2 Accuracy**: 80-90% (model should reliably identify top contenders)
- **Attention Model should outperform MLP Baseline** by capturing player importance

---

## Ethical Considerations

As noted in the proposal:
- The model uses only performance-based features (stats, records, ratings)
- No demographic or personal information about players
- Analysis includes checking for bias toward large-market vs small-market teams

---

## References

1. Khanmohammadi et al. (2024). MambaNet: A Hybrid Neural Network for Predicting the NBA Playoffs. SN Computer Science.
2. Zhao et al. (2023). Enhancing Basketball Game Outcome Prediction through Fused Graph Convolutional Networks. Entropy.
3. Guan et al. (2023). NBA2Vec: Dense Feature Representations of NBA Players. arXiv.
4. Ouyang et al. (2024). Integration of Machine Learning XGBoost and SHAP Models for NBA Game Outcome Prediction. PLoS ONE.

---

## License

This project is for academic purposes (CSCI 5931 Deep Learning, Spring 2026).
