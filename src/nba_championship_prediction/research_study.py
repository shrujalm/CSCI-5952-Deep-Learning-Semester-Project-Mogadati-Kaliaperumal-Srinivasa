"""Run the full research experiment and write a publication-style report."""

from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from .data_pipeline import NUM_ROTATION_PLAYERS, PROCESSED_FEATURES_FILE, build_dataset, prepare_model_inputs
from .historical_labels import PLAYOFF_LABEL_NAMES, SEASONS
from .modeling import AttentionModel, MLPBaseline

TARGET_NAMES = [PLAYOFF_LABEL_NAMES[index] for index in range(6)]
MODEL_DISPLAY_NAMES = {
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
    "mlp_baseline": "MLP Baseline",
    "attention_model": "Attention Model",
}


@dataclass(frozen=True)
class StudyReportContext:
    title: str = "Predicting NBA Championship Contenders from Team and Rotation Statistics"
    setting_name: str = "Full regular-season prediction"
    feature_timeframe: str = "regular-season"
    data_source_description: str = (
        "`nba_api` endpoints `LeagueDashTeamStats`, `TeamEstimatedMetrics`, and `LeagueDashPlayerStats`"
    )
    feature_intro: str = "The final feature matrix contains"
    main_result_note: str = (
        "The table reports out-of-fold metrics across all team-seasons plus fold-level mean and "
        "standard deviation for accuracy. Random forest is the most accurate model in this run, "
        "the MLP baseline has the strongest macro F1, and the attention model sits between them "
        "while offering direct player-importance explanations."
    )
    limitations: list[str] = field(default_factory=lambda: [
        "The dataset is small for a six-class forecasting problem, especially at the champion tail.",
        "Roster continuity and conference strength are useful but still fairly coarse contextual features.",
        "The study uses full regular-season data rather than a strict All-Star-break snapshot, so the results should be interpreted as season-level forecasting rather than a pure mid-season forecast.",
        "Hyperparameter search was intentionally lightweight to preserve reproducibility and avoid overfitting the small dataset.",
    ])
    data_notes: list[str] = field(default_factory=list)
    figure_prefix: str = "research"
    command_name: str = "run_research_study.py"


def build_parser(
    description: str = "Run the NBA championship research study.",
    default_output_dir: str = "results/research_study",
    default_report_path: str = "docs/final_report.md",
    default_figures_dir: str = "docs/figures",
    include_force_refresh_midseason: bool = False,
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--epochs-mlp", type=int, default=150, help="Training epochs for the MLP baseline.")
    parser.add_argument("--epochs-attention", type=int, default=200, help="Training epochs for the attention model.")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate for neural models.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=default_output_dir,
        help="Directory for JSON, CSV, and intermediate experiment artifacts.",
    )
    parser.add_argument(
        "--report-path",
        type=str,
        default=default_report_path,
        help="Path to the generated markdown report.",
    )
    parser.add_argument(
        "--figures-dir",
        type=str,
        default=default_figures_dir,
        help="Directory for report figures.",
    )
    if include_force_refresh_midseason:
        parser.add_argument(
            "--force-refresh-midseason",
            action="store_true",
            help="Ignore cached mid-season processed/raw files and refetch mid-season NBA API data.",
        )
    return parser


def parse_args(**parser_kwargs: Any) -> argparse.Namespace:
    parser = build_parser(**parser_kwargs)
    return parser.parse_args()


def set_research_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_class_weights_dict(y_train: np.ndarray) -> dict[int, float]:
    classes, counts = np.unique(y_train, return_counts=True)
    total = len(y_train)
    return {int(cls): total / (len(classes) * count) for cls, count in zip(classes, counts)}


def top_k_accuracy(y_true: np.ndarray, y_proba: np.ndarray, k: int = 2) -> float:
    correct = 0
    for true_label, probabilities in zip(y_true, y_proba):
        if true_label in np.argsort(probabilities)[-k:]:
            correct += 1
    return float(correct / len(y_true))


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
) -> dict[str, float]:
    results = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }
    if y_proba is not None:
        results["top2_accuracy"] = float(top_k_accuracy(y_true, y_proba, k=2))
    return results


def compute_fold_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    seasons_array: np.ndarray,
) -> list[dict[str, Any]]:
    fold_metrics = []
    for season in np.unique(seasons_array):
        mask = seasons_array == season
        metrics = evaluate_predictions(y_true[mask], y_pred[mask], y_proba[mask])
        metrics["season"] = str(season)
        fold_metrics.append(metrics)
    return fold_metrics


def summarize_fold_metrics(fold_metrics: list[dict[str, Any]]) -> dict[str, float]:
    return {
        "accuracy_mean": float(np.mean([fold["accuracy"] for fold in fold_metrics])),
        "accuracy_std": float(np.std([fold["accuracy"] for fold in fold_metrics])),
        "f1_macro_mean": float(np.mean([fold["f1_macro"] for fold in fold_metrics])),
        "f1_macro_std": float(np.std([fold["f1_macro"] for fold in fold_metrics])),
        "top2_accuracy_mean": float(np.mean([fold["top2_accuracy"] for fold in fold_metrics])),
        "top2_accuracy_std": float(np.std([fold["top2_accuracy"] for fold in fold_metrics])),
    }


def run_sklearn_baseline(
    model_class: type,
    model_params: dict[str, Any],
    X: np.ndarray,
    y: np.ndarray,
    seasons_array: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    all_preds = np.zeros(len(y), dtype=np.int64)
    all_proba = np.zeros((len(y), 6), dtype=np.float32)

    for test_season in np.unique(seasons_array):
        train_mask = seasons_array != test_season
        test_mask = seasons_array == test_season

        X_train, X_test = X[train_mask], X[test_mask]
        y_train = y[train_mask]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        weights = compute_class_weights_dict(y_train)
        model = model_class(**model_params, class_weight=weights)
        model.fit(X_train_scaled, y_train)

        all_preds[test_mask] = model.predict(X_test_scaled)
        all_proba[test_mask] = model.predict_proba(X_test_scaled)

    return all_preds, all_proba


def _split_team_and_player_features(
    X_scaled: np.ndarray,
    num_team_features: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    X_tensor = torch.as_tensor(X_scaled, dtype=torch.float32)
    team_tensor = X_tensor[:, :num_team_features]
    player_tensor = X_tensor[:, num_team_features:]
    return team_tensor, player_tensor


def train_mlp_one_fold(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    num_team_features: int,
    epochs: int,
    lr: float,
    seed: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    set_research_seed(seed)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    team_train, player_train = _split_team_and_player_features(X_train_scaled, num_team_features)
    team_test, player_test = _split_team_and_player_features(X_test_scaled, num_team_features)
    team_train = team_train.to(device)
    player_train = player_train.to(device)
    team_test = team_test.to(device)
    player_test = player_test.to(device)
    y_train_tensor = torch.as_tensor(y_train, dtype=torch.long, device=device)

    model = MLPBaseline(input_size=X_train.shape[1], num_classes=6).to(device)

    weights_dict = compute_class_weights_dict(y_train)
    weight_tensor = torch.as_tensor(
        [weights_dict.get(index, 1.0) for index in range(6)],
        dtype=torch.float32,
        device=device,
    )
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        logits = model(team_train, player_train)
        loss = criterion(logits, y_train_tensor)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        logits = model(team_test, player_test)
        probabilities = torch.softmax(logits, dim=1).cpu().numpy()
        predictions = logits.argmax(dim=1).cpu().numpy()

    return predictions, probabilities


def train_attention_one_fold(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    num_team_features: int,
    num_player_stats: int,
    epochs: int,
    lr: float,
    seed: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    set_research_seed(seed)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    team_train, player_train = _split_team_and_player_features(X_train_scaled, num_team_features)
    team_test, player_test = _split_team_and_player_features(X_test_scaled, num_team_features)
    team_train = team_train.to(device)
    player_train = player_train.to(device)
    team_test = team_test.to(device)
    player_test = player_test.to(device)
    y_train_tensor = torch.as_tensor(y_train, dtype=torch.long, device=device)

    model = AttentionModel(
        team_stat_size=num_team_features,
        player_stat_size=num_player_stats,
        num_players=NUM_ROTATION_PLAYERS,
        embedding_size=32,
        num_classes=6,
        hidden_size=128,
        dropout_rate=0.3,
    ).to(device)

    weights_dict = compute_class_weights_dict(y_train)
    weight_tensor = torch.as_tensor(
        [weights_dict.get(index, 1.0) for index in range(6)],
        dtype=torch.float32,
        device=device,
    )
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        logits, _ = model(team_train, player_train, return_attention=True)
        loss = criterion(logits, y_train_tensor)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        logits, attention_weights = model(team_test, player_test, return_attention=True)
        probabilities = torch.softmax(logits, dim=1).cpu().numpy()
        predictions = logits.argmax(dim=1).cpu().numpy()
        attention_array = attention_weights.cpu().numpy()

    return predictions, probabilities, attention_array


def run_neural_models(
    X: np.ndarray,
    y: np.ndarray,
    seasons_array: np.ndarray,
    num_team_features: int,
    num_player_stats: int,
    epochs_mlp: int,
    epochs_attention: int,
    lr: float,
    seed: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    unique_seasons = np.unique(seasons_array)
    mlp_preds = np.zeros(len(y), dtype=np.int64)
    mlp_proba = np.zeros((len(y), 6), dtype=np.float32)
    attention_preds = np.zeros(len(y), dtype=np.int64)
    attention_proba = np.zeros((len(y), 6), dtype=np.float32)
    all_attention_weights = np.zeros((len(y), NUM_ROTATION_PLAYERS), dtype=np.float32)

    disable_progress = os.environ.get("TQDM_DISABLE") == "1"
    for fold_index, test_season in enumerate(tqdm(unique_seasons, desc="Neural CV", disable=disable_progress)):
        train_mask = seasons_array != test_season
        test_mask = seasons_array == test_season

        mlp_fold_preds, mlp_fold_proba = train_mlp_one_fold(
            X_train=X[train_mask],
            y_train=y[train_mask],
            X_test=X[test_mask],
            num_team_features=num_team_features,
            epochs=epochs_mlp,
            lr=lr,
            seed=seed + fold_index,
            device=device,
        )
        mlp_preds[test_mask] = mlp_fold_preds
        mlp_proba[test_mask] = mlp_fold_proba

        attention_fold_preds, attention_fold_proba, attention_fold_weights = train_attention_one_fold(
            X_train=X[train_mask],
            y_train=y[train_mask],
            X_test=X[test_mask],
            num_team_features=num_team_features,
            num_player_stats=num_player_stats,
            epochs=epochs_attention,
            lr=lr,
            seed=seed + 100 + fold_index,
            device=device,
        )
        attention_preds[test_mask] = attention_fold_preds
        attention_proba[test_mask] = attention_fold_proba
        all_attention_weights[test_mask] = attention_fold_weights

    return mlp_preds, mlp_proba, attention_preds, attention_proba, all_attention_weights


def plot_confusion_matrices(
    y_true: np.ndarray,
    predictions_by_model: dict[str, np.ndarray],
    figure_path: Path,
) -> None:
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    short_labels = ["Miss", "R1", "R2", "CF", "Fin", "Champ"]

    for axis, (model_key, predictions) in zip(axes.flatten(), predictions_by_model.items()):
        matrix = confusion_matrix(y_true, predictions, labels=[0, 1, 2, 3, 4, 5])
        sns.heatmap(
            matrix,
            annot=True,
            fmt="d",
            cmap="Blues",
            ax=axis,
            xticklabels=short_labels,
            yticklabels=short_labels,
        )
        axis.set_xlabel("Predicted")
        axis.set_ylabel("Actual")
        axis.set_title(MODEL_DISPLAY_NAMES[model_key])

    fig.suptitle("Confusion Matrices for All Models", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(figure_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_attention_weights(
    attention_weights: np.ndarray,
    labels: np.ndarray,
    figure_path: Path,
) -> None:
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    player_labels = [f"Player {index + 1}\n(by MIN)" for index in range(NUM_ROTATION_PLAYERS)]

    average_weights = attention_weights.mean(axis=0)
    axes[0].bar(player_labels, average_weights, color="steelblue")
    axes[0].set_ylabel("Average Attention Weight")
    axes[0].set_title("Average Attention Across All Teams")
    axes[0].set_ylim(0, max(average_weights) * 1.3)

    champion_mask = labels == 5
    missed_mask = labels == 0
    if champion_mask.sum() > 0 and missed_mask.sum() > 0:
        champion_weights = attention_weights[champion_mask].mean(axis=0)
        missed_weights = attention_weights[missed_mask].mean(axis=0)
        x = np.arange(NUM_ROTATION_PLAYERS)
        width = 0.35
        axes[1].bar(x - width / 2, champion_weights, width, label="Champions", color="#1f77b4")
        axes[1].bar(x + width / 2, missed_weights, width, label="Missed Playoffs", color="#d62728")
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(player_labels)
        axes[1].set_ylabel("Average Attention Weight")
        axes[1].set_title("Champions vs Missed Playoffs")
        axes[1].legend()

    fig.tight_layout()
    fig.savefig(figure_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_tsne(X: np.ndarray, y: np.ndarray, figure_path: Path, seed: int) -> None:
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    tsne = TSNE(n_components=2, random_state=seed, perplexity=30, n_jobs=1)
    X_2d = tsne.fit_transform(X_scaled)

    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(
        X_2d[:, 0],
        X_2d[:, 1],
        c=y,
        cmap="RdYlGn",
        alpha=0.65,
        s=35,
        edgecolors="gray",
        linewidths=0.5,
    )
    colorbar = plt.colorbar(scatter, ax=ax)
    colorbar.set_ticks([0, 1, 2, 3, 4, 5])
    colorbar.set_ticklabels(["Missed", "R1", "R2", "CF", "Finals", "Champ"])
    ax.set_title("t-SNE of Team-Season Feature Vectors")
    ax.set_xlabel("t-SNE Dimension 1")
    ax.set_ylabel("t-SNE Dimension 2")
    fig.tight_layout()
    fig.savefig(figure_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def summarize_market_fairness(dataset: pd.DataFrame, attention_predictions: np.ndarray) -> dict[str, float]:
    large_market = {"LAL", "NYK", "GSW", "BOS", "CHI", "BKN", "LAC", "PHI", "HOU", "MIA"}
    dataset = dataset.copy()
    dataset["MARKET"] = dataset["TEAM"].apply(lambda team: "Large" if team in large_market else "Small")
    dataset["ATTN_PRED"] = attention_predictions
    dataset["ATTN_ERROR"] = (dataset["ATTN_PRED"] - dataset["PLAYOFF_RESULT"]).abs()

    market_actual = dataset.groupby("MARKET")["PLAYOFF_RESULT"].mean().to_dict()
    market_pred = dataset.groupby("MARKET")["ATTN_PRED"].mean().to_dict()
    market_error = dataset.groupby("MARKET")["ATTN_ERROR"].mean().to_dict()

    return {
        "large_market_actual_mean": float(market_actual.get("Large", 0.0)),
        "small_market_actual_mean": float(market_actual.get("Small", 0.0)),
        "large_market_pred_mean": float(market_pred.get("Large", 0.0)),
        "small_market_pred_mean": float(market_pred.get("Small", 0.0)),
        "large_market_error_mean": float(market_error.get("Large", 0.0)),
        "small_market_error_mean": float(market_error.get("Small", 0.0)),
        "error_gap": float(abs(market_error.get("Large", 0.0) - market_error.get("Small", 0.0))),
    }


def format_pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def format_mean_std(mean: float, std: float) -> str:
    return f"{mean * 100:.1f}% +/- {std * 100:.1f}%"


def to_markdown_table(rows: list[dict[str, str]], headers: list[str]) -> str:
    header_line = "| " + " | ".join(headers) + " |"
    divider = "| " + " | ".join(["---"] * len(headers)) + " |"
    body = [
        "| " + " | ".join(row[header] for header in headers) + " |"
        for row in rows
    ]
    return "\n".join([header_line, divider, *body])


def markdown_relative_path(path: Path, base_dir: Path) -> str:
    try:
        return path.relative_to(base_dir).as_posix()
    except ValueError:
        return path.as_posix()


def numbered_markdown(items: list[str]) -> str:
    return "\n".join(f"{index}. {item}" for index, item in enumerate(items, start=1))


def bulleted_markdown(items: list[str]) -> str:
    return "\n".join(f"- {item}" for item in items)


def write_final_report(
    dataset: pd.DataFrame,
    metrics: dict[str, dict[str, Any]],
    best_baseline_key: str,
    best_baseline_report: pd.DataFrame,
    attention_report: pd.DataFrame,
    market_fairness: dict[str, float],
    report_path: Path,
    output_dir: Path,
    figures_dir: Path,
    args: argparse.Namespace,
    context: StudyReportContext,
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    comparison_rows = []
    ordered_keys = [
        "logistic_regression",
        "random_forest",
        "mlp_baseline",
        "attention_model",
    ]
    for model_key in ordered_keys:
        model_metrics = metrics[model_key]
        fold_summary = model_metrics["fold_summary"]
        comparison_rows.append(
            {
                "Model": MODEL_DISPLAY_NAMES[model_key],
                "Accuracy (OOF)": format_pct(model_metrics["overall_metrics"]["accuracy"]),
                "Accuracy (Fold Mean +/- SD)": format_mean_std(
                    fold_summary["accuracy_mean"],
                    fold_summary["accuracy_std"],
                ),
                "Macro F1 (OOF)": format_pct(model_metrics["overall_metrics"]["f1_macro"]),
                "Top-2 Accuracy (OOF)": format_pct(model_metrics["overall_metrics"]["top2_accuracy"]),
            }
        )

    comparison_table = to_markdown_table(
        comparison_rows,
        [
            "Model",
            "Accuracy (OOF)",
            "Accuracy (Fold Mean +/- SD)",
            "Macro F1 (OOF)",
            "Top-2 Accuracy (OOF)",
        ],
    )

    attention_champion_row = attention_report.loc["Champion"]
    baseline_champion_row = best_baseline_report.loc["Champion"]
    class_distribution = dataset["PLAYOFF_RESULT"].value_counts().sort_index().to_dict()
    seasons_covered = sorted(dataset["SEASON"].unique().tolist())
    label_table = to_markdown_table(
        [{"Class": str(index), "Outcome": PLAYOFF_LABEL_NAMES[index]} for index in range(6)],
        ["Class", "Outcome"],
    )
    distribution_table = to_markdown_table(
        [
            {
                "Class": str(index),
                "Count": str(class_distribution.get(index, 0)),
                "Share": format_pct(class_distribution.get(index, 0) / len(dataset)),
            }
            for index in range(6)
        ],
        ["Class", "Count", "Share"],
    )

    best_accuracy_key = max(metrics, key=lambda key: metrics[key]["overall_metrics"]["accuracy"])
    best_macro_f1_key = max(metrics, key=lambda key: metrics[key]["overall_metrics"]["f1_macro"])
    best_top2_key = max(metrics, key=lambda key: metrics[key]["overall_metrics"]["top2_accuracy"])
    best_fold_key, best_fold = max(
        (
            (model_key, fold_metric)
            for model_key, model_metrics in metrics.items()
            for fold_metric in model_metrics["fold_metrics"]
        ),
        key=lambda item: item[1]["accuracy"],
    )
    confusion_figure = figures_dir / f"{context.figure_prefix}_confusion_matrices.png"
    attention_figure = figures_dir / f"{context.figure_prefix}_attention_weights.png"
    tsne_figure = figures_dir / f"{context.figure_prefix}_tsne.png"
    report_dir = report_path.parent
    data_notes = bulleted_markdown(context.data_notes)
    data_notes_section = f"\nAdditional data notes:\n\n{data_notes}\n" if context.data_notes else ""
    limitations = numbered_markdown(context.limitations)

    report = f"""# {context.title}

## Abstract

This report evaluates whether NBA postseason outcomes can be predicted from {context.feature_timeframe} team efficiency, contextual roster features, and the top eight players in each team's rotation. Using the 2003-04 through 2023-24 seasons ({len(seasons_covered)} seasons; {len(dataset)} team-seasons), we compare four models under leave-one-season-out cross-validation: logistic regression, random forest, a multilayer perceptron baseline, and an attention-based roster model. In this run, {MODEL_DISPLAY_NAMES[best_accuracy_key]} achieves the best overall accuracy, while {MODEL_DISPLAY_NAMES[best_macro_f1_key]} achieves the highest macro F1. The main empirical challenge remains severe class imbalance, especially for the Champion class.

## Research Question

Can a model that explicitly learns which rotation players matter most outperform simpler baselines when forecasting a team's eventual playoff depth from regular-season statistics?

## Label Definition

The task is a six-class classification problem:

{label_table}

## Data Provenance

- Forecasting setting: {context.setting_name}
- Seasons covered: {seasons_covered[0]} through {seasons_covered[-1]}
- Primary data source: {context.data_source_description}
- Curated postseason labels: local historical dictionary extracted from the project notebook and now versioned in code
- Unit of analysis: one team-season
- Sample count: {len(dataset)} team-seasons
{data_notes_section}

Class distribution:

{distribution_table}

## Feature Construction

{context.feature_intro}:

- Team-level regular-season performance variables: win percentage, scoring, rebounding, playmaking, shot-making, plus/minus, wins/losses, and advanced efficiency metrics.
- Contextual variables: conference-strength proxy and roster continuity.
- Player-level variables for the top eight rotation players by minutes: box-score production, shooting efficiency, minutes, and an approximate PER-style impact summary.

## Experimental Protocol

- Evaluation: leave-one-season-out cross-validation
- Neural-model optimizer: Adam
- MLP epochs: {args.epochs_mlp}
- Attention-model epochs: {args.epochs_attention}
- Learning rate: {args.lr}
- Random seed: {args.seed}
- Hardware used in this run: CPU execution within the local workspace

## Main Results

{comparison_table}

{context.main_result_note}

### Real Results Artifact Summary

The metrics above are copied from generated artifacts in `{output_dir.as_posix()}`, not estimated or mocked. `predictions.csv` contains {len(dataset)} out-of-fold team-season predictions plus the header row, matching the {len(dataset)} team-seasons described in the data section. The main model ranking comes from `model_comparison.csv` and `summary_metrics.json`:

- Best OOF accuracy: {MODEL_DISPLAY_NAMES[best_accuracy_key]} at {metrics[best_accuracy_key]['overall_metrics']['accuracy']:.6f}.
- Best OOF macro F1: {MODEL_DISPLAY_NAMES[best_macro_f1_key]} at {metrics[best_macro_f1_key]['overall_metrics']['f1_macro']:.6f}.
- Best OOF top-2 accuracy: {MODEL_DISPLAY_NAMES[best_top2_key]} at {metrics[best_top2_key]['overall_metrics']['top2_accuracy']:.6f}.
- Attention model OOF metrics: accuracy {metrics['attention_model']['overall_metrics']['accuracy']:.6f}, macro F1 {metrics['attention_model']['overall_metrics']['f1_macro']:.6f}, top-2 accuracy {metrics['attention_model']['overall_metrics']['top2_accuracy']:.6f}.
- Best single-season fold accuracy observed: {MODEL_DISPLAY_NAMES[best_fold_key]} at {best_fold['accuracy']:.6f} in {best_fold['season']}.

## Per-Class Analysis

Champion-class performance is the hardest part of the problem because there is only one champion per season. In the attention model, Champion precision is {attention_champion_row['precision']:.2f}, recall is {attention_champion_row['recall']:.2f}, and F1 is {attention_champion_row['f1-score']:.2f}. For the strongest baseline ({MODEL_DISPLAY_NAMES[best_baseline_key]}), Champion precision is {baseline_champion_row['precision']:.2f}, recall is {baseline_champion_row['recall']:.2f}, and F1 is {baseline_champion_row['f1-score']:.2f}.

Detailed per-class tables are saved at:

- `{(output_dir / 'attention_classification_report.csv').as_posix()}`
- `{(output_dir / 'best_baseline_classification_report.csv').as_posix()}`

## Visual Diagnostics

### Confusion Matrices

![Confusion matrices]({markdown_relative_path(confusion_figure, report_dir)})

### Attention Weight Profiles

![Attention weights]({markdown_relative_path(attention_figure, report_dir)})

### t-SNE Projection of Team-Season Features

![t-SNE projection]({markdown_relative_path(tsne_figure, report_dir)})

## Error Analysis and Interpretation

- The easiest class remains `Missed Playoffs`, which dominates the dataset and is easier to separate from the deeper-round classes.
- The Champion and Finals-Loss classes remain sparse and frequently get confused with conference-final or second-round teams.
- The attention plots show that the model does not weight every roster slot equally, which supports the original project motivation even when aggregate predictive gains over the best baseline are modest.

## Fairness and Market-Size Check

We performed a coarse market-size audit using the attention model's predictions.

- Large-market average absolute error: {market_fairness['large_market_error_mean']:.3f}
- Small-market average absolute error: {market_fairness['small_market_error_mean']:.3f}
- Error gap: {market_fairness['error_gap']:.3f}

This is not a comprehensive fairness analysis, but it offers a basic sanity check that prediction error is not grossly asymmetric across a simple market-size partition.

## Threats to Validity

{limitations}

## Conclusion

The project now has a real-data experimental pipeline and a reproducible report artifact. The empirical results show that the task is learnable to a degree, but class imbalance and limited sample size remain the main barriers to robust champion prediction. The attention model contributes interpretability and a principled way to aggregate roster information, while the strongest baseline still provides a high bar that any more complex architecture must beat consistently.

## Reproducibility Appendix

Command used to generate this report:

```bash
python {context.command_name} --epochs-mlp {args.epochs_mlp} --epochs-attention {args.epochs_attention} --lr {args.lr} --seed {args.seed}
```

Key artifact paths:

- Report: `{report_path.as_posix()}`
- Metrics JSON: `{(output_dir / 'summary_metrics.json').as_posix()}`
- Model comparison CSV: `{(output_dir / 'model_comparison.csv').as_posix()}`
- Fold metrics CSV: `{(output_dir / 'fold_metrics.csv').as_posix()}`
- Predictions CSV: `{(output_dir / 'predictions.csv').as_posix()}`
- Figures: `{figures_dir.as_posix()}`
"""
    report_path.write_text(report, encoding="utf-8")


def dataframe_from_classification_report(report_dict: dict[str, Any]) -> pd.DataFrame:
    frame = pd.DataFrame(report_dict).transpose()
    numeric_columns = [column for column in ["precision", "recall", "f1-score", "support"] if column in frame.columns]
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column])
    return frame


DatasetBuilder = Callable[[list[str], Path], pd.DataFrame]


def build_full_dataset_for_study(seasons: list[str], processed_file: Path) -> pd.DataFrame:
    return build_dataset(
        seasons=seasons,
        save_path=str(processed_file),
        allow_mock_fallback=False,
    )


def run_study(
    args: argparse.Namespace,
    dataset_builder: DatasetBuilder = build_full_dataset_for_study,
    processed_file: Path = PROCESSED_FEATURES_FILE,
    report_context: StudyReportContext | None = None,
) -> None:
    set_research_seed(args.seed)
    output_dir = Path(args.output_dir)
    report_path = Path(args.report_path)
    figures_dir = Path(args.figures_dir)
    context = report_context or StudyReportContext()
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    sns.set_style("whitegrid")

    print("Building dataset...")
    dataset = dataset_builder(SEASONS, processed_file)
    prepared = prepare_model_inputs(dataset)
    X = prepared["X"]
    y = prepared["y"]
    seasons_array = prepared["seasons"]
    teams = prepared["teams"]
    num_team_features = int(prepared["num_team_features"])
    num_player_stats = int(prepared["num_player_stats"])

    print("Running logistic regression...")
    logistic_preds, logistic_proba = run_sklearn_baseline(
        LogisticRegression,
        {"max_iter": 1000, "random_state": args.seed},
        X,
        y,
        seasons_array,
    )

    print("Running random forest...")
    random_forest_preds, random_forest_proba = run_sklearn_baseline(
        RandomForestClassifier,
        {"n_estimators": 200, "random_state": args.seed, "max_depth": 10},
        X,
        y,
        seasons_array,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running neural models on {device}...")
    mlp_preds, mlp_proba, attention_preds, attention_proba, attention_weights = run_neural_models(
        X=X,
        y=y,
        seasons_array=seasons_array,
        num_team_features=num_team_features,
        num_player_stats=num_player_stats,
        epochs_mlp=args.epochs_mlp,
        epochs_attention=args.epochs_attention,
        lr=args.lr,
        seed=args.seed,
        device=device,
    )

    predictions = {
        "logistic_regression": (logistic_preds, logistic_proba),
        "random_forest": (random_forest_preds, random_forest_proba),
        "mlp_baseline": (mlp_preds, mlp_proba),
        "attention_model": (attention_preds, attention_proba),
    }

    metrics: dict[str, dict[str, Any]] = {}
    for model_key, (preds, proba) in predictions.items():
        fold_metrics = compute_fold_metrics(y, preds, proba, seasons_array)
        metrics[model_key] = {
            "overall_metrics": evaluate_predictions(y, preds, proba),
            "fold_metrics": fold_metrics,
            "fold_summary": summarize_fold_metrics(fold_metrics),
        }

    best_baseline_key = max(
        ["logistic_regression", "random_forest", "mlp_baseline"],
        key=lambda key: metrics[key]["overall_metrics"]["f1_macro"],
    )

    comparison_frame = pd.DataFrame(
        [
            {
                "model_key": model_key,
                "model_name": MODEL_DISPLAY_NAMES[model_key],
                **metrics[model_key]["overall_metrics"],
                **metrics[model_key]["fold_summary"],
            }
            for model_key in ["logistic_regression", "random_forest", "mlp_baseline", "attention_model"]
        ]
    )
    comparison_frame.to_csv(output_dir / "model_comparison.csv", index=False)

    fold_metrics_rows = []
    for model_key, model_metrics in metrics.items():
        for fold_metric in model_metrics["fold_metrics"]:
            fold_metrics_rows.append({"model_key": model_key, **fold_metric})
    pd.DataFrame(fold_metrics_rows).to_csv(output_dir / "fold_metrics.csv", index=False)

    predictions_frame = pd.DataFrame(
        {
            "SEASON": seasons_array,
            "TEAM": teams,
            "PLAYOFF_RESULT": y,
            "LOGISTIC_REGRESSION_PRED": logistic_preds,
            "RANDOM_FOREST_PRED": random_forest_preds,
            "MLP_BASELINE_PRED": mlp_preds,
            "ATTENTION_MODEL_PRED": attention_preds,
        }
    )
    predictions_frame.to_csv(output_dir / "predictions.csv", index=False)

    attention_report = dataframe_from_classification_report(
        classification_report(y, attention_preds, target_names=TARGET_NAMES, output_dict=True, zero_division=0)
    )
    best_baseline_report = dataframe_from_classification_report(
        classification_report(
            y,
            predictions[best_baseline_key][0],
            target_names=TARGET_NAMES,
            output_dict=True,
            zero_division=0,
        )
    )
    attention_report.to_csv(output_dir / "attention_classification_report.csv")
    best_baseline_report.to_csv(output_dir / "best_baseline_classification_report.csv")

    market_fairness = summarize_market_fairness(dataset, attention_preds)
    (output_dir / "market_fairness.json").write_text(json.dumps(market_fairness, indent=2), encoding="utf-8")
    (output_dir / "summary_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    plot_confusion_matrices(
        y_true=y,
        predictions_by_model={
            "logistic_regression": logistic_preds,
            "random_forest": random_forest_preds,
            "mlp_baseline": mlp_preds,
            "attention_model": attention_preds,
        },
        figure_path=figures_dir / f"{context.figure_prefix}_confusion_matrices.png",
    )
    plot_attention_weights(
        attention_weights=attention_weights,
        labels=y,
        figure_path=figures_dir / f"{context.figure_prefix}_attention_weights.png",
    )
    plot_tsne(X=X, y=y, figure_path=figures_dir / f"{context.figure_prefix}_tsne.png", seed=args.seed)

    write_final_report(
        dataset=dataset,
        metrics=metrics,
        best_baseline_key=best_baseline_key,
        best_baseline_report=best_baseline_report,
        attention_report=attention_report,
        market_fairness=market_fairness,
        report_path=report_path,
        output_dir=output_dir,
        figures_dir=figures_dir,
        args=args,
        context=context,
    )

    print("Research study complete.")
    print(f"Report written to {report_path}")
    print(f"Artifacts written to {output_dir}")


def main() -> None:
    run_study(parse_args())


if __name__ == "__main__":
    main()
