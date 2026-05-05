"""Generate additional real-data presentation figures for the NBA study."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "features_research.csv"
PREDICTIONS_PATH = PROJECT_ROOT / "results" / "research_study" / "predictions.csv"
MODEL_COMPARISON_PATH = PROJECT_ROOT / "results" / "research_study" / "model_comparison.csv"
ATTENTION_FIGURE_PATH = PROJECT_ROOT / "docs" / "figures" / "research_attention_weights.png"
FIGURES_DIR = PROJECT_ROOT / "docs" / "figures"

LABEL_NAMES = {
    0: "Missed Playoffs",
    1: "First Round Exit",
    2: "Second Round Exit",
    3: "Conference Finals",
    4: "Finals Loss",
    5: "Champion",
}

LABEL_SHORT = {
    0: "Miss",
    1: "R1",
    2: "R2",
    3: "Conf",
    4: "Finals L",
    5: "Champ",
}

MODEL_COLUMNS = {
    "Logistic Regression": "LOGISTIC_REGRESSION_PRED",
    "Random Forest": "RANDOM_FOREST_PRED",
    "MLP Baseline": "MLP_BASELINE_PRED",
    "Attention Model": "ATTENTION_MODEL_PRED",
}

CLASS_COLORS = [
    "#7f8c8d",
    "#4e79a7",
    "#59a14f",
    "#f28e2b",
    "#e15759",
    "#b07aa1",
]

DISCRETE_CMAP = ListedColormap(CLASS_COLORS)
DISCRETE_NORM = BoundaryNorm(np.arange(-0.5, 6.5, 1), DISCRETE_CMAP.N)


def load_artifacts() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    features = pd.read_csv(FEATURES_PATH)
    predictions = pd.read_csv(PREDICTIONS_PATH)
    model_comparison = pd.read_csv(MODEL_COMPARISON_PATH)
    return features, predictions, model_comparison


def save_figure(fig: plt.Figure, filename: str) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    output_path = FIGURES_DIR / filename
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path.relative_to(PROJECT_ROOT)}")


def add_discrete_colorbar(fig: plt.Figure, axis: plt.Axes, label: str = "Playoff outcome") -> None:
    colorbar = fig.colorbar(
        plt.cm.ScalarMappable(cmap=DISCRETE_CMAP, norm=DISCRETE_NORM),
        ax=axis,
        ticks=list(LABEL_NAMES.keys()),
        fraction=0.046,
        pad=0.04,
    )
    colorbar.ax.set_yticklabels([LABEL_SHORT[index] for index in LABEL_NAMES])
    colorbar.set_label(label)


def plot_tsne_cluster_analysis(features: pd.DataFrame) -> None:
    metadata_columns = {"SEASON", "TEAM", "TEAM_ID", "PLAYOFF_RESULT", "PLAYOFF_LABEL"}
    feature_columns = [column for column in features.columns if column not in metadata_columns]
    x = features[feature_columns].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    y = features["PLAYOFF_RESULT"].astype(int).to_numpy()

    x_scaled = StandardScaler().fit_transform(x)
    embedding = TSNE(
        n_components=2,
        perplexity=30,
        learning_rate="auto",
        init="pca",
        random_state=42,
    ).fit_transform(x_scaled)

    cluster_score = silhouette_score(embedding, y)
    plot_data = features[["SEASON", "TEAM", "PLAYOFF_RESULT", "PLAYOFF_LABEL"]].copy()
    plot_data["TSNE_1"] = embedding[:, 0]
    plot_data["TSNE_2"] = embedding[:, 1]

    fig, axis = plt.subplots(figsize=(12, 9))
    markers = {
        0: "o",
        1: "s",
        2: "^",
        3: "D",
        4: "P",
        5: "*",
    }

    for label_index, label_name in LABEL_NAMES.items():
        subset = plot_data[plot_data["PLAYOFF_RESULT"] == label_index]
        size = 190 if label_index == 5 else 95 if label_index == 4 else 65
        edgecolor = "black" if label_index >= 4 else "white"
        linewidth = 0.9 if label_index >= 4 else 0.35
        axis.scatter(
            subset["TSNE_1"],
            subset["TSNE_2"],
            label=f"{label_name} (n={len(subset)})",
            c=[CLASS_COLORS[label_index]],
            marker=markers[label_index],
            s=size,
            alpha=0.78,
            edgecolors=edgecolor,
            linewidths=linewidth,
        )

    centroids = plot_data.groupby("PLAYOFF_RESULT")[["TSNE_1", "TSNE_2"]].mean()
    for label_index, row in centroids.iterrows():
        axis.scatter(row["TSNE_1"], row["TSNE_2"], c="black", s=55, marker="x", linewidths=2)
        axis.text(
            row["TSNE_1"],
            row["TSNE_2"],
            f"  {LABEL_SHORT[int(label_index)]}",
            fontsize=10,
            fontweight="bold",
            va="center",
        )

    recent_finals = plot_data[
        (plot_data["PLAYOFF_RESULT"].isin([4, 5]))
        & (plot_data["SEASON"].isin(["2021-22", "2022-23", "2023-24"]))
    ]
    for _, row in recent_finals.iterrows():
        axis.annotate(
            f"{row['SEASON']} {row['TEAM']}",
            (row["TSNE_1"], row["TSNE_2"]),
            textcoords="offset points",
            xytext=(7, 7),
            fontsize=8,
            color="#333333",
        )

    axis.set_title(
        "t-SNE Clustering Analysis of Team-Seasons",
        fontsize=16,
        fontweight="bold",
    )
    axis.set_xlabel("t-SNE dimension 1")
    axis.set_ylabel("t-SNE dimension 2")
    axis.text(
        0.01,
        0.01,
        f"Real feature matrix: {len(features)} team-seasons, {len(feature_columns)} numeric features\n"
        f"2D silhouette by playoff-depth label: {cluster_score:.3f}",
        transform=axis.transAxes,
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.92},
    )
    axis.legend(loc="upper right", fontsize=9, frameon=True)
    axis.grid(alpha=0.25)
    save_figure(fig, "research_tsne_cluster_analysis.png")


def format_prediction_matrix(predictions: pd.DataFrame, include_all_finals: bool) -> tuple[np.ndarray, list[str], list[str]]:
    if include_all_finals:
        rows = predictions[predictions["PLAYOFF_RESULT"].isin([4, 5])].copy()
        rows = rows.sort_values(["SEASON", "PLAYOFF_RESULT", "TEAM"], ascending=[False, False, True])
        row_labels = [
            f"{row.SEASON} {row.TEAM} ({LABEL_SHORT[int(row.PLAYOFF_RESULT)]})"
            for row in rows.itertuples(index=False)
        ]
    else:
        selected_cases = [
            ("2023-24", "BOS"),
            ("2023-24", "DAL"),
            ("2023-24", "DEN"),
            ("2022-23", "DEN"),
            ("2022-23", "MIA"),
            ("2022-23", "BOS"),
            ("2018-19", "TOR"),
            ("2018-19", "GSW"),
        ]
        selected_frames = []
        for season, team in selected_cases:
            selected_frames.append(
                predictions[(predictions["SEASON"] == season) & (predictions["TEAM"] == team)]
            )
        rows = pd.concat(selected_frames, ignore_index=True)
        row_labels = [
            f"{row.SEASON} {row.TEAM} | actual: {LABEL_SHORT[int(row.PLAYOFF_RESULT)]} | finals: "
            f"{'Yes' if int(row.PLAYOFF_RESULT) >= 4 else 'No'}"
            for row in rows.itertuples(index=False)
        ]

    columns = ["Actual"] + list(MODEL_COLUMNS.keys())
    matrix_columns = ["PLAYOFF_RESULT"] + list(MODEL_COLUMNS.values())
    matrix = rows[matrix_columns].astype(int).to_numpy()
    return matrix, row_labels, columns


def plot_prediction_heatmap(
    matrix: np.ndarray,
    row_labels: list[str],
    column_labels: list[str],
    title: str,
    filename: str,
) -> None:
    fig_height = max(6, 0.32 * len(row_labels) + 1.5)
    fig, axis = plt.subplots(figsize=(12, fig_height))
    axis.imshow(matrix, cmap=DISCRETE_CMAP, norm=DISCRETE_NORM, aspect="auto")

    axis.set_xticks(np.arange(len(column_labels)))
    axis.set_xticklabels(column_labels, rotation=20, ha="right")
    axis.set_yticks(np.arange(len(row_labels)))
    axis.set_yticklabels(row_labels, fontsize=8 if len(row_labels) > 12 else 10)
    axis.set_title(title, fontsize=15, fontweight="bold")

    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            value = int(matrix[row_index, column_index])
            text_color = "white" if value in (3, 4, 5) else "black"
            axis.text(
                column_index,
                row_index,
                LABEL_SHORT[value],
                ha="center",
                va="center",
                fontsize=8,
                color=text_color,
                fontweight="bold",
            )

    axis.set_xticks(np.arange(-0.5, len(column_labels), 1), minor=True)
    axis.set_yticks(np.arange(-0.5, len(row_labels), 1), minor=True)
    axis.grid(which="minor", color="white", linestyle="-", linewidth=1.0)
    axis.tick_params(which="minor", bottom=False, left=False)
    add_discrete_colorbar(fig, axis)
    save_figure(fig, filename)


def plot_finals_detection(predictions: pd.DataFrame) -> None:
    finals = predictions[predictions["PLAYOFF_RESULT"].isin([4, 5])].copy()
    records = []
    for model_name, column in MODEL_COLUMNS.items():
        predictions_as_int = finals[column].astype(int)
        actual_as_int = finals["PLAYOFF_RESULT"].astype(int)
        records.append(
            {
                "Model": model_name,
                "Predicted Finalist": (predictions_as_int >= 4).mean(),
                "Exact Finals Class": (predictions_as_int == actual_as_int).mean(),
                "Predicted Champion for Champions": (
                    predictions_as_int[actual_as_int == 5] == 5
                ).mean(),
            }
        )

    detection = pd.DataFrame(records)
    x = np.arange(len(detection))
    width = 0.25
    fig, axis = plt.subplots(figsize=(11, 6))
    colors = ["#4e79a7", "#59a14f", "#b07aa1"]
    metric_columns = ["Predicted Finalist", "Exact Finals Class", "Predicted Champion for Champions"]

    for offset, metric in enumerate(metric_columns):
        values = detection[metric].to_numpy()
        bars = axis.bar(x + (offset - 1) * width, values, width=width, label=metric, color=colors[offset])
        for bar, value in zip(bars, values):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.015,
                f"{value * 100:.0f}%",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    axis.set_title("How Often Actual Finals Teams Were Recognized", fontsize=15, fontweight="bold")
    axis.set_ylabel("Share of actual Finals team-seasons")
    max_value = detection[metric_columns].to_numpy().max()
    axis.set_ylim(0, max(0.5, max_value * 1.25))
    axis.set_xticks(x)
    axis.set_xticklabels(detection["Model"], rotation=15, ha="right")
    axis.text(
        0.01,
        0.95,
        f"Real out-of-fold predictions for {len(finals)} actual Finals teams\n"
        "Finalist means predicted class is Finals Loss or Champion.",
        transform=axis.transAxes,
        fontsize=10,
        va="top",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.92},
    )
    axis.legend(loc="upper right")
    axis.grid(axis="y", alpha=0.25)
    save_figure(fig, "finals_detection_by_model.png")


def plot_full_season_results_bridge(model_comparison: pd.DataFrame) -> None:
    metrics = [
        ("accuracy", "Accuracy"),
        ("f1_macro", "Macro F1"),
        ("top2_accuracy", "Top-2 Accuracy"),
    ]
    model_comparison = model_comparison.copy()
    model_comparison["model_name"] = pd.Categorical(
        model_comparison["model_name"],
        ["Logistic Regression", "Random Forest", "MLP Baseline", "Attention Model"],
        ordered=True,
    )
    model_comparison = model_comparison.sort_values("model_name")

    fig, axes = plt.subplots(2, 1, figsize=(14, 13), gridspec_kw={"height_ratios": [1.0, 1.15]})
    axis = axes[0]
    x = np.arange(len(model_comparison))
    width = 0.24
    metric_colors = ["#4e79a7", "#59a14f", "#f28e2b"]

    for offset, (column, label) in enumerate(metrics):
        values = model_comparison[column].astype(float).to_numpy()
        bars = axis.bar(x + (offset - 1) * width, values, width=width, label=label, color=metric_colors[offset])
        for bar, value in zip(bars, values):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.01,
                f"{value * 100:.1f}%",
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=90,
            )

    axis.set_title("Full-Season Results: Random Forest Leads Accuracy", fontsize=14, fontweight="bold")
    axis.set_ylabel("Score")
    axis.set_ylim(0, 0.95)
    axis.set_xticks(x)
    axis.set_xticklabels(model_comparison["model_name"], rotation=20, ha="right")
    axis.legend(loc="upper left")
    axis.grid(axis="y", alpha=0.25)

    rf_row = model_comparison[model_comparison["model_name"] == "Random Forest"].iloc[0]
    attention_row = model_comparison[model_comparison["model_name"] == "Attention Model"].iloc[0]
    axis.text(
        0.02,
        0.04,
        "Key result:\n"
        f"Random Forest accuracy: {rf_row['accuracy'] * 100:.1f}%\n"
        f"Attention accuracy: {attention_row['accuracy'] * 100:.1f}%\n"
        "Attention remains competitive while exposing roster-slot weights.",
        transform=axis.transAxes,
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.95},
    )

    attention_axis = axes[1]
    if ATTENTION_FIGURE_PATH.exists():
        attention_image = mpimg.imread(ATTENTION_FIGURE_PATH)
        attention_axis.imshow(attention_image)
        attention_axis.set_title("Attention Adds Interpretability", fontsize=14, fontweight="bold")
        attention_axis.axis("off")
        attention_axis.text(
            0.02,
            0.91,
            "Real attention-weight artifact:\nwhich rotation slots the model used by outcome class",
            transform=attention_axis.transAxes,
            fontsize=10,
            va="top",
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.95},
        )
    else:
        attention_axis.text(0.5, 0.5, "Attention-weight figure missing", ha="center", va="center")
        attention_axis.axis("off")

    fig.suptitle(
        "Full-Season Model Story: Best Accuracy vs. Interpretable Deep Model",
        fontsize=16,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    save_figure(fig, "full_season_results_random_forest_attention.png")


def main() -> None:
    features, predictions, model_comparison = load_artifacts()
    plot_tsne_cluster_analysis(features)

    matrix, row_labels, column_labels = format_prediction_matrix(predictions, include_all_finals=True)
    plot_prediction_heatmap(
        matrix,
        row_labels,
        column_labels,
        "Actual Finals Teams: Model Predictions vs. Real Outcomes",
        "finals_team_prediction_heatmap.png",
    )

    matrix, row_labels, column_labels = format_prediction_matrix(predictions, include_all_finals=False)
    plot_prediction_heatmap(
        matrix,
        row_labels,
        column_labels,
        "Selected Finals and Near-Finals Case Studies",
        "finals_case_study_predictions.png",
    )

    plot_finals_detection(predictions)
    plot_full_season_results_bridge(model_comparison)


if __name__ == "__main__":
    main()
