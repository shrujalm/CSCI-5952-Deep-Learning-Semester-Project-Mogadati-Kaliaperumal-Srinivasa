"""Interpretability routines for the championship prediction models."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

from sklearn.manifold import TSNE

from .modeling import AttentionModel, MLPBaseline, PLAYOFF_CLASS_NAMES


def _resolve_class_names(class_names: Optional[List[str]]) -> List[str]:
    return class_names or list(PLAYOFF_CLASS_NAMES)


def _save_or_show(fig: plt.Figure, save_path: Optional[str]) -> None:
    if save_path:
        output_path = Path(save_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved figure to {output_path}")
        plt.close(fig)
        return
    plt.show()


def _build_tsne(embedding_count: int) -> TSNE:
    perplexity = min(30, max(1, embedding_count - 1))
    return TSNE(
        n_components=2,
        random_state=42,
        perplexity=perplexity,
        n_jobs=1,
    )


def extract_team_embeddings(
    model: AttentionModel,
    team_features: torch.Tensor,
    player_features: torch.Tensor,
) -> torch.Tensor:
    """Return the attention model's joint team representation."""
    model.eval()
    with torch.no_grad():
        combined, _ = model.encode_team_representation(team_features, player_features)
    return combined


def visualize_attention_weights(
    model: AttentionModel,
    team_features: torch.Tensor,
    player_features: torch.Tensor,
    labels: torch.Tensor,
    player_names: Optional[List[str]] = None,
    class_names: Optional[List[str]] = None,
    save_path: Optional[str] = None,
) -> Dict:
    """Plot attention distributions stratified by playoff outcome class."""
    player_names = player_names or [f"Player {index + 1}" for index in range(model.num_players)]
    class_names = _resolve_class_names(class_names)

    model.eval()
    with torch.no_grad():
        _, attention_weights = model(team_features, player_features, return_attention=True)

    attention_weights_np = attention_weights.cpu().numpy()
    labels_np = labels.cpu().numpy()

    class_attention = {}
    for class_idx in range(model.num_classes):
        class_mask = labels_np == class_idx
        if class_mask.sum() == 0:
            continue
        class_attention[class_idx] = {
            "mean": attention_weights_np[class_mask].mean(axis=0),
            "std": attention_weights_np[class_mask].std(axis=0),
            "count": int(class_mask.sum()),
        }

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    flat_axes = axes.flatten()
    x = np.arange(len(player_names))

    for class_idx, axis in enumerate(flat_axes):
        if class_idx in class_attention:
            mean_attention = class_attention[class_idx]["mean"]
            std_attention = class_attention[class_idx]["std"]
            axis.bar(x, mean_attention, yerr=std_attention, capsize=3, alpha=0.75)
            axis.set_xticks(x)
            axis.set_xticklabels(player_names, rotation=45, ha="right")
            axis.set_ylabel("Attention Weight")
            axis.set_ylim(0, 1)
            axis.axhline(
                y=1 / model.num_players,
                color="red",
                linestyle="--",
                label="Uniform attention",
            )
            axis.legend()
            axis.set_title(f"{class_names[class_idx]}\n(n={class_attention[class_idx]['count']})")
        else:
            axis.text(0.5, 0.5, "No samples", ha="center", va="center", transform=axis.transAxes)
            axis.set_title(class_names[class_idx])

    fig.suptitle(
        "Attention Weights by Playoff Outcome",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout()
    _save_or_show(fig, save_path)
    return class_attention


def visualize_tsne_embeddings(
    model: AttentionModel,
    team_features: torch.Tensor,
    player_features: torch.Tensor,
    labels: torch.Tensor,
    class_names: Optional[List[str]] = None,
    save_path: Optional[str] = None,
) -> np.ndarray:
    """Project the learned joint team representation into two dimensions."""
    class_names = _resolve_class_names(class_names)
    embeddings = extract_team_embeddings(model, team_features, player_features).cpu().numpy()
    labels_np = labels.cpu().numpy()

    print("Computing t-SNE projection...")
    tsne_embeddings = _build_tsne(len(embeddings)).fit_transform(embeddings)

    fig, axis = plt.subplots(figsize=(12, 10))
    colors = plt.cm.RdYlGn(np.linspace(0.1, 0.9, len(class_names)))
    markers = ["o", "s", "^", "D", "v", "*"]

    for class_idx in range(len(class_names)):
        class_mask = labels_np == class_idx
        if class_mask.sum() == 0:
            continue
        axis.scatter(
            tsne_embeddings[class_mask, 0],
            tsne_embeddings[class_mask, 1],
            c=[colors[class_idx]],
            marker=markers[class_idx],
            label=class_names[class_idx],
            alpha=0.65,
            s=100,
            edgecolors="black",
            linewidth=0.5,
        )

    axis.set_xlabel("t-SNE Dimension 1")
    axis.set_ylabel("t-SNE Dimension 2")
    axis.set_title(
        "t-SNE of Attention-Model Team Embeddings",
        fontsize=14,
        fontweight="bold",
    )
    axis.legend(loc="best", title="Playoff Outcome")
    axis.grid(True, alpha=0.3)
    fig.tight_layout()
    _save_or_show(fig, save_path)

    return tsne_embeddings


def analyze_champion_clustering(
    tsne_embeddings: np.ndarray,
    labels: np.ndarray,
    class_names: Optional[List[str]] = None,
) -> Dict:
    """Measure whether champions occupy a more coherent local neighborhood."""
    _ = _resolve_class_names(class_names)
    champion_label = 5
    champion_mask = labels == champion_label
    non_champion_mask = ~champion_mask

    if champion_mask.sum() < 2:
        return {"error": "Not enough champion samples to analyze clustering."}

    champion_embeddings = tsne_embeddings[champion_mask]
    non_champion_embeddings = tsne_embeddings[non_champion_mask]

    champ_to_champ_distances = []
    for i in range(len(champion_embeddings)):
        for j in range(i + 1, len(champion_embeddings)):
            champ_to_champ_distances.append(
                float(np.linalg.norm(champion_embeddings[i] - champion_embeddings[j]))
            )

    champ_to_nonchamp_distances = []
    for champion_embedding in champion_embeddings:
        distances = np.linalg.norm(non_champion_embeddings - champion_embedding, axis=1)
        champ_to_nonchamp_distances.append(float(distances.min()))

    clustering_ratio = float(np.mean(champ_to_nonchamp_distances) / np.mean(champ_to_champ_distances))
    if clustering_ratio > 1.5:
        interpretation = "Strong clustering: champions form a distinct neighborhood."
    elif clustering_ratio > 1.2:
        interpretation = "Moderate clustering: champions are somewhat more compact than the rest."
    else:
        interpretation = "Weak clustering: champions remain interspersed with non-champions."

    return {
        "num_champions": int(champion_mask.sum()),
        "num_non_champions": int(non_champion_mask.sum()),
        "avg_champ_to_champ_distance": float(np.mean(champ_to_champ_distances)),
        "std_champ_to_champ_distance": float(np.std(champ_to_champ_distances)),
        "avg_champ_to_nonchamp_distance": float(np.mean(champ_to_nonchamp_distances)),
        "std_champ_to_nonchamp_distance": float(np.std(champ_to_nonchamp_distances)),
        "clustering_ratio": clustering_ratio,
        "interpretation": interpretation,
    }


def compare_model_embeddings(
    attention_model: AttentionModel,
    mlp_model: MLPBaseline,
    team_features: torch.Tensor,
    player_features: torch.Tensor,
    labels: torch.Tensor,
    class_names: Optional[List[str]] = None,
    save_path: Optional[str] = None,
) -> Dict:
    """Compare embedding geometry induced by the two architectures."""
    class_names = _resolve_class_names(class_names)
    attention_embeddings = extract_team_embeddings(
        attention_model,
        team_features,
        player_features,
    ).cpu().numpy()

    mlp_model.eval()
    with torch.no_grad():
        mlp_embeddings = mlp_model.encode_features(team_features, player_features).cpu().numpy()

    labels_np = labels.cpu().numpy()
    attention_tsne = _build_tsne(len(attention_embeddings)).fit_transform(attention_embeddings)
    mlp_tsne = _build_tsne(len(mlp_embeddings)).fit_transform(mlp_embeddings)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    colors = plt.cm.RdYlGn(np.linspace(0.1, 0.9, len(class_names)))
    markers = ["o", "s", "^", "D", "v", "*"]

    for plot_index, (embeddings, title, axis) in enumerate(
        [
            (attention_tsne, "Attention Model", axes[0]),
            (mlp_tsne, "MLP Baseline", axes[1]),
        ]
    ):
        for class_idx in range(len(class_names)):
            class_mask = labels_np == class_idx
            if class_mask.sum() == 0:
                continue
            axis.scatter(
                embeddings[class_mask, 0],
                embeddings[class_mask, 1],
                c=[colors[class_idx]],
                marker=markers[class_idx],
                label=class_names[class_idx] if plot_index == 0 else "",
                alpha=0.65,
                s=100,
                edgecolors="black",
                linewidth=0.5,
            )

        axis.set_xlabel("t-SNE Dimension 1")
        axis.set_ylabel("t-SNE Dimension 2")
        axis.set_title(title, fontsize=12, fontweight="bold")
        axis.grid(True, alpha=0.3)

    axes[0].legend(loc="best", title="Playoff Outcome")
    fig.suptitle("Embedding Geometry Comparison", fontsize=14, fontweight="bold")
    fig.tight_layout()
    _save_or_show(fig, save_path)

    return {
        "attention_model": analyze_champion_clustering(attention_tsne, labels_np),
        "mlp_baseline": analyze_champion_clustering(mlp_tsne, labels_np),
    }
