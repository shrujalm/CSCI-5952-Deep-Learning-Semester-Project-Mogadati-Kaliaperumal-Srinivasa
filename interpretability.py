"""
interpretability.py - Interpretability Analysis for NBA Championship Prediction

Implements the interpretability analysis from Section 4 of the proposal:
- Visualize attention weights to see which players matter most
- t-SNE visualization of learned team embeddings
- Check if championship teams cluster together
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from typing import Dict, List, Tuple, Optional
import json

from models import AttentionModel, MLPBaseline


def extract_team_embeddings(model: AttentionModel,
                            team_features: torch.Tensor,
                            player_features: torch.Tensor) -> torch.Tensor:
    """
    Extract the combined team representation (team stats + attention-pooled player embeddings).
    
    This is the representation that goes into the final classifier.
    
    Args:
        model: Trained AttentionModel
        team_features: (batch_size, team_stat_size)
        player_features: (batch_size, num_players * player_stat_size)
    
    Returns:
        team_embeddings: (batch_size, team_stat_size + embedding_size)
    """
    model.eval()
    with torch.no_grad():
        batch_size = player_features.shape[0]
        
        # Get player embeddings
        players = player_features.view(batch_size, model.num_players, model.player_stat_size)
        embeddings = []
        for i in range(model.num_players):
            emb = model.player_embedding(players[:, i, :])
            embeddings.append(emb)
        embeddings = torch.stack(embeddings, dim=1)
        
        # Apply attention pooling
        team_player_summary, _ = model.attention(embeddings)
        
        # Combine with team features (this is what the classifier sees)
        combined = torch.cat([team_features, team_player_summary], dim=1)
    
    return combined


def visualize_attention_weights(model: AttentionModel,
                                team_features: torch.Tensor,
                                player_features: torch.Tensor,
                                labels: torch.Tensor,
                                player_names: Optional[List[str]] = None,
                                class_names: Optional[List[str]] = None,
                                save_path: Optional[str] = None) -> Dict:
    """
    Visualize attention weights to understand which players the model considers most important.
    
    From proposal: "We will visualize the attention weights to see which players 
    the model thinks are most important for championship contention."
    
    Args:
        model: Trained AttentionModel
        team_features: (num_samples, team_stat_size)
        player_features: (num_samples, num_players * player_stat_size)
        labels: (num_samples,) true labels
        player_names: Optional list of player position names (e.g., ['Star', 'Starter2', ...])
        class_names: Optional list of class names
        save_path: Optional path to save the figure
    
    Returns:
        Dictionary with attention statistics by class
    """
    if player_names is None:
        player_names = [f"Player {i+1}" for i in range(model.num_players)]
    
    if class_names is None:
        class_names = ['Missed Playoffs', 'First Round', 'Second Round', 
                      'Conf Finals', 'Finals', 'Champion']
    
    model.eval()
    with torch.no_grad():
        _, attention_weights = model(team_features, player_features, return_attention=True)
    
    attention_weights = attention_weights.cpu().numpy()
    labels = labels.cpu().numpy()
    
    # Compute statistics by class
    class_attention = {}
    for class_idx in range(model.classifier[-1].out_features):
        mask = labels == class_idx
        if mask.sum() > 0:
            class_attention[class_idx] = {
                'mean': attention_weights[mask].mean(axis=0),
                'std': attention_weights[mask].std(axis=0),
                'count': int(mask.sum())
            }
    
    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for class_idx, ax in enumerate(axes):
        if class_idx in class_attention:
            mean_attn = class_attention[class_idx]['mean']
            std_attn = class_attention[class_idx]['std']
            
            x = np.arange(len(player_names))
            ax.bar(x, mean_attn, yerr=std_attn, capsize=3, alpha=0.7)
            ax.set_xticks(x)
            ax.set_xticklabels(player_names, rotation=45, ha='right')
            ax.set_ylabel('Attention Weight')
            ax.set_title(f'{class_names[class_idx]}\n(n={class_attention[class_idx]["count"]})')
            ax.set_ylim(0, 1)
            ax.axhline(y=1/model.num_players, color='r', linestyle='--', 
                      label='Uniform (no preference)')
            ax.legend()
        else:
            ax.text(0.5, 0.5, 'No samples', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(class_names[class_idx])
    
    plt.suptitle('Attention Weights by Playoff Outcome\n(Which players matter most?)', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Attention visualization saved to {save_path}")
    
    plt.show()
    
    return class_attention


def visualize_tsne_embeddings(model: AttentionModel,
                              team_features: torch.Tensor,
                              player_features: torch.Tensor,
                              labels: torch.Tensor,
                              class_names: Optional[List[str]] = None,
                              save_path: Optional[str] = None) -> np.ndarray:
    """
    Create t-SNE visualization of learned team embeddings.
    
    From proposal: "We will also use t-SNE to visualize the learned team embeddings 
    and see if championship teams cluster together in the embedding space."
    
    Args:
        model: Trained AttentionModel
        team_features: (num_samples, team_stat_size)
        player_features: (num_samples, num_players * player_stat_size)
        labels: (num_samples,) true labels
        class_names: Optional list of class names
        save_path: Optional path to save the figure
    
    Returns:
        tsne_embeddings: (num_samples, 2) 2D coordinates
    """
    if class_names is None:
        class_names = ['Missed Playoffs', 'First Round', 'Second Round', 
                      'Conf Finals', 'Finals', 'Champion']
    
    # Extract team embeddings
    embeddings = extract_team_embeddings(model, team_features, player_features)
    embeddings = embeddings.cpu().numpy()
    labels = labels.cpu().numpy()
    
    # Apply t-SNE
    print("Computing t-SNE (this may take a moment)...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings)-1))
    tsne_embeddings = tsne.fit_transform(embeddings)
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(12, 10))
    
    colors = plt.cm.RdYlGn(np.linspace(0.1, 0.9, 6))
    markers = ['o', 's', '^', 'D', 'v', '*']
    
    for class_idx in range(6):
        mask = labels == class_idx
        if mask.sum() > 0:
            ax.scatter(tsne_embeddings[mask, 0], tsne_embeddings[mask, 1],
                      c=[colors[class_idx]], marker=markers[class_idx],
                      label=class_names[class_idx], alpha=0.6, s=100,
                      edgecolors='black', linewidth=0.5)
    
    ax.set_xlabel('t-SNE Dimension 1')
    ax.set_ylabel('t-SNE Dimension 2')
    ax.set_title('t-SNE Visualization of Team Embeddings\n(Do championship teams cluster together?)',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='best', title='Playoff Outcome')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"t-SNE visualization saved to {save_path}")
    
    plt.show()
    
    return tsne_embeddings


def analyze_champion_clustering(tsne_embeddings: np.ndarray,
                                labels: np.ndarray,
                                class_names: Optional[List[str]] = None) -> Dict:
    """
    Analyze whether championship teams form a distinct cluster.
    
    Computes the average distance from champions to other champions vs 
    distance from champions to non-champions.
    
    Args:
        tsne_embeddings: (num_samples, 2) from t-SNE
        labels: (num_samples,) true labels
        class_names: Optional list of class names
    
    Returns:
        Dictionary with clustering statistics
    """
    if class_names is None:
        class_names = ['Missed Playoffs', 'First Round', 'Second Round', 
                      'Conf Finals', 'Finals', 'Champion']
    
    champion_label = 5  # Champion class index
    champion_mask = labels == champion_label
    non_champion_mask = ~champion_mask
    
    if champion_mask.sum() < 2:
        return {"error": "Not enough champion samples to analyze clustering"}
    
    champion_embeddings = tsne_embeddings[champion_mask]
    non_champion_embeddings = tsne_embeddings[non_champion_mask]
    
    # Compute pairwise distances between champions
    champ_to_champ_distances = []
    for i in range(len(champion_embeddings)):
        for j in range(i+1, len(champion_embeddings)):
            dist = np.linalg.norm(champion_embeddings[i] - champion_embeddings[j])
            champ_to_champ_distances.append(dist)
    
    # Compute distances from champions to nearest non-champions
    champ_to_nonchamp_distances = []
    for champ_emb in champion_embeddings:
        distances = np.linalg.norm(non_champion_embeddings - champ_emb, axis=1)
        champ_to_nonchamp_distances.append(distances.min())
    
    results = {
        "num_champions": int(champion_mask.sum()),
        "num_non_champions": int(non_champion_mask.sum()),
        "avg_champ_to_champ_distance": float(np.mean(champ_to_champ_distances)),
        "std_champ_to_champ_distance": float(np.std(champ_to_champ_distances)),
        "avg_champ_to_nonchamp_distance": float(np.mean(champ_to_nonchamp_distances)),
        "std_champ_to_nonchamp_distance": float(np.std(champ_to_nonchamp_distances)),
        "clustering_ratio": float(np.mean(champ_to_nonchamp_distances) / np.mean(champ_to_champ_distances)),
        "interpretation": ""
    }
    
    if results["clustering_ratio"] > 1.5:
        results["interpretation"] = "Strong clustering: Champions form a distinct group"
    elif results["clustering_ratio"] > 1.2:
        results["interpretation"] = "Moderate clustering: Champions tend to cluster together"
    else:
        results["interpretation"] = "Weak clustering: Champions are spread throughout the embedding space"
    
    return results


def compare_model_embeddings(attention_model: AttentionModel,
                             mlp_model: MLPBaseline,
                             team_features: torch.Tensor,
                             player_features: torch.Tensor,
                             labels: torch.Tensor,
                             class_names: Optional[List[str]] = None,
                             save_path: Optional[str] = None) -> Dict:
    """
    Compare the embedding spaces of AttentionModel vs MLPBaseline.
    
    Args:
        attention_model: Trained AttentionModel
        mlp_model: Trained MLPBaseline
        team_features: (num_samples, team_stat_size)
        player_features: (num_samples, num_players * player_stat_size)
        labels: (num_samples,) true labels
        class_names: Optional list of class names
        save_path: Optional path to save the figure
    
    Returns:
        Dictionary with comparison statistics
    """
    if class_names is None:
        class_names = ['Missed Playoffs', 'First Round', 'Second Round', 
                      'Conf Finals', 'Finals', 'Champion']
    
    # Get embeddings from both models
    attention_embeddings = extract_team_embeddings(attention_model, team_features, player_features)
    attention_embeddings = attention_embeddings.cpu().numpy()
    
    mlp_model.eval()
    with torch.no_grad():
        # For MLP, use the output of the second-to-last layer as "embeddings"
        x = torch.cat([team_features, player_features], dim=1)
        x = mlp_model.network[0](x)  # First linear
        x = mlp_model.network[1](x)  # ReLU
        x = mlp_model.network[2](x)  # Dropout
        x = mlp_model.network[3](x)  # Second linear
        mlp_embeddings = mlp_model.network[4](x).cpu().numpy()  # ReLU
    
    labels_np = labels.cpu().numpy()
    
    # Apply t-SNE to both
    print("Computing t-SNE for both models...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(attention_embeddings)-1))
    
    attention_tsne = tsne.fit_transform(attention_embeddings)
    mlp_tsne = tsne.fit_transform(mlp_embeddings)
    
    # Create side-by-side visualization
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    colors = plt.cm.RdYlGn(np.linspace(0.1, 0.9, 6))
    markers = ['o', 's', '^', 'D', 'v', '*']
    
    for idx, (embeddings, title, ax) in enumerate([
        (attention_tsne, 'AttentionModel', axes[0]),
        (mlp_tsne, 'MLP Baseline', axes[1])
    ]):
        for class_idx in range(6):
            mask = labels_np == class_idx
            if mask.sum() > 0:
                ax.scatter(embeddings[mask, 0], embeddings[mask, 1],
                          c=[colors[class_idx]], marker=markers[class_idx],
                          label=class_names[class_idx] if idx == 0 else "",
                          alpha=0.6, s=100, edgecolors='black', linewidth=0.5)
        
        ax.set_xlabel('t-SNE Dimension 1')
        ax.set_ylabel('t-SNE Dimension 2')
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    axes[0].legend(loc='best', title='Playoff Outcome')
    fig.suptitle('Embedding Space Comparison: AttentionModel vs MLP Baseline',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Comparison visualization saved to {save_path}")
    
    plt.show()
    
    # Analyze clustering for both
    attention_clustering = analyze_champion_clustering(attention_tsne, labels_np)
    mlp_clustering = analyze_champion_clustering(mlp_tsne, labels_np)
    
    return {
        "attention_model": attention_clustering,
        "mlp_baseline": mlp_clustering
    }


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("NBA Championship Prediction - Interpretability Analysis")
    print("=" * 60)
    
    # Configuration
    CONFIG = {
        'team_stat_size': 15,
        'player_stat_size': 10,
        'num_players': 8,
        'num_classes': 6,
        'embedding_size': 32,
        'hidden_size': 128,
        'dropout_rate': 0.3
    }
    
    # Generate dummy data for demonstration
    np.random.seed(42)
    torch.manual_seed(42)
    
    n_samples = 300
    team_features = torch.randn(n_samples, CONFIG['team_stat_size'])
    player_features = torch.randn(n_samples, CONFIG['num_players'] * CONFIG['player_stat_size'])
    
    # Simulate imbalanced class distribution
    labels = torch.tensor(np.random.choice(
        CONFIG['num_classes'], 
        size=n_samples, 
        p=[0.45, 0.20, 0.15, 0.10, 0.07, 0.03]
    ))
    
    # Initialize and load models (in practice, load trained weights)
    attention_model = AttentionModel(
        team_stat_size=CONFIG['team_stat_size'],
        player_stat_size=CONFIG['player_stat_size'],
        num_players=CONFIG['num_players'],
        embedding_size=CONFIG['embedding_size'],
        num_classes=CONFIG['num_classes'],
        hidden_size=CONFIG['hidden_size'],
        dropout_rate=CONFIG['dropout_rate']
    )
    
    mlp_model = MLPBaseline(
        input_size=CONFIG['team_stat_size'] + (CONFIG['num_players'] * CONFIG['player_stat_size']),
        num_classes=CONFIG['num_classes'],
        hidden_size=CONFIG['hidden_size'],
        dropout_rate=CONFIG['dropout_rate']
    )
    
    # Set models to eval mode
    attention_model.eval()
    mlp_model.eval()
    
    # Player names (could be actual player names in practice)
    player_names = ['Star Player', '2nd Star', 'Starter 3', 'Starter 4',
                   'Starter 5', '6th Man', 'Bench 7', 'Bench 8']
    
    class_names = ['Missed Playoffs', 'First Round', 'Second Round',
                   'Conf Finals', 'Finals', 'Champion']
    
    # 1. Visualize attention weights
    print("\n--- Attention Weight Analysis ---")
    attention_stats = visualize_attention_weights(
        attention_model, team_features, player_features, labels,
        player_names=player_names,
        class_names=class_names,
        save_path='attention_weights.png'
    )
    
    # Print statistics
    print("\nAttention Statistics by Outcome:")
    for class_idx, stats in attention_stats.items():
        print(f"\n  {class_names[class_idx]}:")
        print(f"    Most attended player: {player_names[stats['mean'].argmax()]}")
        print(f"    Mean attention: {stats['mean'].round(3)}")
    
    # 2. t-SNE visualization
    print("\n--- t-SNE Embedding Analysis ---")
    tsne_embeddings = visualize_tsne_embeddings(
        attention_model, team_features, player_features, labels,
        class_names=class_names,
        save_path='tsne_embeddings.png'
    )
    
    # 3. Champion clustering analysis
    print("\n--- Champion Clustering Analysis ---")
    clustering_results = analyze_champion_clustering(
        tsne_embeddings, labels.cpu().numpy()
    )
    
    print(f"\n  Number of champions in dataset: {clustering_results['num_champions']}")
    print(f"  Avg champion-to-champion distance: {clustering_results['avg_champ_to_champ_distance']:.3f}")
    print(f"  Avg champion-to-non-champion distance: {clustering_results['avg_champ_to_nonchamp_distance']:.3f}")
    print(f"  Clustering ratio: {clustering_results['clustering_ratio']:.3f}")
    print(f"  Interpretation: {clustering_results['interpretation']}")
    
    # 4. Model comparison
    print("\n--- Model Comparison ---")
    comparison = compare_model_embeddings(
        attention_model, mlp_model,
        team_features, player_features, labels,
        class_names=class_names,
        save_path='model_comparison.png'
    )
    
    print(f"\n  AttentionModel: {comparison['attention_model']['interpretation']}")
    print(f"  MLP Baseline: {comparison['mlp_baseline']['interpretation']}")
    
    print("\n" + "=" * 60)
    print("Interpretability analysis complete!")
    print("=" * 60)
