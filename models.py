"""
models.py - PyTorch Model Definitions for NBA Championship Prediction

Implements the exact architecture from the project proposal:
- Player embedding network (shared subnetwork)
- Attention pooling to combine player embeddings  
- Combined with team-level stats
- Fully connected layers for 6-class playoff outcome prediction

Classes:
  0 = Missed Playoffs
  1 = First Round Exit
  2 = Second Round Exit
  3 = Conference Finals
  4 = Finals Appearance
  5 = Champion
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class PlayerEmbeddingNetwork(nn.Module):
    """
    Shared subnetwork that takes ONE player's stats and produces a compact embedding.
    
    The same network is shared across all 8 players - it learns a general way to 
    summarize any player, not just specific ones.
    
    Args:
        player_stat_size: Number of stats per player (~10 from proposal)
        embedding_size: Output embedding dimension (default: 32)
        hidden_size: Hidden layer size (default: 64)
    """

    def __init__(self, player_stat_size: int, embedding_size: int = 32, hidden_size: int = 64):
        super(PlayerEmbeddingNetwork, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(player_stat_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, embedding_size),
            nn.ReLU()
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Xavier initialization for stable training."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, player_stats: torch.Tensor) -> torch.Tensor:
        """
        Args:
            player_stats: (batch_size, player_stat_size) for one player
        
        Returns:
            embedding: (batch_size, embedding_size) - the player's learned summary
        """
        return self.network(player_stats)


class AttentionPooling(nn.Module):
    """
    Attention mechanism that learns which players matter most for championship contention.
    
    Instead of treating all 8 players equally, this learns to weight them.
    For example, it might learn that the star player's embedding should count
    more than the 8th man's embedding when predicting championship odds.
    
    How it works:
    1. Each player embedding gets a score (how important is this player?)
    2. Scores are normalized to add up to 1 (using softmax)
    3. Player embeddings are combined using these weights
    
    Args:
        embedding_size: Size of player embeddings
    """

    def __init__(self, embedding_size: int):
        super(AttentionPooling, self).__init__()

        # Learns to score each player's importance
        self.attention = nn.Sequential(
            nn.Linear(embedding_size, embedding_size // 2),
            nn.Tanh(),
            nn.Linear(embedding_size // 2, 1)  # One score per player
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, player_embeddings: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            player_embeddings: (batch_size, num_players, embedding_size)
        
        Returns:
            combined: (batch_size, embedding_size) - weighted combination of all players
            weights: (batch_size, num_players) - attention weights (sum to 1)
        """
        # Get importance score for each player
        scores = self.attention(player_embeddings)        # (batch, num_players, 1)
        scores = scores.squeeze(-1)                        # (batch, num_players)

        # Turn scores into weights that add up to 1
        weights = F.softmax(scores, dim=1)                 # (batch, num_players)

        # Weighted combination: multiply each player's embedding by their weight and sum
        weights_expanded = weights.unsqueeze(-1)           # (batch, num_players, 1)
        combined = (player_embeddings * weights_expanded).sum(dim=1)  # (batch, embedding_size)

        return combined, weights


class AttentionModel(nn.Module):
    """
    Full model from the proposal with player embeddings + attention pooling.
    
    Architecture (as described in Section 4 of proposal):
    1. Each player's stats go through a shared embedding network
    2. Attention pooling combines the 8 player embeddings into one team-player summary
    3. That summary is joined with the team-level stats
    4. Everything goes through fully connected layers for the final prediction
    
    Visual:
    Player 1 stats -> [Embedding Net] -> emb1 \
    Player 2 stats -> [Embedding Net] -> emb2  |-> [Attention] -> team_player_summary
    ...                                        |                        |
    Player 8 stats -> [Embedding Net] -> emb8 /                         v
                                                                        
    Team stats ------------------------------------------------> [Combine] -> [FC Layers] -> Prediction
    
    Args:
        team_stat_size: Number of team-level stats (~15 from proposal)
        player_stat_size: Number of stats per player (~10 from proposal)
        num_players: Number of players per team (default: 8 from proposal)
        embedding_size: Player embedding dimension (default: 32)
        num_classes: Number of playoff outcomes (default: 6)
        hidden_size: Hidden layer size for classifier (default: 128)
        dropout_rate: Dropout probability (default: 0.3)
    """

    def __init__(self, 
                 team_stat_size: int, 
                 player_stat_size: int, 
                 num_players: int = 8,
                 embedding_size: int = 32, 
                 num_classes: int = 6,
                 hidden_size: int = 128,
                 dropout_rate: float = 0.3):
        super(AttentionModel, self).__init__()

        self.num_players = num_players
        self.player_stat_size = player_stat_size
        self.embedding_size = embedding_size

        # Shared network that embeds each player
        self.player_embedding = PlayerEmbeddingNetwork(player_stat_size, embedding_size)

        # Attention to weight and combine player embeddings
        self.attention = AttentionPooling(embedding_size)

        # Final prediction layers (takes team stats + attention output)
        combined_size = team_stat_size + embedding_size
        self.classifier = nn.Sequential(
            nn.Linear(combined_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size // 2, num_classes)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Xavier initialization for all linear layers."""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, 
                team_features: torch.Tensor, 
                player_features: torch.Tensor,
                return_attention: bool = False) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            team_features: (batch_size, team_stat_size) - team-level stats
            player_features: (batch_size, num_players * player_stat_size) - flattened player stats
            return_attention: If True, also return attention weights for interpretability
        
        Returns:
            logits: (batch_size, num_classes) - raw prediction scores
            attention_weights (optional): (batch_size, num_players) - how much each player mattered
        """
        batch_size = player_features.shape[0]

        # Reshape flat player features into (batch, num_players, stats_per_player)
        players = player_features.view(batch_size, self.num_players, self.player_stat_size)

        # Pass each player through the shared embedding network
        # We process all players in a loop (there's only 8, so it's fast)
        embeddings = []
        for i in range(self.num_players):
            emb = self.player_embedding(players[:, i, :])  # embed one player
            embeddings.append(emb)

        # Stack embeddings: (batch, num_players, embedding_size)
        embeddings = torch.stack(embeddings, dim=1)

        # Attention pooling: combine players with learned weights
        team_player_summary, attention_weights = self.attention(embeddings)

        # Combine team stats with the player summary
        combined = torch.cat([team_features, team_player_summary], dim=1)

        # Final prediction
        logits = self.classifier(combined)

        if return_attention:
            return logits, attention_weights
        return logits


class MLPBaseline(nn.Module):
    """
    Simple baseline model (plain neural network without player embeddings).
    
    Takes ALL the features (team stats + player stats) as one big flat vector
    and passes them through fully connected layers to make a prediction.
    
    This is one of the baselines mentioned in the proposal for comparison.
    
    Args:
        input_size: Total input size (team_stats + num_players * player_stats)
        num_classes: Number of playoff outcomes (default: 6)
        hidden_size: Hidden layer size (default: 128)
        dropout_rate: Dropout probability (default: 0.3)
    """

    def __init__(self, 
                 input_size: int, 
                 num_classes: int = 6, 
                 hidden_size: int = 128,
                 dropout_rate: float = 0.3):
        super(MLPBaseline, self).__init__()

        # Three layers that gradually shrink the data down to our 6 predictions
        self.network = nn.Sequential(
            # Layer 1: takes raw features, outputs hidden_size values
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            # Layer 2: shrinks from hidden_size to hidden_size/2
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            # Layer 3: final prediction - outputs one score per class (6 total)
            nn.Linear(hidden_size // 2, num_classes)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Xavier initialization for stable training."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, 
                team_features: torch.Tensor, 
                player_features: torch.Tensor,
                return_probs: bool = False) -> torch.Tensor:
        """
        Args:
            team_features: (batch_size, num_team_stats)
            player_features: (batch_size, num_players * num_player_stats)
            return_probs: If True, return softmax probabilities instead of logits
        
        Returns:
            output: (batch_size, num_classes) - logits or probabilities
        """
        # Combine team and player stats into one flat vector
        x = torch.cat([team_features, player_features], dim=1)
        logits = self.network(x)
        
        if return_probs:
            return F.softmax(logits, dim=1)
        return logits


# ============================================================================
# Utility Functions for Training and Evaluation (as mentioned in proposal)
# ============================================================================

def compute_class_weights(labels: torch.Tensor, num_classes: int = 6) -> torch.Tensor:
    """
    Compute class weights for imbalanced data (as mentioned in proposal Section 4).
    
    There is only one champion per year but many teams that miss the playoffs,
    so we need to weight the loss function accordingly.
    
    Args:
        labels: Tensor of class labels
        num_classes: Number of classes
    
    Returns:
        weights: (num_classes,) tensor of class weights
    """
    class_counts = torch.bincount(labels, minlength=num_classes).float()
    # Inverse frequency weighting
    weights = 1.0 / (class_counts + 1e-6)
    # Normalize so mean weight is 1
    weights = weights * num_classes / weights.sum()
    return weights


def get_attention_visualization(attention_weights: torch.Tensor, 
                                 player_names: Optional[list] = None) -> dict:
    """
    Extract attention weights for interpretability analysis (Section 4 of proposal).
    
    Args:
        attention_weights: (batch_size, num_players) tensor from model
        player_names: Optional list of player names for each position
    
    Returns:
        dict with attention statistics
    """
    if player_names is None:
        player_names = [f"Player {i+1}" for i in range(attention_weights.shape[1])]
    
    attention_weights = attention_weights.detach()
    mean_weights = attention_weights.mean(dim=0).cpu().numpy()
    std_weights = attention_weights.std(dim=0).cpu().numpy()
    
    return {
        "player_names": player_names,
        "mean_attention": mean_weights.tolist(),
        "std_attention": std_weights.tolist(),
        "most_important_idx": int(mean_weights.argmax()),
        "most_important_player": player_names[int(mean_weights.argmax())],
        "attention_entropy": -(attention_weights * torch.log(attention_weights + 1e-10)).sum(dim=1).mean().item()
    }


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    # Test configuration matching the proposal
    batch_size = 4
    team_stat_size = 15      # ~15 team-level stats from proposal
    player_stat_size = 10    # ~10 stats per player from proposal
    num_players = 8          # Top 8 players from proposal
    num_classes = 6          # 6 playoff outcomes from proposal
    
    # Create dummy data
    team_features = torch.randn(batch_size, team_stat_size)
    player_features = torch.randn(batch_size, num_players * player_stat_size)
    
    print("=" * 60)
    print("NBA Championship Prediction Model Test")
    print("=" * 60)
    print(f"\nTest Configuration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Team stats: {team_stat_size}")
    print(f"  Players per team: {num_players}")
    print(f"  Stats per player: {player_stat_size}")
    print(f"  Number of classes: {num_classes}")
    
    # Test MLPBaseline
    print("\n--- Testing MLPBaseline ---")
    input_size = team_stat_size + (num_players * player_stat_size)
    mlp = MLPBaseline(input_size, num_classes)
    output = mlp(team_features, player_features)
    probs = mlp(team_features, player_features, return_probs=True)
    print(f"  Logits shape: {output.shape}")
    print(f"  Probabilities shape: {probs.shape}")
    print(f"  Probabilities sum (should be 1.0): {probs[0].sum().item():.4f}")
    print(f"  ✓ MLPBaseline works!")
    
    # Test AttentionModel
    print("\n--- Testing AttentionModel ---")
    attn_model = AttentionModel(team_stat_size, player_stat_size, num_players)
    logits, attention_weights = attn_model(team_features, player_features, return_attention=True)
    print(f"  Logits shape: {logits.shape}")
    print(f"  Attention weights shape: {attention_weights.shape}")
    print(f"  Attention weights sum (should be 1.0): {attention_weights[0].sum().item():.4f}")
    
    # Test attention visualization
    viz = get_attention_visualization(attention_weights)
    print(f"  Most important player position: {viz['most_important_player']}")
    print(f"  Mean attention distribution: {[f'{w:.3f}' for w in viz['mean_attention']]}")
    print(f"  ✓ AttentionModel works!")
    
    # Test class weights
    print("\n--- Testing Class Weight Computation ---")
    dummy_labels = torch.tensor([0, 0, 0, 1, 1, 2, 3, 4, 5])  # Imbalanced
    weights = compute_class_weights(dummy_labels)
    print(f"  Class weights: {weights.numpy()}")
    print(f"  ✓ Class weight computation works!")
    
    # Parameter count
    print("\n--- Parameter Count ---")
    mlp_params = sum(p.numel() for p in mlp.parameters())
    attn_params = sum(p.numel() for p in attn_model.parameters())
    print(f"  MLPBaseline: {mlp_params:,} parameters")
    print(f"  AttentionModel: {attn_params:,} parameters")
    
    print("\n" + "=" * 60)
    print("All tests passed! Models ready for training.")
    print("=" * 60)
