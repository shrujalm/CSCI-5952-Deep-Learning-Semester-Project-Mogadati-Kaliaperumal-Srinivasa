"""
models_improved.py - Enhanced PyTorch Models for NBA Championship Prediction

Improvements over original:
1. Vectorized operations (no Python loops in forward pass)
2. BatchNorm and LayerNorm for more stable training
3. Xavier weight initialization
4. Configurable architecture
5. Positional encoding for player ordering
6. Better gradient flow
7. Optional probability output
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPBaselineV2(nn.Module):
    """
    Improved MLP Baseline with BatchNorm and configurable architecture.
    
    Args:
        input_size: Total size of input features
        num_classes: Number of output classes (default: 6)
        hidden_sizes: List of hidden layer sizes (default: [256, 128, 64])
        dropout_rate: Dropout probability (default: 0.3)
        use_batch_norm: Whether to use BatchNorm (default: True)
    """

    def __init__(self, input_size, num_classes=6, hidden_sizes=[256, 128, 64], 
                 dropout_rate=0.3, use_batch_norm=True):
        super(MLPBaselineV2, self).__init__()
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, num_classes))
        self.network = nn.Sequential(*layers)
        self._init_weights()
    
    def _init_weights(self):
        """Xavier initialization for better convergence."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, team_features, player_features, return_probs=False):
        """
        Args:
            team_features: (batch_size, num_team_stats)
            player_features: (batch_size, num_players * num_player_stats)
            return_probs: If True, return softmax probabilities
        
        Returns:
            logits: (batch_size, num_classes) or probabilities if return_probs=True
        """
        x = torch.cat([team_features, player_features], dim=1)
        logits = self.network(x)
        if return_probs:
            return F.softmax(logits, dim=1)
        return logits


class PlayerEmbeddingNetworkV2(nn.Module):
    """
    Improved player embedding with LayerNorm for stable embeddings.
    
    Args:
        player_stat_size: Number of stats per player
        embedding_size: Output embedding dimension (default: 32)
        use_layer_norm: Whether to use LayerNorm (default: True)
    """

    def __init__(self, player_stat_size, embedding_size=32, use_layer_norm=True):
        super(PlayerEmbeddingNetworkV2, self).__init__()
        
        self.fc1 = nn.Linear(player_stat_size, 64)
        self.fc2 = nn.Linear(64, embedding_size)
        self.use_layer_norm = use_layer_norm
        
        if use_layer_norm:
            self.ln1 = nn.LayerNorm(64)
            self.ln2 = nn.LayerNorm(embedding_size)
        
        self._init_weights()
    
    def _init_weights(self):
        for m in [self.fc1, self.fc2]:
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, player_stats):
        """
        Args:
            player_stats: (batch_size, num_player_stats)
        
        Returns:
            embedding: (batch_size, embedding_size)
        """
        x = self.fc1(player_stats)
        if self.use_layer_norm:
            x = self.ln1(x)
        x = F.relu(x)
        x = self.fc2(x)
        if self.use_layer_norm:
            x = self.ln2(x)
        x = F.relu(x)
        return x


class AttentionPoolingV2(nn.Module):
    """
    Improved attention pooling with better numerical stability.
    
    Args:
        embedding_size: Size of player embeddings
        temperature: Softmax temperature for sharper/softer attention (default: 1.0)
    """

    def __init__(self, embedding_size, temperature=1.0):
        super(AttentionPoolingV2, self).__init__()
        self.temperature = temperature
        self.embedding_size = embedding_size
        
        self.key_proj = nn.Linear(embedding_size, embedding_size)
        self.value_proj = nn.Linear(embedding_size, embedding_size)
        self.query = nn.Parameter(torch.randn(1, 1, embedding_size))
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.xavier_uniform_(self.key_proj.weight)
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.normal_(self.query, std=0.02)

    def forward(self, player_embeddings):
        """
        Args:
            player_embeddings: (batch_size, num_players, embedding_size)
        
        Returns:
            combined: (batch_size, embedding_size) - weighted player summary
            weights: (batch_size, num_players) - attention weights per player
        """
        batch_size = player_embeddings.size(0)
        
        keys = self.key_proj(player_embeddings)
        values = self.value_proj(player_embeddings)
        
        query = self.query.expand(batch_size, -1, -1)
        scores = torch.bmm(query, keys.transpose(1, 2)) / self.temperature
        scores = scores.squeeze(1)
        
        weights = F.softmax(scores, dim=1)
        weights_expanded = weights.unsqueeze(1)
        combined = torch.bmm(weights_expanded, values).squeeze(1)
        
        return combined, weights


class AttentionModelV2(nn.Module):
    """
    Improved attention-based model with vectorized operations.
    
    Key improvements:
    - No Python loops in forward pass (fully vectorized)
    - Positional encoding for player ordering
    - BatchNorm in classifier
    - Xavier initialization
    
    Args:
        team_stat_size: Number of team-level stats
        player_stat_size: Number of stats per player
        num_players: Number of players per team (default: 8)
        embedding_size: Player embedding dimension (default: 32)
        num_classes: Number of output classes (default: 6)
        use_positional_encoding: Add learnable positional encoding (default: True)
        dropout_rate: Dropout probability (default: 0.3)
    """

    def __init__(self, team_stat_size, player_stat_size, num_players=8,
                 embedding_size=32, num_classes=6, 
                 use_positional_encoding=True, dropout_rate=0.3):
        super(AttentionModelV2, self).__init__()
        
        self.num_players = num_players
        self.player_stat_size = player_stat_size
        self.embedding_size = embedding_size
        self.use_positional_encoding = use_positional_encoding
        
        # Shared player embedding network
        self.player_embedding = PlayerEmbeddingNetworkV2(player_stat_size, embedding_size)
        
        # Positional encoding
        if use_positional_encoding:
            self.pos_encoding = nn.Parameter(
                torch.randn(1, num_players, embedding_size) * 0.02
            )
        
        # Attention pooling
        self.attention = AttentionPoolingV2(embedding_size)
        
        # Classifier with BatchNorm
        combined_size = team_stat_size + embedding_size
        self.fc1 = nn.Linear(combined_size, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, num_classes)
        self.dropout = nn.Dropout(dropout_rate)
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, team_features, player_features, return_attention=False):
        """
        Vectorized forward pass.
        
        Args:
            team_features: (batch_size, team_stat_size)
            player_features: (batch_size, num_players * player_stat_size)
            return_attention: If True, also return attention weights
        
        Returns:
            logits: (batch_size, num_classes)
            attention_weights (optional): (batch_size, num_players)
        """
        batch_size = player_features.size(0)
        
        # Vectorized player embedding: reshape and process all at once
        players = player_features.view(batch_size * self.num_players, self.player_stat_size)
        embeddings = self.player_embedding(players)
        embeddings = embeddings.view(batch_size, self.num_players, self.embedding_size)
        
        # Add positional encoding
        if self.use_positional_encoding:
            embeddings = embeddings + self.pos_encoding
        
        # Attention pooling
        team_player_summary, attention_weights = self.attention(embeddings)
        
        # Combine and classify
        combined = torch.cat([team_features, team_player_summary], dim=1)
        
        x = self.fc1(combined)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout(x)
        logits = self.fc3(x)
        
        if return_attention:
            return logits, attention_weights
        return logits


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    # Test configuration
    batch_size = 4
    team_stat_size = 20
    player_stat_size = 15
    num_players = 8
    num_classes = 6
    
    # Create dummy data
    team_features = torch.randn(batch_size, team_stat_size)
    player_features = torch.randn(batch_size, num_players * player_stat_size)
    
    # Test MLPBaselineV2
    input_size = team_stat_size + (num_players * player_stat_size)
    mlp = MLPBaselineV2(input_size, num_classes)
    logits = mlp(team_features, player_features)
    print(f"MLP Output: {logits.shape}")
    
    # Test AttentionModelV2
    attn = AttentionModelV2(team_stat_size, player_stat_size, num_players)
    logits, weights = attn(team_features, player_features, return_attention=True)
    print(f"Attention Output: {logits.shape}")
    print(f"Attention Weights: {weights.shape}, Sum: {weights[0].sum().item():.4f}")
    
    # Count parameters
    mlp_params = sum(p.numel() for p in mlp.parameters())
    attn_params = sum(p.numel() for p in attn.parameters())
    print(f"\nMLP Parameters: {mlp_params:,}")
    print(f"Attention Model Parameters: {attn_params:,}")
