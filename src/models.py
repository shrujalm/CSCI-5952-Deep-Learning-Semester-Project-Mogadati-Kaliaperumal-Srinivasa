"""
models.py - PyTorch model definitions for NBA Championship Prediction

Contains 2 models:
1. MLPBaseline - A simple fully connected neural network (our baseline)
2. AttentionModel - The fancier model with player embeddings + attention pooling

Both models take in team stats + player stats and predict one of 6 outcomes:
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


class MLPBaseline(nn.Module):
    """
    Simple baseline model.

    Takes ALL the features (team stats + player stats) as one big flat vector
    and passes them through a few fully connected layers to make a prediction.

    Think of it like: stats -> layer 1 -> layer 2 -> prediction
    """

    def __init__(self, input_size, num_classes=6, hidden_size=128):
        super(MLPBaseline, self).__init__()

        # Three layers that gradually shrink the data down to our 6 predictions
        self.network = nn.Sequential(
            # Layer 1: takes raw features, outputs 128 values
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),            # activation function (keeps only positive values)
            nn.Dropout(0.3),      # randomly turns off 30% of neurons during training to prevent overfitting

            # Layer 2: shrinks from 128 to 64
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.3),

            # Layer 3: final prediction - outputs one score per class (6 total)
            nn.Linear(hidden_size // 2, num_classes)
        )

    def forward(self, team_features, player_features):
        """
        team_features: shape (batch_size, num_team_stats)
        player_features: shape (batch_size, num_players * num_player_stats)

        We just smash them together into one big vector and feed it through.
        """
        # Combine team and player stats into one flat vector
        x = torch.cat([team_features, player_features], dim=1)
        return self.network(x)


class PlayerEmbeddingNetwork(nn.Module):
    """
    A small neural network that takes ONE player's stats and compresses them
    into a compact "embedding" (a learned summary of that player).

    The same network is shared across all players - so it learns a general
    way to summarize any player, not just specific ones.
    """

    def __init__(self, player_stat_size, embedding_size=32):
        super(PlayerEmbeddingNetwork, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(player_stat_size, 64),
            nn.ReLU(),
            nn.Linear(64, embedding_size),
            nn.ReLU()
        )

    def forward(self, player_stats):
        """
        player_stats: shape (batch_size, num_player_stats) for one player
        Returns: shape (batch_size, embedding_size) - the player's learned summary
        """
        return self.network(player_stats)


class AttentionPooling(nn.Module):
    """
    Attention mechanism that figures out which players matter most.

    Instead of treating all 8 players equally, this learns to weight them.
    For example, it might learn that the star player's embedding should count
    3x more than the 8th man's embedding when predicting championship odds.

    How it works:
    1. Each player embedding gets a score (how important is this player?)
    2. Scores are normalized to add up to 1 (using softmax)
    3. Player embeddings are combined using these weights
    """

    def __init__(self, embedding_size):
        super(AttentionPooling, self).__init__()

        # This learns to score each player's importance
        self.attention = nn.Sequential(
            nn.Linear(embedding_size, embedding_size // 2),
            nn.Tanh(),
            nn.Linear(embedding_size // 2, 1)  # outputs one score per player
        )

    def forward(self, player_embeddings):
        """
        player_embeddings: shape (batch_size, num_players, embedding_size)

        Returns:
            combined: shape (batch_size, embedding_size) - weighted combination of all players
            weights: shape (batch_size, num_players) - how much each player mattered
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
    The full model from the proposal.

    Architecture:
    1. Each player's stats go through a shared embedding network
    2. Attention pooling combines the 8 player embeddings into one team-player summary
    3. That summary is joined with the team-level stats
    4. Everything goes through fully connected layers for the final prediction

    Visual:
    Player 1 stats -> [Embedding Net] -> emb1 \
    Player 2 stats -> [Embedding Net] -> emb2  |-> [Attention] -> team_player_summary
    ...                                        |                        |
    Player 8 stats -> [Embedding Net] -> emb8 /                         |
                                                                        v
    Team stats ------------------------------------------------> [Combine] -> [FC Layers] -> Prediction
    """

    def __init__(self, team_stat_size, player_stat_size, num_players=8,
                 embedding_size=32, num_classes=6):
        super(AttentionModel, self).__init__()

        self.num_players = num_players
        self.player_stat_size = player_stat_size

        # Shared network that embeds each player
        self.player_embedding = PlayerEmbeddingNetwork(player_stat_size, embedding_size)

        # Attention to weight and combine player embeddings
        self.attention = AttentionPooling(embedding_size)

        # Final prediction layers (takes team stats + attention output)
        combined_size = team_stat_size + embedding_size
        self.classifier = nn.Sequential(
            nn.Linear(combined_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, team_features, player_features):
        """
        team_features: shape (batch_size, team_stat_size)
        player_features: shape (batch_size, num_players * player_stat_size)
            - player stats are flattened, so we reshape them first

        Returns:
            logits: shape (batch_size, num_classes) - raw prediction scores
            attention_weights: shape (batch_size, num_players) - how much each player mattered
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

        return logits, attention_weights
