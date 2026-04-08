"""Neural architectures for NBA championship prediction."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_PLAYOFF_CLASSES = 6
PLAYOFF_CLASS_NAMES = (
    "Missed Playoffs",
    "First Round Exit",
    "Second Round Exit",
    "Conference Finals",
    "Finals Loss",
    "Champion",
)
PLAYER_SLOT_NAMES = (
    "Star",
    "Second Star",
    "Starter 3",
    "Starter 4",
    "Starter 5",
    "Sixth Man",
    "Bench 7",
    "Bench 8",
)


class PlayerEmbeddingNetwork(nn.Module):
    """Shared subnetwork that encodes one player's stat line."""

    def __init__(
        self,
        player_stat_size: int,
        embedding_size: int = 32,
        hidden_size: int = 64,
    ) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(player_stat_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, embedding_size),
            nn.ReLU(),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, player_stats: torch.Tensor) -> torch.Tensor:
        return self.network(player_stats)


class AttentionPooling(nn.Module):
    """Learned weighting over player embeddings."""

    def __init__(self, embedding_size: int) -> None:
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(embedding_size, embedding_size // 2),
            nn.Tanh(),
            nn.Linear(embedding_size // 2, 1),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, player_embeddings: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        scores = self.attention(player_embeddings).squeeze(-1)
        weights = F.softmax(scores, dim=1)
        combined = torch.sum(player_embeddings * weights.unsqueeze(-1), dim=1)
        return combined, weights


class AttentionModel(nn.Module):
    """Attention-based model that summarizes a roster before classification."""

    def __init__(
        self,
        team_stat_size: int,
        player_stat_size: int,
        num_players: int = 8,
        embedding_size: int = 32,
        num_classes: int = NUM_PLAYOFF_CLASSES,
        hidden_size: int = 128,
        dropout_rate: float = 0.3,
    ) -> None:
        super().__init__()
        self.num_players = num_players
        self.player_stat_size = player_stat_size
        self.embedding_size = embedding_size
        self.num_classes = num_classes

        self.player_embedding = PlayerEmbeddingNetwork(
            player_stat_size=player_stat_size,
            embedding_size=embedding_size,
        )
        self.attention = AttentionPooling(embedding_size)

        combined_size = team_stat_size + embedding_size
        self.classifier = nn.Sequential(
            nn.Linear(combined_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size // 2, num_classes),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.classifier.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def embed_players(self, player_features: torch.Tensor) -> torch.Tensor:
        """Vectorized player encoding used by both training and analysis."""
        batch_size = player_features.shape[0]
        players = player_features.reshape(batch_size * self.num_players, self.player_stat_size)
        embeddings = self.player_embedding(players)
        return embeddings.reshape(batch_size, self.num_players, self.embedding_size)

    def encode_team_representation(
        self,
        team_features: torch.Tensor,
        player_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        player_embeddings = self.embed_players(player_features)
        roster_summary, attention_weights = self.attention(player_embeddings)
        combined = torch.cat([team_features, roster_summary], dim=1)
        return combined, attention_weights

    def forward(
        self,
        team_features: torch.Tensor,
        player_features: torch.Tensor,
        return_attention: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        combined, attention_weights = self.encode_team_representation(
            team_features=team_features,
            player_features=player_features,
        )
        logits = self.classifier(combined)
        if return_attention:
            return logits, attention_weights
        return logits


class MLPBaseline(nn.Module):
    """Flat-feature baseline for comparison against the attention model."""

    def __init__(
        self,
        input_size: int,
        num_classes: int = NUM_PLAYOFF_CLASSES,
        hidden_size: int = 128,
        dropout_rate: float = 0.3,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )
        self.classifier = nn.Linear(hidden_size // 2, num_classes)
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def encode_features(
        self,
        team_features: torch.Tensor,
        player_features: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.cat([team_features, player_features], dim=1)
        return self.encoder(x)

    def forward(
        self,
        team_features: torch.Tensor,
        player_features: torch.Tensor,
        return_probs: bool = False,
    ) -> torch.Tensor:
        encoded = self.encode_features(team_features, player_features)
        logits = self.classifier(encoded)
        if return_probs:
            return F.softmax(logits, dim=1)
        return logits


def compute_class_weights(
    labels: torch.Tensor,
    num_classes: int = NUM_PLAYOFF_CLASSES,
) -> torch.Tensor:
    """Inverse-frequency loss weights with unit mean."""
    class_counts = torch.bincount(labels, minlength=num_classes).float()
    weights = 1.0 / (class_counts + 1e-6)
    return weights * num_classes / weights.sum()


def summarize_attention(
    attention_weights: torch.Tensor,
    player_names: Optional[list[str]] = None,
) -> dict:
    """Aggregate attention weights for reporting."""
    if player_names is None:
        player_names = [f"Player {index + 1}" for index in range(attention_weights.shape[1])]

    detached = attention_weights.detach()
    mean_weights = detached.mean(dim=0).cpu().numpy()
    std_weights = detached.std(dim=0).cpu().numpy()

    return {
        "player_names": player_names,
        "mean_attention": mean_weights.tolist(),
        "std_attention": std_weights.tolist(),
        "most_important_idx": int(mean_weights.argmax()),
        "most_important_player": player_names[int(mean_weights.argmax())],
        "attention_entropy": (
            -(detached * torch.log(detached + 1e-10)).sum(dim=1).mean().item()
        ),
    }
