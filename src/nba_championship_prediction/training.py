"""Training utilities for NBA championship prediction models."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .modeling import compute_class_weights


class NBADataset(Dataset):
    """Dataset of team-level and roster-level features for one study sample."""

    def __init__(
        self,
        team_features: np.ndarray,
        player_features: np.ndarray,
        labels: np.ndarray,
        seasons: Optional[np.ndarray] = None,
    ) -> None:
        self.team_features = torch.as_tensor(team_features, dtype=torch.float32)
        self.player_features = torch.as_tensor(player_features, dtype=torch.float32)
        self.labels = torch.as_tensor(labels, dtype=torch.long)
        self.seasons = seasons if seasons is not None else np.arange(len(labels))

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor | int]:
        return {
            "team_features": self.team_features[idx],
            "player_features": self.player_features[idx],
            "label": self.labels[idx],
            "season": self.seasons[idx],
        }


def compute_top_k_accuracy(
    predictions: np.ndarray,
    labels: np.ndarray,
    k: int = 2,
) -> float:
    """Return the fraction of samples whose true label appears in the top-k logits."""
    top_k_preds = np.argsort(predictions, axis=1)[:, -k:]
    correct = np.any(top_k_preds == labels.reshape(-1, 1), axis=1)
    return float(correct.mean())


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> Dict[str, float]:
    """Train a model for one epoch."""
    model.train()
    total_loss = 0.0
    all_preds: list[int] = []
    all_labels: list[int] = []

    for batch in dataloader:
        team_feat = batch["team_features"].to(device)
        player_feat = batch["player_features"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        logits = model(team_feat, player_feat)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        all_preds.extend(logits.argmax(dim=1).cpu().numpy().tolist())
        all_labels.extend(labels.cpu().numpy().tolist())

    return {
        "loss": total_loss / max(len(dataloader), 1),
        "accuracy": accuracy_score(all_labels, all_preds),
    }


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    """Evaluate a model on a dataloader."""
    model.eval()
    total_loss = 0.0
    all_preds: list[int] = []
    all_labels: list[int] = []
    all_probs: list[np.ndarray] = []

    with torch.no_grad():
        for batch in dataloader:
            team_feat = batch["team_features"].to(device)
            player_feat = batch["player_features"].to(device)
            labels = batch["label"].to(device)

            logits = model(team_feat, player_feat)
            loss = criterion(logits, labels)
            total_loss += loss.item()

            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = logits.argmax(dim=1).cpu().numpy()
            all_probs.extend(probs)
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    probs_array = np.asarray(all_probs)
    preds_array = np.asarray(all_preds)
    labels_array = np.asarray(all_labels)

    return {
        "loss": total_loss / max(len(dataloader), 1),
        "accuracy": accuracy_score(labels_array, preds_array),
        "f1": f1_score(labels_array, preds_array, average="weighted", zero_division=0),
        "top2_accuracy": compute_top_k_accuracy(probs_array, labels_array, k=2),
    }


def leave_one_season_out_cv(
    model_class: type[nn.Module],
    model_kwargs: dict,
    dataset: NBADataset,
    num_epochs: int = 50,
    batch_size: int = 32,
    lr: float = 0.001,
    val_split: float = 0.1,
    seed: int = 42,
    device: Optional[torch.device] = None,
) -> Dict:
    """Evaluate a model under leave-one-season-out cross-validation."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    unique_seasons = np.unique(dataset.seasons)
    fold_results = []

    print(f"Running leave-one-season-out CV across {len(unique_seasons)} seasons...")

    for test_season in tqdm(unique_seasons, desc="CV Folds"):
        train_indices = [index for index, season in enumerate(dataset.seasons) if season != test_season]
        test_indices = [index for index, season in enumerate(dataset.seasons) if season == test_season]

        full_train_dataset = torch.utils.data.Subset(dataset, train_indices)
        test_dataset = torch.utils.data.Subset(dataset, test_indices)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        n_val = min(max(1, int(len(full_train_dataset) * val_split)), max(len(full_train_dataset) - 1, 1))
        n_train = len(full_train_dataset) - n_val

        if n_train > 0 and n_val > 0:
            generator = torch.Generator().manual_seed(seed)
            train_dataset, val_dataset = torch.utils.data.random_split(
                full_train_dataset,
                [n_train, n_val],
                generator=generator,
            )
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
            train_label_indices = [train_indices[index] for index in train_dataset.indices]
        else:
            train_dataset = full_train_dataset
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = None
            train_label_indices = train_indices

        model = model_class(**model_kwargs).to(device)
        train_labels = dataset.labels[train_label_indices]
        class_weights = compute_class_weights(train_labels).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = optim.Adam(model.parameters(), lr=lr)

        best_val_acc = -1.0
        best_model_state = None

        for epoch in range(num_epochs):
            train_epoch(model, train_loader, criterion, optimizer, device)
            if val_loader is not None:
                val_metrics = evaluate(model, val_loader, criterion, device)
                if val_metrics["accuracy"] > best_val_acc:
                    best_val_acc = val_metrics["accuracy"]
                    best_model_state = copy.deepcopy(model.state_dict())
            elif epoch == num_epochs - 1:
                best_model_state = copy.deepcopy(model.state_dict())

        if best_model_state is not None:
            model.load_state_dict(best_model_state)

        test_metrics = evaluate(model, test_loader, criterion, device)
        test_metrics["test_season"] = int(test_season)
        fold_results.append(test_metrics)

    return {
        "accuracy": float(np.mean([result["accuracy"] for result in fold_results])),
        "accuracy_std": float(np.std([result["accuracy"] for result in fold_results])),
        "f1": float(np.mean([result["f1"] for result in fold_results])),
        "f1_std": float(np.std([result["f1"] for result in fold_results])),
        "top2_accuracy": float(np.mean([result["top2_accuracy"] for result in fold_results])),
        "top2_accuracy_std": float(np.std([result["top2_accuracy"] for result in fold_results])),
        "fold_results": fold_results,
    }


def train_final_model(
    model_class: type[nn.Module],
    model_kwargs: dict,
    dataset: NBADataset,
    num_epochs: int = 50,
    batch_size: int = 32,
    lr: float = 0.001,
    val_split: float = 0.2,
    seed: int = 42,
    device: Optional[torch.device] = None,
) -> Tuple[nn.Module, Dict]:
    """Train one final model with a reproducible validation split."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    n_val = max(1, int(len(dataset) * val_split))
    n_train = len(dataset) - n_val
    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset,
        [n_train, n_val],
        generator=generator,
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = model_class(**model_kwargs).to(device)
    train_labels = dataset.labels[train_dataset.indices]
    class_weights = compute_class_weights(train_labels).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=5)

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "val_f1": [],
        "val_top2": [],
    }
    best_val_acc = -1.0
    best_model_state = None
    patience = 10
    patience_counter = 0

    for epoch in range(num_epochs):
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate(model, val_loader, criterion, device)

        history["train_loss"].append(train_metrics["loss"])
        history["train_acc"].append(train_metrics["accuracy"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_acc"].append(val_metrics["accuracy"])
        history["val_f1"].append(val_metrics["f1"])
        history["val_top2"].append(val_metrics["top2_accuracy"])

        scheduler.step(val_metrics["accuracy"])

        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch + 1}/{num_epochs}: "
                f"Train Loss={train_metrics['loss']:.4f}, Acc={train_metrics['accuracy']:.4f} | "
                f"Val Loss={val_metrics['loss']:.4f}, Acc={val_metrics['accuracy']:.4f}, "
                f"F1={val_metrics['f1']:.4f}, Top2={val_metrics['top2_accuracy']:.4f}"
            )

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model, history


def save_results(results: Dict, filepath: str | Path) -> None:
    """Serialize nested metrics dictionaries to JSON."""

    def convert(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {key: convert(value) for key, value in obj.items()}
        if isinstance(obj, list):
            return [convert(item) for item in obj]
        return obj

    output_path = Path(filepath)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(convert(results), indent=2), encoding="utf-8")
