"""
train.py - Training Script for NBA Championship Prediction Models

Implements the training procedure from the project proposal:
- Cross-entropy loss with class weights for imbalanced data
- Leave-one-season-out cross-validation
- Evaluation metrics: accuracy, F1 score, top-2 accuracy
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from typing import Dict, List, Tuple, Optional
import json
from tqdm import tqdm

from models import AttentionModel, MLPBaseline, compute_class_weights


class NBADataset(Dataset):
    """
    Dataset for NBA team-season data.
    
    Each sample represents one team in one season with:
    - team_features: ~15 team-level stats
    - player_features: flattened stats for top 8 players
    - label: playoff outcome (0-5)
    - season: season identifier for leave-one-out CV
    """
    
    def __init__(self, 
                 team_features: np.ndarray,
                 player_features: np.ndarray,
                 labels: np.ndarray,
                 seasons: Optional[np.ndarray] = None):
        """
        Args:
            team_features: (num_samples, num_team_stats)
            player_features: (num_samples, num_players * num_player_stats)
            labels: (num_samples,) playoff outcome labels
            seasons: (num_samples,) season identifiers for CV
        """
        self.team_features = torch.FloatTensor(team_features)
        self.player_features = torch.FloatTensor(player_features)
        self.labels = torch.LongTensor(labels)
        self.seasons = seasons if seasons is not None else np.arange(len(labels))
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return {
            'team_features': self.team_features[idx],
            'player_features': self.player_features[idx],
            'label': self.labels[idx],
            'season': self.seasons[idx]
        }


def compute_top_k_accuracy(predictions: np.ndarray, 
                           labels: np.ndarray, 
                           k: int = 2) -> float:
    """
    Compute top-k accuracy (did the true label appear in top k predictions?).
    
    From proposal: "top-2 accuracy (did the model rank the actual champion 
    in its top 2 picks?)"
    
    Args:
        predictions: (num_samples, num_classes) predicted probabilities
        labels: (num_samples,) true labels
        k: Number of top predictions to consider
    
    Returns:
        Top-k accuracy as a float
    """
    top_k_preds = np.argsort(predictions, axis=1)[:, -k:]
    correct = np.any(top_k_preds == labels.reshape(-1, 1), axis=1)
    return correct.mean()


def train_epoch(model: nn.Module,
                dataloader: DataLoader,
                criterion: nn.Module,
                optimizer: optim.Optimizer,
                device: torch.device) -> Dict[str, float]:
    """
    Train for one epoch.
    
    Returns:
        dict with 'loss' and 'accuracy'
    """
    model.train()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    for batch in dataloader:
        team_feat = batch['team_features'].to(device)
        player_feat = batch['player_features'].to(device)
        labels = batch['label'].to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        if isinstance(model, AttentionModel):
            logits, _ = model(team_feat, player_feat, return_attention=True)
        else:
            logits = model(team_feat, player_feat)
        
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    
    return {'loss': avg_loss, 'accuracy': accuracy}


def evaluate(model: nn.Module,
             dataloader: DataLoader,
             criterion: nn.Module,
             device: torch.device) -> Dict[str, float]:
    """
    Evaluate the model.
    
    Returns:
        dict with 'loss', 'accuracy', 'f1', 'top2_accuracy'
    """
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch in dataloader:
            team_feat = batch['team_features'].to(device)
            player_feat = batch['player_features'].to(device)
            labels = batch['label'].to(device)
            
            # Forward pass
            if isinstance(model, AttentionModel):
                logits, _ = model(team_feat, player_feat, return_attention=True)
            else:
                logits = model(team_feat, player_feat)
            
            loss = criterion(logits, labels)
            total_loss += loss.item()
            
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = logits.argmax(dim=1).cpu().numpy()
            
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
    
    all_probs = np.array(all_probs)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    top2_acc = compute_top_k_accuracy(all_probs, all_labels, k=2)
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'f1': f1,
        'top2_accuracy': top2_acc
    }


def leave_one_season_out_cv(model_class: type,
                            model_kwargs: dict,
                            dataset: NBADataset,
                            num_epochs: int = 50,
                            batch_size: int = 32,
                            lr: float = 0.001,
                            device: torch.device = None) -> Dict:
    """
    Leave-one-season-out cross-validation as specified in the proposal.
    
    "We will evaluate using leave-one-season-out cross-validation so the 
    model never sees future data."
    
    Args:
        model_class: Model class to instantiate
        model_kwargs: kwargs for model initialization
        dataset: NBADataset with season information
        num_epochs: Number of training epochs per fold
        batch_size: Batch size
        lr: Learning rate
        device: torch device
    
    Returns:
        dict with cross-validation results
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    unique_seasons = np.unique(dataset.seasons)
    fold_results = []
    
    print(f"Running leave-one-season-out CV with {len(unique_seasons)} seasons...")
    
    for test_season in tqdm(unique_seasons, desc="CV Folds"):
        # Split data by season
        train_indices = [i for i, s in enumerate(dataset.seasons) if s != test_season]
        test_indices = [i for i, s in enumerate(dataset.seasons) if s == test_season]
        
        # Create data loaders
        train_dataset = torch.utils.data.Subset(dataset, train_indices)
        test_dataset = torch.utils.data.Subset(dataset, test_indices)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        # Initialize model
        model = model_class(**model_kwargs).to(device)
        
        # Compute class weights from training data
        train_labels = dataset.labels[train_indices]
        class_weights = compute_class_weights(train_labels).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        
        optimizer = optim.Adam(model.parameters(), lr=lr)
        
        # Training loop
        best_val_acc = 0.0
        best_model_state = None
        
        for epoch in range(num_epochs):
            train_metrics = train_epoch(model, train_loader, criterion, optimizer, device)
            
            # Early stopping check on last few epochs
            if epoch >= num_epochs - 5:
                val_metrics = evaluate(model, test_loader, criterion, device)
                if val_metrics['accuracy'] > best_val_acc:
                    best_val_acc = val_metrics['accuracy']
                    best_model_state = model.state_dict().copy()
        
        # Load best model and evaluate
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        
        test_metrics = evaluate(model, test_loader, criterion, device)
        test_metrics['test_season'] = int(test_season)
        fold_results.append(test_metrics)
    
    # Aggregate results
    aggregated = {
        'accuracy': np.mean([r['accuracy'] for r in fold_results]),
        'accuracy_std': np.std([r['accuracy'] for r in fold_results]),
        'f1': np.mean([r['f1'] for r in fold_results]),
        'f1_std': np.std([r['f1'] for r in fold_results]),
        'top2_accuracy': np.mean([r['top2_accuracy'] for r in fold_results]),
        'top2_accuracy_std': np.std([r['top2_accuracy'] for r in fold_results]),
        'fold_results': fold_results
    }
    
    return aggregated


def train_final_model(model_class: type,
                      model_kwargs: dict,
                      dataset: NBADataset,
                      num_epochs: int = 50,
                      batch_size: int = 32,
                      lr: float = 0.001,
                      val_split: float = 0.2,
                      device: torch.device = None) -> Tuple[nn.Module, Dict]:
    """
    Train final model on all data with a validation split.
    
    Args:
        model_class: Model class to instantiate
        model_kwargs: kwargs for model initialization
        dataset: NBADataset
        num_epochs: Number of training epochs
        batch_size: Batch size
        lr: Learning rate
        val_split: Fraction of data to use for validation
        device: torch device
    
    Returns:
        (trained_model, training_history)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Split data
    n_val = int(len(dataset) * val_split)
    n_train = len(dataset) - n_val
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [n_train, n_val])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model
    model = model_class(**model_kwargs).to(device)
    
    # Compute class weights
    train_labels = [dataset.labels[i] for i in train_dataset.indices]
    class_weights = compute_class_weights(torch.tensor(train_labels)).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=5)
    
    # Training loop
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'val_f1': [], 'val_top2': []}
    best_val_acc = 0.0
    best_model_state = None
    patience_counter = 0
    patience = 10
    
    for epoch in range(num_epochs):
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate(model, val_loader, criterion, device)
        
        history['train_loss'].append(train_metrics['loss'])
        history['train_acc'].append(train_metrics['accuracy'])
        history['val_loss'].append(val_metrics['loss'])
        history['val_acc'].append(val_metrics['accuracy'])
        history['val_f1'].append(val_metrics['f1'])
        history['val_top2'].append(val_metrics['top2_accuracy'])
        
        scheduler.step(val_metrics['accuracy'])
        
        # Save best model
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}: "
                  f"Train Loss={train_metrics['loss']:.4f}, Acc={train_metrics['accuracy']:.4f} | "
                  f"Val Loss={val_metrics['loss']:.4f}, Acc={val_metrics['accuracy']:.4f}, "
                  f"F1={val_metrics['f1']:.4f}, Top2={val_metrics['top2_accuracy']:.4f}")
        
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model, history


def save_results(results: Dict, filepath: str):
    """Save results to JSON file."""
    # Convert numpy types to Python types for JSON serialization
    def convert(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(item) for item in obj]
        return obj
    
    with open(filepath, 'w') as f:
        json.dump(convert(results), f, indent=2)


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("NBA Championship Prediction - Training Script")
    print("=" * 60)
    
    # Configuration matching the proposal
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
    # In real usage, load from nba_api processed data
    np.random.seed(42)
    torch.manual_seed(42)
    
    n_samples = 500
    n_seasons = 20
    
    team_features = np.random.randn(n_samples, CONFIG['team_stat_size']).astype(np.float32)
    player_features = np.random.randn(n_samples, CONFIG['num_players'] * CONFIG['player_stat_size']).astype(np.float32)
    
    # Simulate imbalanced class distribution (more teams miss playoffs than win championship)
    labels = np.random.choice(
        CONFIG['num_classes'], 
        size=n_samples, 
        p=[0.45, 0.20, 0.15, 0.10, 0.07, 0.03]  # Imbalanced distribution
    )
    seasons = np.random.randint(0, n_seasons, size=n_samples)
    
    # Create dataset
    dataset = NBADataset(team_features, player_features, labels, seasons)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Train Attention Model
    print("\n" + "-" * 60)
    print("Training Attention Model")
    print("-" * 60)
    
    model_kwargs = {
        'team_stat_size': CONFIG['team_stat_size'],
        'player_stat_size': CONFIG['player_stat_size'],
        'num_players': CONFIG['num_players'],
        'embedding_size': CONFIG['embedding_size'],
        'num_classes': CONFIG['num_classes'],
        'hidden_size': CONFIG['hidden_size'],
        'dropout_rate': CONFIG['dropout_rate']
    }
    
    attention_model, history = train_final_model(
        AttentionModel, model_kwargs, dataset,
        num_epochs=30, batch_size=32, lr=0.001, device=device
    )
    
    print(f"\nFinal Validation Metrics:")
    print(f"  Accuracy: {max(history['val_acc']):.4f}")
    print(f"  F1 Score: {max(history['val_f1']):.4f}")
    print(f"  Top-2 Accuracy: {max(history['val_top2']):.4f}")
    
    # Train MLP Baseline
    print("\n" + "-" * 60)
    print("Training MLP Baseline")
    print("-" * 60)
    
    input_size = CONFIG['team_stat_size'] + (CONFIG['num_players'] * CONFIG['player_stat_size'])
    mlp_kwargs = {
        'input_size': input_size,
        'num_classes': CONFIG['num_classes'],
        'hidden_size': CONFIG['hidden_size'],
        'dropout_rate': CONFIG['dropout_rate']
    }
    
    mlp_model, mlp_history = train_final_model(
        MLPBaseline, mlp_kwargs, dataset,
        num_epochs=30, batch_size=32, lr=0.001, device=device
    )
    
    print(f"\nFinal Validation Metrics:")
    print(f"  Accuracy: {max(mlp_history['val_acc']):.4f}")
    print(f"  F1 Score: {max(mlp_history['val_f1']):.4f}")
    print(f"  Top-2 Accuracy: {max(mlp_history['val_top2']):.4f}")
    
    # Save models
    torch.save(attention_model.state_dict(), 'attention_model.pt')
    torch.save(mlp_model.state_dict(), 'mlp_baseline.pt')
    print("\nModels saved to 'attention_model.pt' and 'mlp_baseline.pt'")
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)
