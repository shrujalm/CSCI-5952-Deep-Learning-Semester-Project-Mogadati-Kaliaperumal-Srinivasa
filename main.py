"""
main.py - Main Entry Point for NBA Championship Prediction

This script orchestrates the full pipeline:
1. Data Collection (from nba_api)
2. Model Training (AttentionModel + MLP Baseline)
3. Evaluation (Leave-one-season-out CV, accuracy, F1, top-2 accuracy)
4. Interpretability Analysis (attention weights, t-SNE)

Usage:
    python main.py --mode full
    python main.py --mode train
    python main.py --mode evaluate
    python main.py --mode interpret
"""

import argparse
import torch
import numpy as np
import json
import os
from datetime import datetime

from data_collection import build_dataset, prepare_tensors, SEASONS
from train import NBADataset, leave_one_season_out_cv, train_final_model, save_results
from models import AttentionModel, MLPBaseline
from interpretability import (
    visualize_attention_weights, 
    visualize_tsne_embeddings,
    analyze_champion_clustering,
    compare_model_embeddings
)


def parse_args():
    parser = argparse.ArgumentParser(
        description='NBA Championship Prediction using Deep Learning'
    )
    parser.add_argument(
        '--mode', 
        type=str, 
        default='full',
        choices=['full', 'data', 'train', 'evaluate', 'interpret'],
        help='Mode to run (default: full)'
    )
    parser.add_argument(
        '--seasons',
        type=int,
        default=20,
        help='Number of seasons to include (default: 20)'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=50,
        help='Number of training epochs (default: 50)'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Batch size (default: 32)'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=0.001,
        help='Learning rate (default: 0.001)'
    )
    parser.add_argument(
        '--embedding_size',
        type=int,
        default=32,
        help='Player embedding size (default: 32)'
    )
    parser.add_argument(
        '--hidden_size',
        type=int,
        default=128,
        help='Hidden layer size (default: 128)'
    )
    parser.add_argument(
        '--dropout',
        type=float,
        default=0.3,
        help='Dropout rate (default: 0.3)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='outputs',
        help='Output directory (default: outputs)'
    )
    return parser.parse_args()


def setup_directories(output_dir: str):
    """Create output directories."""
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/models", exist_ok=True)
    os.makedirs(f"{output_dir}/figures", exist_ok=True)
    os.makedirs(f"{output_dir}/results", exist_ok=True)


def run_data_collection(args):
    """Run data collection pipeline."""
    print("\n" + "=" * 60)
    print("STEP 1: Data Collection")
    print("=" * 60)
    
    seasons = SEASONS[-args.seasons:]
    
    dataset = build_dataset(
        seasons=seasons,
        top_n_players=8,
        save_path=f"{args.output_dir}/nba_dataset.csv"
    )
    
    tensors = prepare_tensors(dataset)
    np.savez(f"{args.output_dir}/nba_tensors.npz", **tensors)
    
    print("\n✓ Data collection complete!")
    return tensors


def run_training(args, tensors):
    """Run model training."""
    print("\n" + "=" * 60)
    print("STEP 2: Model Training")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create dataset
    dataset = NBADataset(
        tensors['team_features'],
        tensors['player_features'],
        tensors['labels'],
        tensors['seasons']
    )
    
    # Model configuration
    team_stat_size = tensors['team_features'].shape[1]
    player_stat_size = tensors['player_features'].shape[1] // 8
    
    model_kwargs = {
        'team_stat_size': team_stat_size,
        'player_stat_size': player_stat_size,
        'num_players': 8,
        'embedding_size': args.embedding_size,
        'num_classes': 6,
        'hidden_size': args.hidden_size,
        'dropout_rate': args.dropout
    }
    
    # Train Attention Model
    print("\n--- Training AttentionModel ---")
    attention_model, history = train_final_model(
        AttentionModel,
        model_kwargs,
        dataset,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=device
    )
    
    # Save model
    torch.save(attention_model.state_dict(), f"{args.output_dir}/models/attention_model.pt")
    torch.save(history, f"{args.output_dir}/results/attention_history.pt")
    
    print(f"\nAttentionModel Best Validation:")
    print(f"  Accuracy: {max(history['val_acc']):.4f}")
    print(f"  F1 Score: {max(history['val_f1']):.4f}")
    print(f"  Top-2 Accuracy: {max(history['val_top2']):.4f}")
    
    # Train MLP Baseline
    print("\n--- Training MLP Baseline ---")
    input_size = team_stat_size + (8 * player_stat_size)
    mlp_kwargs = {
        'input_size': input_size,
        'num_classes': 6,
        'hidden_size': args.hidden_size,
        'dropout_rate': args.dropout
    }
    
    mlp_model, mlp_history = train_final_model(
        MLPBaseline,
        mlp_kwargs,
        dataset,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=device
    )
    
    # Save model
    torch.save(mlp_model.state_dict(), f"{args.output_dir}/models/mlp_baseline.pt")
    torch.save(mlp_history, f"{args.output_dir}/results/mlp_history.pt")
    
    print(f"\nMLP Baseline Best Validation:")
    print(f"  Accuracy: {max(mlp_history['val_acc']):.4f}")
    print(f"  F1 Score: {max(mlp_history['val_f1']):.4f}")
    print(f"  Top-2 Accuracy: {max(mlp_history['val_top2']):.4f}")
    
    print("\n✓ Training complete!")
    return attention_model, mlp_model


def run_evaluation(args, tensors):
    """Run leave-one-season-out cross-validation."""
    print("\n" + "=" * 60)
    print("STEP 3: Cross-Validation Evaluation")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create dataset
    dataset = NBADataset(
        tensors['team_features'],
        tensors['player_features'],
        tensors['labels'],
        tensors['seasons']
    )
    
    team_stat_size = tensors['team_features'].shape[1]
    player_stat_size = tensors['player_features'].shape[1] // 8
    
    # Attention Model CV
    print("\n--- AttentionModel Cross-Validation ---")
    model_kwargs = {
        'team_stat_size': team_stat_size,
        'player_stat_size': player_stat_size,
        'num_players': 8,
        'embedding_size': args.embedding_size,
        'num_classes': 6,
        'hidden_size': args.hidden_size,
        'dropout_rate': args.dropout
    }
    
    attention_cv_results = leave_one_season_out_cv(
        AttentionModel,
        model_kwargs,
        dataset,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=device
    )
    
    print(f"\nAttentionModel CV Results:")
    print(f"  Accuracy: {attention_cv_results['accuracy']:.4f} ± {attention_cv_results['accuracy_std']:.4f}")
    print(f"  F1 Score: {attention_cv_results['f1']:.4f} ± {attention_cv_results['f1_std']:.4f}")
    print(f"  Top-2 Accuracy: {attention_cv_results['top2_accuracy']:.4f} ± {attention_cv_results['top2_accuracy_std']:.4f}")
    
    save_results(attention_cv_results, f"{args.output_dir}/results/attention_cv_results.json")
    
    # MLP Baseline CV
    print("\n--- MLP Baseline Cross-Validation ---")
    input_size = team_stat_size + (8 * player_stat_size)
    mlp_kwargs = {
        'input_size': input_size,
        'num_classes': 6,
        'hidden_size': args.hidden_size,
        'dropout_rate': args.dropout
    }
    
    mlp_cv_results = leave_one_season_out_cv(
        MLPBaseline,
        mlp_kwargs,
        dataset,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=device
    )
    
    print(f"\nMLP Baseline CV Results:")
    print(f"  Accuracy: {mlp_cv_results['accuracy']:.4f} ± {mlp_cv_results['accuracy_std']:.4f}")
    print(f"  F1 Score: {mlp_cv_results['f1']:.4f} ± {mlp_cv_results['f1_std']:.4f}")
    print(f"  Top-2 Accuracy: {mlp_cv_results['top2_accuracy']:.4f} ± {mlp_cv_results['top2_accuracy_std']:.4f}")
    
    save_results(mlp_cv_results, f"{args.output_dir}/results/mlp_cv_results.json")
    
    print("\n✓ Evaluation complete!")
    return attention_cv_results, mlp_cv_results


def run_interpretability(args, tensors, attention_model, mlp_model):
    """Run interpretability analysis."""
    print("\n" + "=" * 60)
    print("STEP 4: Interpretability Analysis")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    team_features = torch.FloatTensor(tensors['team_features']).to(device)
    player_features = torch.FloatTensor(tensors['player_features']).to(device)
    labels = torch.LongTensor(tensors['labels']).to(device)
    
    # Player names for visualization
    player_names = ['Star', '2nd Star', 'Starter 3', 'Starter 4',
                   'Starter 5', '6th Man', 'Bench 7', 'Bench 8']
    class_names = ['Missed Playoffs', 'First Round', 'Second Round',
                   'Conf Finals', 'Finals', 'Champion']
    
    # 1. Attention Weight Visualization
    print("\n--- Attention Weight Analysis ---")
    attention_stats = visualize_attention_weights(
        attention_model,
        team_features,
        player_features,
        labels,
        player_names=player_names,
        class_names=class_names,
        save_path=f"{args.output_dir}/figures/attention_weights.png"
    )
    
    save_results(attention_stats, f"{args.output_dir}/results/attention_stats.json")
    
    print("\nKey Findings:")
    for class_idx, stats in attention_stats.items():
        most_important = player_names[stats['mean'].argmax()]
        print(f"  {class_names[class_idx]}: Most attended player = {most_important}")
    
    # 2. t-SNE Visualization
    print("\n--- t-SNE Embedding Analysis ---")
    tsne_embeddings = visualize_tsne_embeddings(
        attention_model,
        team_features,
        player_features,
        labels,
        class_names=class_names,
        save_path=f"{args.output_dir}/figures/tsne_embeddings.png"
    )
    
    # 3. Champion Clustering Analysis
    print("\n--- Champion Clustering Analysis ---")
    clustering_results = analyze_champion_clustering(
        tsne_embeddings,
        labels.cpu().numpy()
    )
    
    save_results(clustering_results, f"{args.output_dir}/results/clustering_analysis.json")
    
    print(f"\nClustering Results:")
    print(f"  {clustering_results['interpretation']}")
    print(f"  Clustering ratio: {clustering_results['clustering_ratio']:.3f}")
    
    # 4. Model Comparison
    print("\n--- Model Embedding Comparison ---")
    comparison = compare_model_embeddings(
        attention_model,
        mlp_model,
        team_features,
        player_features,
        labels,
        class_names=class_names,
        save_path=f"{args.output_dir}/figures/model_comparison.png"
    )
    
    save_results(comparison, f"{args.output_dir}/results/model_comparison.json")
    
    print("\n✓ Interpretability analysis complete!")


def main():
    args = parse_args()
    setup_directories(args.output_dir)
    
    print("=" * 60)
    print("NBA Championship Prediction - Deep Learning")
    print("=" * 60)
    print(f"Configuration:")
    print(f"  Mode: {args.mode}")
    print(f"  Seasons: {args.seasons}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  Learning Rate: {args.lr}")
    print(f"  Embedding Size: {args.embedding_size}")
    print(f"  Hidden Size: {args.hidden_size}")
    print(f"  Dropout: {args.dropout}")
    print("=" * 60)
    
    tensors = None
    attention_model = None
    mlp_model = None
    
    # Run selected modes
    if args.mode in ['full', 'data']:
        tensors = run_data_collection(args)
    
    if args.mode in ['full', 'train', 'evaluate', 'interpret']:
        # Load tensors if not just collected
        if tensors is None:
            print("\nLoading pre-collected data...")
            data = np.load(f"{args.output_dir}/nba_tensors.npz")
            tensors = {
                'team_features': data['team_features'],
                'player_features': data['player_features'],
                'labels': data['labels'],
                'seasons': data['seasons'],
                'team_ids': data['team_ids']
            }
    
    if args.mode in ['full', 'train']:
        attention_model, mlp_model = run_training(args, tensors)
    
    if args.mode in ['full', 'evaluate', 'interpret']:
        # Load models if not just trained
        if attention_model is None:
            print("\nLoading pre-trained models...")
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            team_stat_size = tensors['team_features'].shape[1]
            player_stat_size = tensors['player_features'].shape[1] // 8
            
            attention_model = AttentionModel(
                team_stat_size=team_stat_size,
                player_stat_size=player_stat_size,
                num_players=8,
                embedding_size=args.embedding_size,
                num_classes=6,
                hidden_size=args.hidden_size,
                dropout_rate=args.dropout
            ).to(device)
            attention_model.load_state_dict(torch.load(f"{args.output_dir}/models/attention_model.pt"))
            
            input_size = team_stat_size + (8 * player_stat_size)
            mlp_model = MLPBaseline(
                input_size=input_size,
                num_classes=6,
                hidden_size=args.hidden_size,
                dropout_rate=args.dropout
            ).to(device)
            mlp_model.load_state_dict(torch.load(f"{args.output_dir}/models/mlp_baseline.pt"))
    
    if args.mode in ['full', 'evaluate']:
        run_evaluation(args, tensors)
    
    if args.mode in ['full', 'interpret']:
        run_interpretability(args, tensors, attention_model, mlp_model)
    
    print("\n" + "=" * 60)
    print("Pipeline Complete!")
    print(f"Results saved to: {args.output_dir}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
