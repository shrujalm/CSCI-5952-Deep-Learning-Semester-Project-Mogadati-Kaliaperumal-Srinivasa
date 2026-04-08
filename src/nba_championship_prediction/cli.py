"""Command-line entrypoint for the NBA championship prediction pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from .data_pipeline import SEASONS, build_dataset, prepare_tensors
from .interpretability import (
    analyze_champion_clustering,
    compare_model_embeddings,
    visualize_attention_weights,
    visualize_tsne_embeddings,
)
from .modeling import AttentionModel, MLPBaseline, PLAYOFF_CLASS_NAMES, PLAYER_SLOT_NAMES
from .training import NBADataset, leave_one_season_out_cv, save_results, train_final_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="NBA Championship Prediction using Deep Learning"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="full",
        choices=["full", "data", "train", "evaluate", "interpret"],
        help="Pipeline stage to run.",
    )
    parser.add_argument(
        "--seasons",
        type=int,
        default=20,
        help="Number of most recent seasons to include.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Training epochs.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Mini-batch size.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Optimizer learning rate.",
    )
    parser.add_argument(
        "--embedding_size",
        type=int,
        default=32,
        help="Player embedding dimension.",
    )
    parser.add_argument(
        "--hidden_size",
        type=int,
        default=128,
        help="Shared hidden size used by the classifiers.",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.3,
        help="Dropout rate.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="Directory for datasets, models, figures, and metrics.",
    )
    return parser.parse_args()


def setup_directories(output_dir: str) -> Path:
    """Create the standard output tree for the pipeline."""
    output_root = Path(output_dir)
    for subdirectory in ("models", "figures", "results"):
        (output_root / subdirectory).mkdir(parents=True, exist_ok=True)
    return output_root


def build_attention_kwargs(tensors: dict, args: argparse.Namespace) -> dict:
    team_stat_size = tensors["team_features"].shape[1]
    player_stat_size = tensors["player_features"].shape[1] // 8
    return {
        "team_stat_size": team_stat_size,
        "player_stat_size": player_stat_size,
        "num_players": 8,
        "embedding_size": args.embedding_size,
        "num_classes": len(PLAYOFF_CLASS_NAMES),
        "hidden_size": args.hidden_size,
        "dropout_rate": args.dropout,
    }


def build_mlp_kwargs(tensors: dict, args: argparse.Namespace) -> dict:
    team_stat_size = tensors["team_features"].shape[1]
    player_stat_size = tensors["player_features"].shape[1] // 8
    return {
        "input_size": team_stat_size + (8 * player_stat_size),
        "num_classes": len(PLAYOFF_CLASS_NAMES),
        "hidden_size": args.hidden_size,
        "dropout_rate": args.dropout,
    }


def load_tensors(output_root: Path) -> dict:
    tensor_path = output_root / "nba_tensors.npz"
    if not tensor_path.exists():
        raise FileNotFoundError(
            f"Missing cached tensors at {tensor_path}. "
            "Run '--mode data' or '--mode full' first."
        )

    data = np.load(tensor_path, allow_pickle=True)
    return {
        "team_features": data["team_features"],
        "player_features": data["player_features"],
        "labels": data["labels"],
        "seasons": data["seasons"],
        "team_ids": data["team_ids"],
    }


def run_data_collection(args: argparse.Namespace, output_root: Path) -> dict:
    print("\n" + "=" * 60)
    print("STEP 1: Data Collection")
    print("=" * 60)

    seasons = SEASONS[-args.seasons :]
    dataset = build_dataset(
        seasons=seasons,
        top_n_players=8,
        save_path=str(output_root / "nba_dataset.csv"),
    )
    tensors = prepare_tensors(dataset)
    np.savez(output_root / "nba_tensors.npz", **tensors)
    print("\nData collection complete.")
    return tensors


def run_training(
    args: argparse.Namespace,
    tensors: dict,
    output_root: Path,
) -> tuple[AttentionModel, MLPBaseline]:
    print("\n" + "=" * 60)
    print("STEP 2: Model Training")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset = NBADataset(
        tensors["team_features"],
        tensors["player_features"],
        tensors["labels"],
        tensors["seasons"],
    )

    print("\n--- Training Attention Model ---")
    attention_model, attention_history = train_final_model(
        AttentionModel,
        build_attention_kwargs(tensors, args),
        dataset,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=device,
    )
    torch.save(attention_model.state_dict(), output_root / "models" / "attention_model.pt")
    torch.save(attention_history, output_root / "results" / "attention_history.pt")

    print(f"  Best validation accuracy: {max(attention_history['val_acc']):.4f}")
    print(f"  Best validation F1: {max(attention_history['val_f1']):.4f}")
    print(f"  Best validation top-2 accuracy: {max(attention_history['val_top2']):.4f}")

    print("\n--- Training MLP Baseline ---")
    mlp_model, mlp_history = train_final_model(
        MLPBaseline,
        build_mlp_kwargs(tensors, args),
        dataset,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=device,
    )
    torch.save(mlp_model.state_dict(), output_root / "models" / "mlp_baseline.pt")
    torch.save(mlp_history, output_root / "results" / "mlp_history.pt")

    print(f"  Best validation accuracy: {max(mlp_history['val_acc']):.4f}")
    print(f"  Best validation F1: {max(mlp_history['val_f1']):.4f}")
    print(f"  Best validation top-2 accuracy: {max(mlp_history['val_top2']):.4f}")
    print("\nTraining complete.")

    return attention_model, mlp_model


def run_evaluation(args: argparse.Namespace, tensors: dict, output_root: Path) -> tuple[dict, dict]:
    print("\n" + "=" * 60)
    print("STEP 3: Cross-Validation Evaluation")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = NBADataset(
        tensors["team_features"],
        tensors["player_features"],
        tensors["labels"],
        tensors["seasons"],
    )

    print("\n--- Attention Model Cross-Validation ---")
    attention_cv_results = leave_one_season_out_cv(
        AttentionModel,
        build_attention_kwargs(tensors, args),
        dataset,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=device,
    )
    save_results(attention_cv_results, output_root / "results" / "attention_cv_results.json")
    print(
        f"  Accuracy: {attention_cv_results['accuracy']:.4f} +/- {attention_cv_results['accuracy_std']:.4f}"
    )
    print(f"  F1: {attention_cv_results['f1']:.4f} +/- {attention_cv_results['f1_std']:.4f}")
    print(
        "  Top-2 Accuracy: "
        f"{attention_cv_results['top2_accuracy']:.4f} +/- {attention_cv_results['top2_accuracy_std']:.4f}"
    )

    print("\n--- MLP Baseline Cross-Validation ---")
    mlp_cv_results = leave_one_season_out_cv(
        MLPBaseline,
        build_mlp_kwargs(tensors, args),
        dataset,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=device,
    )
    save_results(mlp_cv_results, output_root / "results" / "mlp_cv_results.json")
    print(f"  Accuracy: {mlp_cv_results['accuracy']:.4f} +/- {mlp_cv_results['accuracy_std']:.4f}")
    print(f"  F1: {mlp_cv_results['f1']:.4f} +/- {mlp_cv_results['f1_std']:.4f}")
    print(
        "  Top-2 Accuracy: "
        f"{mlp_cv_results['top2_accuracy']:.4f} +/- {mlp_cv_results['top2_accuracy_std']:.4f}"
    )

    print("\nEvaluation complete.")
    return attention_cv_results, mlp_cv_results


def run_interpretability(
    args: argparse.Namespace,
    tensors: dict,
    attention_model: AttentionModel,
    mlp_model: MLPBaseline,
    output_root: Path,
) -> None:
    print("\n" + "=" * 60)
    print("STEP 4: Interpretability Analysis")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    team_features = torch.as_tensor(tensors["team_features"], dtype=torch.float32, device=device)
    player_features = torch.as_tensor(tensors["player_features"], dtype=torch.float32, device=device)
    labels = torch.as_tensor(tensors["labels"], dtype=torch.long, device=device)

    class_names = list(PLAYOFF_CLASS_NAMES)
    player_names = list(PLAYER_SLOT_NAMES)

    print("\n--- Attention Weight Analysis ---")
    attention_stats = visualize_attention_weights(
        attention_model,
        team_features,
        player_features,
        labels,
        player_names=player_names,
        class_names=class_names,
        save_path=str(output_root / "figures" / "attention_weights.png"),
    )
    save_results(attention_stats, output_root / "results" / "attention_stats.json")
    for class_idx, stats in attention_stats.items():
        most_important = player_names[int(stats["mean"].argmax())]
        print(f"  {class_names[class_idx]}: most attended slot = {most_important}")

    print("\n--- t-SNE Embedding Analysis ---")
    tsne_embeddings = visualize_tsne_embeddings(
        attention_model,
        team_features,
        player_features,
        labels,
        class_names=class_names,
        save_path=str(output_root / "figures" / "tsne_embeddings.png"),
    )

    print("\n--- Champion Clustering Analysis ---")
    clustering_results = analyze_champion_clustering(tsne_embeddings, labels.cpu().numpy())
    save_results(clustering_results, output_root / "results" / "clustering_analysis.json")
    if "error" in clustering_results:
        print(f"  {clustering_results['error']}")
    else:
        print(f"  {clustering_results['interpretation']}")
        print(f"  Clustering ratio: {clustering_results['clustering_ratio']:.3f}")

    print("\n--- Model Embedding Comparison ---")
    comparison = compare_model_embeddings(
        attention_model,
        mlp_model,
        team_features,
        player_features,
        labels,
        class_names=class_names,
        save_path=str(output_root / "figures" / "model_comparison.png"),
    )
    save_results(comparison, output_root / "results" / "model_comparison.json")
    print("Interpretability analysis complete.")


def load_models(
    args: argparse.Namespace,
    tensors: dict,
    output_root: Path,
) -> tuple[AttentionModel, MLPBaseline]:
    """Load trained model checkpoints from disk."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    attention_checkpoint = output_root / "models" / "attention_model.pt"
    mlp_checkpoint = output_root / "models" / "mlp_baseline.pt"

    if not attention_checkpoint.exists() or not mlp_checkpoint.exists():
        raise FileNotFoundError(
            "Missing model checkpoints. Run '--mode train' or '--mode full' first."
        )

    attention_model = AttentionModel(**build_attention_kwargs(tensors, args)).to(device)
    attention_model.load_state_dict(
        torch.load(attention_checkpoint, map_location=device)
    )

    mlp_model = MLPBaseline(**build_mlp_kwargs(tensors, args)).to(device)
    mlp_model.load_state_dict(
        torch.load(mlp_checkpoint, map_location=device)
    )

    return attention_model, mlp_model


def main() -> None:
    args = parse_args()
    output_root = setup_directories(args.output_dir)

    print("=" * 60)
    print("NBA Championship Prediction - Deep Learning Pipeline")
    print("=" * 60)
    print("Configuration:")
    print(f"  Mode: {args.mode}")
    print(f"  Seasons: {args.seasons}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  Learning Rate: {args.lr}")
    print(f"  Embedding Size: {args.embedding_size}")
    print(f"  Hidden Size: {args.hidden_size}")
    print(f"  Dropout: {args.dropout}")
    print(f"  Output Directory: {output_root}")
    print("=" * 60)

    tensors = None
    attention_model = None
    mlp_model = None

    if args.mode in {"full", "data"}:
        tensors = run_data_collection(args, output_root)

    if args.mode in {"full", "train", "evaluate", "interpret"} and tensors is None:
        print("\nLoading cached tensors from disk...")
        tensors = load_tensors(output_root)

    if args.mode in {"full", "train"}:
        attention_model, mlp_model = run_training(args, tensors, output_root)

    if args.mode in {"full", "evaluate"}:
        run_evaluation(args, tensors, output_root)

    if args.mode in {"full", "interpret"}:
        if attention_model is None or mlp_model is None:
            print("\nLoading pretrained models from disk...")
            attention_model, mlp_model = load_models(args, tensors, output_root)
        run_interpretability(args, tensors, attention_model, mlp_model, output_root)

    print("\n" + "=" * 60)
    print("Pipeline complete.")
    print(f"Artifacts saved to {output_root}")
    print("=" * 60)


if __name__ == "__main__":
    main()
