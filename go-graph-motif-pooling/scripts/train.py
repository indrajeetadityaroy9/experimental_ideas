import argparse
import sys
from pathlib import Path
import torch
import numpy as np

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))

from go_motif_pooling.core import load_config, Trainer
from go_motif_pooling.models import MPoolModel
from go_motif_pooling.data import DatasetGenerator, boards_to_dataset, GoBoardGraphDataset

def parse_args():

    parser = argparse.ArgumentParser(description='Train Go Motif Pooling model')

    parser.add_argument(
        '--config',
        type=str,
        default='configs/default.yaml',
        help='Path to config file'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=None,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=None,
        help='Batch size'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=None,
        help='Learning rate'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='Path to save checkpoint'
    )
    parser.add_argument(
        '--num-samples',
        type=int,
        default=None,
        help='Number of training samples to generate'
    )
    parser.add_argument(
        '--complexity',
        type=str,
        choices=['low', 'medium', 'high'],
        default=None,
        help='Dataset complexity'
    )

    return parser.parse_args()

def main():
    args = parse_args()
    print(f'Loading config from {args.config}')
    config = load_config(args.config)

    if args.epochs is not None:
        config.training.num_epochs = args.epochs
    if args.batch_size is not None:
        config.training.batch_size = args.batch_size
    if args.lr is not None:
        config.optimizer.lr = args.lr
    if args.seed is not None:
        config.experiment.seed = args.seed
    if args.num_samples is not None:
        config.data.num_samples = args.num_samples
    if args.complexity is not None:
        config.data.complexity = args.complexity

    torch.manual_seed(config.experiment.seed)
    np.random.seed(config.experiment.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.experiment.seed)

    print('\n' + '=' * 80)
    print('GO MOTIF POOLING - TRAINING')
    print('=' * 80)
    print(f'Experiment: {config.experiment.name}')
    print(f'Board size: {config.model.board_size}x{config.model.board_size}')
    print(f'Seed: {config.experiment.seed}')
    print(f'Device: {"cuda" if torch.cuda.is_available() else "cpu"}')
    print('=' * 80 + '\n')

    print('Generating training dataset...')
    data_gen = DatasetGenerator(
        board_size=config.model.board_size,
        seed=config.experiment.seed
    )

    boards, motif_stats = data_gen.generate_mixed_dataset(
        num_samples=config.data.num_samples,
        complexity=config.data.complexity
    )

    print(f'\nConverting {len(boards)} boards to graph dataset...')
    graphs = boards_to_dataset(boards, targets=[float(i) for i in range(len(boards))])
    dataset = GoBoardGraphDataset(graphs)
    print(f'Created dataset with {len(dataset)} graphs\n')

    print('Creating model...')
    model = MPoolModel(
        size=config.model.board_size,
        in_dim=config.model.in_dim,
        hidden=config.model.hidden_dim,
        num_layers=config.model.num_layers,
        pool_ratios=config.model.pool_ratios,
        cluster_ratios=config.model.cluster_ratios,
        lambda_cut=config.model.lambda_cut,
        lambda_ortho=config.model.lambda_ortho
    )
    print(f'Model created with {sum(p.numel() for p in model.parameters())} parameters\n')

    trainer = Trainer(model, dataset, config=config.__dict__)

    print('Starting training...\n')
    best_val = trainer.train(
        num_epochs=config.training.num_epochs,
        batch_size=config.training.batch_size,
        lr=config.optimizer.lr,
        patience=config.training.patience,
        seed=config.experiment.seed
    )

    print('\nEvaluating on test set...')
    test_metrics = trainer.evaluate(split='test')
    print(f"Test metrics: {test_metrics}")

    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        trainer.save_checkpoint(str(checkpoint_path))

    print('\n' + '=' * 80)
    print('TRAINING COMPLETE')
    print('=' * 80)
    print(f'Best validation loss: {best_val:.4f}')
    print(f'Test task loss: {test_metrics["task"]:.4f}')
    print(f'Test reg loss: {test_metrics["reg"]:.4f}')
    print(f'Test total loss: {test_metrics["total"]:.4f}')
    print('=' * 80)

if __name__ == '__main__':
    main()
