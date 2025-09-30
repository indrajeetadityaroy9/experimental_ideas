import argparse
import json
import sys
from pathlib import Path
import torch
import numpy as np

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))

from go_motif_pooling.core import load_config, Trainer
from go_motif_pooling.models import MPoolModel
from go_motif_pooling.data import DatasetGenerator, boards_to_dataset, GoBoardGraphDataset
from go_motif_pooling.utils.metrics import compute_mse, compute_mae, compute_r2

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate Go Motif Pooling model')

    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/default.yaml',
        help='Path to config file'
    )
    parser.add_argument(
        '--split',
        type=str,
        choices=['train', 'val', 'test'],
        default='test',
        help='Which split to evaluate'
    )
    parser.add_argument(
        '--num-samples',
        type=int,
        default=None,
        help='Number of samples to generate (overrides config)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Path to save evaluation results (JSON)'
    )
    parser.add_argument(
        '--inference-limit',
        type=int,
        default=10,
        help='Number of samples to run detailed inference on'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed (overrides config)'
    )

    return parser.parse_args()

def main():

    args = parse_args()

    print(f'Loading config from {args.config}')
    config = load_config(args.config)

    if args.seed is not None:
        config.experiment.seed = args.seed
    if args.num_samples is not None:
        config.data.num_samples = args.num_samples

    torch.manual_seed(config.experiment.seed)
    np.random.seed(config.experiment.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.experiment.seed)

    print('\n' + '=' * 80)
    print('GO MOTIF POOLING - EVALUATION')
    print('=' * 80)
    print(f'Checkpoint: {args.checkpoint}')
    print(f'Config: {args.config}')
    print(f'Split: {args.split}')
    print(f'Device: {"cuda" if torch.cuda.is_available() else "cpu"}')
    print('=' * 80 + '\n')

    print('Generating evaluation dataset...')
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

    trainer = Trainer(model, dataset)
    trainer.setup_data_splits(seed=config.experiment.seed)
    print(f'Loading checkpoint from {args.checkpoint}...')
    trainer.load_checkpoint(args.checkpoint)

    print(f'\nEvaluating on {args.split} set...')
    metrics = trainer.evaluate(split=args.split)

    print(f'\nRunning detailed inference on {args.inference_limit} samples...')
    inference_results = trainer.run_inference(split=args.split, limit=args.inference_limit)

    predictions = []
    targets = []

    dataset_split = trainer.data_splits[args.split]
    for idx in range(min(args.inference_limit, len(dataset_split))):
        data = dataset_split[idx]
        predictions.append(inference_results[idx]['prediction'])
        targets.append(data.y.item())

    predictions = torch.tensor(predictions)
    targets = torch.tensor(targets)

    mse = compute_mse(predictions, targets)
    mae = compute_mae(predictions, targets)
    r2 = compute_r2(predictions, targets)

    print('\n' + '=' * 80)
    print('EVALUATION RESULTS')
    print('=' * 80)
    print(f'Split: {args.split}')
    print(f'Samples evaluated: {len(predictions)}')
    print('-' * 80)
    print(f'Task Loss (MSE):  {metrics["task"]:.4f}')
    print(f'Reg Loss:         {metrics["reg"]:.4f}')
    print(f'Total Loss:       {metrics["total"]:.4f}')
    print('-' * 80)
    print(f'MSE:              {mse:.4f}')
    print(f'MAE:              {mae:.4f}')
    print(f'R²:               {r2:.4f}')
    print('=' * 80)

    print('\nSample Diagnostics (first 3 samples):')
    for i in range(min(3, len(inference_results))):
        result = inference_results[i]
        print(f'\nSample {i+1}:')
        print(f'  Prediction: {result["prediction"]:.4f}')
        print(f'  Target:     {targets[i].item():.4f}')
        print(f'  Reg Loss:   {result["reg_loss"]:.4f}')

        for layer_idx, layer_diag in enumerate(result['diagnostics'], start=1):
            losses = layer_diag['clustering']['losses']
            print(f'  Layer {layer_idx}: cut={losses["cut"]:.4f}, ortho={losses["ortho"]:.4f}')

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        results = {
            'checkpoint': args.checkpoint,
            'config': args.config,
            'split': args.split,
            'metrics': {
                'task_loss': float(metrics['task']),
                'reg_loss': float(metrics['reg']),
                'total_loss': float(metrics['total']),
                'mse': float(mse),
                'mae': float(mae),
                'r2': float(r2)
            },
            'num_samples': len(predictions),
            'predictions': predictions.tolist(),
            'targets': targets.tolist()
        }

        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

        print(f'\nResults saved to {output_path}')

if __name__ == '__main__':
    main()
