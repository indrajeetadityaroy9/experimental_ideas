import argparse
import sys
from pathlib import Path
import numpy as np

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))

from go_motif_pooling.data import DatasetGenerator

def parse_args():
    parser = argparse.ArgumentParser(description='Generate Go board datasets')

    parser.add_argument(
        '--num-samples',
        type=int,
        default=100,
        help='Number of board states to generate'
    )
    parser.add_argument(
        '--complexity',
        type=str,
        choices=['low', 'medium', 'high'],
        default='medium',
        help='Dataset complexity level'
    )
    parser.add_argument(
        '--board-size',
        type=int,
        default=9,
        help='Board size (default: 9 for 9x9)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/boards.npz',
        help='Output path for generated data (NPZ format)'
    )
    parser.add_argument(
        '--hybrid',
        action='store_true',
        help='Generate hybrid dataset (tactical + game sequences)'
    )
    parser.add_argument(
        '--tactical',
        type=int,
        default=50,
        help='Number of tactical pattern boards (hybrid mode only)'
    )
    parser.add_argument(
        '--game',
        type=int,
        default=50,
        help='Number of game sequence boards (hybrid mode only)'
    )

    return parser.parse_args()

def main():

    args = parse_args()

    print('\n' + '=' * 80)
    print('GO MOTIF POOLING - DATA GENERATION')
    print('=' * 80)
    print(f'Board size: {args.board_size}x{args.board_size}')
    print(f'Seed: {args.seed}')
    print(f'Output: {args.output}')
    print('=' * 80 + '\n')

    data_gen = DatasetGenerator(board_size=args.board_size, seed=args.seed)

    if args.hybrid:
        print(f'Generating hybrid dataset:')
        print(f'  - Tactical boards: {args.tactical}')
        print(f'  - Game boards: {args.game}')
        print()

        boards, motif_stats = data_gen.generate_hybrid_dataset(
            tactical_boards=args.tactical,
            game_boards=args.game
        )
    else:
        print(f'Generating {args.num_samples} boards with {args.complexity} complexity\n')

        boards, motif_stats = data_gen.generate_mixed_dataset(
            num_samples=args.num_samples,
            complexity=args.complexity
        )

    boards_array = np.array(boards)

    motif_counts = {}
    all_patterns = set()
    for stats in motif_stats:
        all_patterns.update(stats.keys())

    for pattern in all_patterns:
        counts = [stats.get(pattern, 0) for stats in motif_stats]
        motif_counts[pattern] = np.array(counts)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    save_dict = {
        'boards': boards_array,
        'board_size': args.board_size,
        'num_samples': len(boards),
        'complexity': args.complexity if not args.hybrid else 'hybrid',
        'seed': args.seed
    }

    for pattern, counts in motif_counts.items():
        save_dict[f'motif_{pattern}'] = counts

    np.savez_compressed(output_path, **save_dict)

    print('\n' + '=' * 80)
    print('DATA GENERATION COMPLETE')
    print('=' * 80)
    print(f'Generated {len(boards)} board states')
    print(f'Saved to: {output_path}')
    print(f'File size: {output_path.stat().st_size / 1024:.2f} KB')
    print('=' * 80)

if __name__ == '__main__':
    main()
