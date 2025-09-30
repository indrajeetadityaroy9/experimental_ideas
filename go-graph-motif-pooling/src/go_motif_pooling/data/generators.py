import numpy as np
import random
from typing import Dict, List, Optional, Tuple
from go_motif_pooling.game.patterns import MotifConv2D

class TacticalPatternGenerator:

    def __init__(self, board_size: int = 9, seed: Optional[int] = None):
        self.board_size = board_size
        self.mconv = MotifConv2D()
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    def empty_board(self) -> np.ndarray:
        return np.zeros((self.board_size, self.board_size), dtype=int)

    def place_stone(self, board: np.ndarray, row: int, col: int, color: int) -> bool:
        if 0 <= row < self.board_size and 0 <= col < self.board_size:
            if board[row, col] == 0:
                board[row, col] = color
                return True
        return False

    def place_bamboo_pattern(self, board: np.ndarray, color: int) -> bool:
        attempts = 0
        max_attempts = 50

        while attempts < max_attempts:
            row = random.randint(0, self.board_size - 2)
            col = random.randint(0, self.board_size - 2)

            if (board[row, col] == 0 and board[row+1, col+1] == 0 and
                board[row, col+1] == 0 and board[row+1, col] == 0):

                board[row, col] = color
                board[row+1, col+1] = color
                return True

            attempts += 1

        return False

    def place_tiger_pattern(self, board: np.ndarray, color: int) -> bool:
        attempts = 0
        max_attempts = 50

        while attempts < max_attempts:
            row = random.randint(0, self.board_size - 2)
            col = random.randint(0, self.board_size - 2)

            positions = [(row, col), (row, col+1), (row+1, col), (row+1, col+1)]
            if all(board[r, c] == 0 for r, c in positions):

                empty_idx = random.randint(0, 3)
                for i, (r, c) in enumerate(positions):
                    if i != empty_idx:
                        board[r, c] = color
                return True

            attempts += 1

        return False

    def place_empty_triangle_pattern(self, board: np.ndarray, color: int) -> bool:
        return self.place_tiger_pattern(board, color)

    def place_eye_pattern(self, board: np.ndarray, color: int) -> bool:
        attempts = 0
        max_attempts = 50

        while attempts < max_attempts:
            row = random.randint(1, self.board_size - 2)
            col = random.randint(1, self.board_size - 2)

            positions = [(row-1, col), (row+1, col), (row, col-1), (row, col+1)]
            if board[row, col] == 0 and all(board[r, c] == 0 for r, c in positions):

                for r, c in positions:
                    board[r, c] = color
                return True

            attempts += 1

        return False

    def place_group(self, board: np.ndarray, color: int, num_stones: int,
                   start_row: Optional[int] = None,
                   start_col: Optional[int] = None) -> int:

        if start_row is None:
            start_row = random.randint(1, self.board_size - 2)
        if start_col is None:
            start_col = random.randint(1, self.board_size - 2)

        if board[start_row, start_col] != 0:
            return 0

        board[start_row, start_col] = color
        placed = 1

        frontier = [(start_row, start_col)]

        while frontier and placed < num_stones:
            current_row, current_col = random.choice(frontier)

            adjacent = [
                (current_row - 1, current_col),
                (current_row + 1, current_col),
                (current_row, current_col - 1),
                (current_row, current_col + 1)
            ]
            random.shuffle(adjacent)

            for r, c in adjacent:
                if placed >= num_stones:
                    break
                if 0 <= r < self.board_size and 0 <= c < self.board_size:
                    if board[r, c] == 0:
                        board[r, c] = color
                        frontier.append((r, c))
                        placed += 1
                        break
            else:

                frontier.remove((current_row, current_col))

        return placed

    def generate_tactical_board(self, pattern_counts: Optional[Dict[str, int]] = None) -> np.ndarray:
        if pattern_counts is None:
            pattern_counts = {
                'bamboo': random.randint(1, 3),
                'tiger': random.randint(1, 2),
                'eye': random.randint(0, 2),
                'empty_triangle': random.randint(0, 2),
                'groups': random.randint(2, 4)
            }

        board = self.empty_board()

        colors = [1, 2]
        color_idx = 0

        for _ in range(pattern_counts.get('bamboo', 0)):
            self.place_bamboo_pattern(board, colors[color_idx % 2])
            color_idx += 1

        for _ in range(pattern_counts.get('tiger', 0)):
            self.place_tiger_pattern(board, colors[color_idx % 2])
            color_idx += 1

        for _ in range(pattern_counts.get('eye', 0)):
            self.place_eye_pattern(board, colors[color_idx % 2])
            color_idx += 1

        for _ in range(pattern_counts.get('empty_triangle', 0)):
            self.place_empty_triangle_pattern(board, colors[color_idx % 2])
            color_idx += 1

        for _ in range(pattern_counts.get('groups', 0)):
            group_size = random.randint(3, 8)
            self.place_group(board, colors[color_idx % 2], group_size)
            color_idx += 1

        return board

    def verify_motifs(self, board: np.ndarray) -> Dict[str, int]:

        hits = self.mconv(board)
        motif_counts = {}

        for pattern_name, mask in hits.items():
            motif_counts[pattern_name] = mask.sum().item()

        return motif_counts

class DatasetGenerator:
    def __init__(self, board_size: int = 9, seed: Optional[int] = None):
        self.board_size = board_size
        self.pattern_gen = TacticalPatternGenerator(board_size, seed)
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    def generate_mixed_dataset(self, num_samples: int = 100,
                               complexity: str = 'medium') -> Tuple[List[np.ndarray], List[Dict]]:

        print(f"Generating {num_samples} board states with {complexity} complexity...")

        if complexity == 'low':
            pattern_ranges = {
                'bamboo': (0, 2),
                'tiger': (0, 1),
                'eye': (0, 1),
                'empty_triangle': (0, 1),
                'groups': (1, 3)
            }
        elif complexity == 'medium':
            pattern_ranges = {
                'bamboo': (1, 3),
                'tiger': (1, 3),
                'eye': (0, 2),
                'empty_triangle': (0, 2),
                'groups': (2, 5)
            }
        else:
            pattern_ranges = {
                'bamboo': (2, 5),
                'tiger': (2, 4),
                'eye': (1, 3),
                'empty_triangle': (1, 3),
                'groups': (3, 6)
            }

        boards = []
        motif_stats = []

        for i in range(num_samples):
            pattern_counts = {
                pattern: random.randint(min_val, max_val)
                for pattern, (min_val, max_val) in pattern_ranges.items()
            }

            board = self.pattern_gen.generate_tactical_board(pattern_counts)
            boards.append(board)

            motifs = self.pattern_gen.verify_motifs(board)
            motif_stats.append(motifs)

            if (i + 1) % 20 == 0:
                print(f"  Generated {i + 1}/{num_samples} boards...")

        print(f"Generated {num_samples} boards")

        self._print_motif_statistics(motif_stats)

        return boards, motif_stats

    def generate_hybrid_dataset(self, tactical_boards: int = 50,
                                game_boards: int = 50) -> Tuple[List[np.ndarray], List[Dict]]:

        print("=" * 80)
        print("GENERATING HYBRID DATASET")
        print("=" * 80)

        print(f"\n1. Generating {tactical_boards} tactical pattern boards...")
        tactical, tactical_stats = self.generate_mixed_dataset(
            tactical_boards, complexity='medium'
        )

        print(f"\n2. Adding {game_boards} additional boards...")
        all_boards = tactical
        all_stats = tactical_stats

        print(f"\n{'=' * 80}")
        print(f"DATASET GENERATED: {len(all_boards)} total board states")
        print(f"  - Tactical patterns: {len(tactical)} boards")
        print(f"{'=' * 80}")

        return all_boards, all_stats

    def _print_motif_statistics(self, motif_stats: List[Dict]):
        print("\nMotif Detection Statistics:")
        all_patterns = set()
        for stats in motif_stats:
            all_patterns.update(stats.keys())

        for pattern in sorted(all_patterns):
            counts = [stats.get(pattern, 0) for stats in motif_stats]
            total = sum(counts)
            boards_with_pattern = sum(1 for c in counts if c > 0)
            avg = total / len(counts) if counts else 0
            max_count = max(counts) if counts else 0

            print(f"  {pattern:25s}: {boards_with_pattern:3d} boards, "
                  f"Total={total:4d}, Avg={avg:.2f}, Max={max_count:2d}")
