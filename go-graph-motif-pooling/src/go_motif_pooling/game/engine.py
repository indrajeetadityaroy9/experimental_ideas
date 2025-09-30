import numpy as np
from typing import Set, Tuple, List, Optional

class GoEngine:
    def __init__(self, size: int = 9):
        self.size = size
        self.board = np.zeros((size, size), dtype=int)
        self.move_history: List[Tuple[int, np.ndarray]] = []
        self.current_player = 1

    def reset(self):
        self.board = np.zeros((self.size, self.size), dtype=int)
        self.move_history.clear()
        self.current_player = 1

    def get_group(self, board: np.ndarray, x: int, y: int) -> Tuple[Set[Tuple[int, int]], Set[Tuple[int, int]]]:
        color = board[x, y]
        if color == 0:
            return set(), set()

        queue = [(x, y)]
        group = set()
        liberties = set()
        visited = {(x, y)}

        while queue:
            current_x, current_y = queue.pop(0)
            group.add((current_x, current_y))

            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                next_x, next_y = current_x + dx, current_y + dy
                if (0 <= next_x < self.size and 0 <= next_y < self.size
                        and (next_x, next_y) not in visited):
                    visited.add((next_x, next_y))
                    if board[next_x, next_y] == 0:
                        liberties.add((next_x, next_y))
                    elif board[next_x, next_y] == color:
                        queue.append((next_x, next_y))

        return group, liberties

    def is_valid_move(self, x: int, y: int, check_ko: bool = False) -> bool:
        if self.board[x, y] != 0:
            return False

        board_copy = self.board.copy()
        board_copy[x, y] = self.current_player

        _, libs = self.get_group(board_copy, x, y)

        if libs:
            return True

        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            next_x, next_y = x + dx, y + dy
            if (0 <= next_x < self.size and 0 <= next_y < self.size
                    and board_copy[next_x, next_y] == 3 - self.current_player):
                _, opp_libs = self.get_group(board_copy, next_x, next_y)
                if not opp_libs:
                    return True

        return False

    def play_move(self, x: int, y: int) -> bool:
        if not self.is_valid_move(x, y):
            return False

        self.move_history.append((self.current_player, self.board.copy()))

        self.board[x, y] = self.current_player
        opponent_color = 3 - self.current_player

        for i in range(self.size):
            for j in range(self.size):
                if self.board[i, j] == opponent_color:
                    group, libs = self.get_group(self.board, i, j)
                    if not libs:

                        for stone_x, stone_y in group:
                            self.board[stone_x, stone_y] = 0

        self.current_player = 3 - self.current_player
        return True

    def undo_move(self) -> bool:
        if not self.move_history:
            return False

        last_player, last_board = self.move_history.pop()
        self.board = last_board.copy()
        self.current_player = last_player
        return True

    def get_board_copy(self) -> np.ndarray:
        return self.board.copy()

    def get_all_board_states(self) -> List[np.ndarray]:
        states = [board.copy() for _, board in self.move_history]
        states.append(self.board.copy())
        return states

    def count_stones(self, color: Optional[int] = None) -> int:
        if color is None:
            return int(np.sum(self.board == 1) - np.sum(self.board == 2))
        else:
            return int(np.sum(self.board == color))
