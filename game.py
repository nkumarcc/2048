import numpy as np
import random


class Game2048:
    def __init__(
        self, board: np.ndarray = np.zeros((4, 4), dtype=int), score=0, is_new_game=True
    ):
        self.board = board
        self.score = score
        if is_new_game:
            self.add_random_tile()
            self.add_random_tile()

    def add_random_tile(self):
        """Adds a random tile (90% 2, 10% 4) to an empty space."""
        empty_tiles = list(zip(*np.where(self.board == 0)))
        if empty_tiles:
            row, col = random.choice(empty_tiles)
            self.board[row, col] = 2 if random.random() < 0.9 else 4

    def get_possible_computer_moves(self):
        """Returns coordinates of all possible computer moves with current board."""
        return list(zip(*np.where(self.board == 0)))

    def get_tiles_and_probabilities(self):
        """Returns all possible tiles and their probabilities."""
        return [(2, 0.9), (4, 0.1)]

    def execute_computer_move(self, tile: int, row: int, col: int):
        """Executes a computer move given a tile value and coordinates."""
        self.board[row, col] = tile

    def execute_player_move(self, direction: str) -> tuple[np.ndarray, int]:
        """Moves tiles in the given direction."""

        if direction not in self.get_possible_moves():
            raise ValueError(f"Invalid direction: {direction}")

        # Copy board and initialize score
        board, score, new_board = self.board.copy(), self.score, []

        if direction in ["UP", "DOWN"]:
            board = np.transpose(board)

        for row in board:
            new_row, score = self._compress_and_merge(
                row if direction in ["UP", "LEFT"] else row[::-1],
                score,
            )
            new_board.append(new_row if direction in ["UP", "LEFT"] else new_row[::-1])

        if direction in ["UP", "DOWN"]:
            new_board = np.transpose(new_board)
        else:
            new_board = np.array(new_board)

        return new_board, score

    def _compress_and_merge(
        self, row: np.ndarray, score: int
    ) -> tuple[np.ndarray, int]:
        """Moves nonzero values left and merges equal adjacent tiles."""
        row = row[row != 0]  # Remove zeros
        i, new_row = 0, []
        while i < len(row):
            if i < len(row) - 1 and row[i] == row[i + 1]:
                new_row.append(row[i] * 2)
                score += row[i] * 2
                i += 2  # Skip next tile since it merged
            else:
                new_row.append(row[i])
                i += 1
        return np.array(new_row + [0] * (4 - len(new_row))), score  # Pad with zeros

    def get_possible_moves(self) -> list[str]:
        return ["UP", "DOWN", "LEFT", "RIGHT"]

    def print_board(self):
        """Prints the board for debugging."""
        print(self.board)
