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

    def generate_move(self, direction: str) -> tuple[np.ndarray, int]:
        """Moves tiles in the given direction and adds a new random tile."""
        new_board = self.board.copy()
        if direction == "UP":
            new_board = np.transpose(
                [self._compress_and_merge(row) for row in np.transpose(new_board)]
            )
        elif direction == "DOWN":
            new_board = np.transpose(
                [
                    self._compress_and_merge(row[::-1])[::-1]
                    for row in np.transpose(new_board)
                ]
            )
        elif direction == "LEFT":
            new_board = np.array([self._compress_and_merge(row) for row in new_board])
        elif direction == "RIGHT":
            new_board = np.array(
                [self._compress_and_merge(row[::-1])[::-1] for row in new_board]
            )
        else:
            raise ValueError(f"Invalid direction: {direction}")
        return new_board, self.score

    def _compress_and_merge(self, row: np.ndarray) -> tuple[np.ndarray, int]:
        """Moves nonzero values left and merges equal adjacent tiles."""
        row = row[row != 0]  # Remove zeros
        i, new_row, score = 0, [], self.score
        while i < len(row):
            if i < len(row) - 1 and row[i] == row[i + 1]:
                new_row.append(row[i] * 2)
                score += row[i] * 2
                i += 2  # Skip next tile since it merged
            else:
                new_row.append(row[i])
                i += 1
        return np.array(new_row + [0] * (4 - len(new_row))), score  # Pad with zeros

    def is_game_over(self):
        """Checks if there are no valid moves left."""
        for direction in self.get_possible_moves():
            temp_game = Game2048()
            temp_game.board = self.board.copy()
            temp_game.move(direction)
            if not np.array_equal(temp_game.board, self.board):
                return False  # There's at least one valid move
        return True

    def get_possible_moves(self) -> list[str]:
        return ["UP", "DOWN", "LEFT", "RIGHT"]

    def print_board(self):
        """Prints the board for debugging."""
        print(self.board)
