import numpy as np
import random
from collections import defaultdict
import argparse
from game import Game2048


class NTupleAgent:
    def __init__(
        self,
        learning_rate=0.1,
        discount_factor=0.9,
        should_mirror_horizontal=False,
        should_mirror_vertical=False,
        should_rotate_90=False,
        should_rotate_270=False,
    ):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.lookup_table = defaultdict(float)  # Default all tuples to 0
        self.should_mirror_horizontal = should_mirror_horizontal
        self.should_mirror_vertical = should_mirror_vertical
        self.should_rotate_90 = should_rotate_90
        self.should_rotate_270 = should_rotate_270

    def extract_tuples(self, board):
        """Extracts predefined N-Tuple patterns (e.g., rows, columns, diagonals)."""
        return [
            (
                board[0, 0],
                board[0, 1],
                board[0, 2],
                board[0, 3],
                board[1, 0],
                board[1, 1],
            ),
            (
                board[0, 0],
                board[0, 1],
                board[0, 2],
                board[0, 3],
                board[1, 0],
                board[1, 2],
            ),
            (
                board[0, 0],
                board[0, 1],
                board[0, 2],
                board[1, 0],
                board[1, 2],
                board[1, 3],
            ),
            (
                board[0, 0],
                board[0, 1],
                board[0, 2],
                board[1, 0],
                board[1, 1],
                board[1, 2],
            ),
        ]

    def get_rotated_boards(self, board):
        boards = [board]
        if self.should_mirror_horizontal:
            boards.append(np.flip(board, axis=1))
        if self.should_mirror_vertical:
            boards.append(np.flip(board, axis=0))
        if self.should_rotate_90:
            boards.append(np.rot90(board))
        if self.should_rotate_270:
            boards.append(np.rot90(board, 3))
        return boards

    def evaluate_board(self, board):
        """Estimates the board value as the sum of the lookup table values."""
        summed_scores = []
        for board in self.get_rotated_boards(board):
            summed_scores.append(0)
            for board_tuple in self.extract_tuples(board):
                summed_scores[-1] += self.lookup_table[board_tuple]

        return sum(summed_scores) / len(summed_scores)

    def evaluate(self, board: np.ndarray):
        empty_tiles = len(np.where(board == 0)[0])
        max_tile = np.max(board)

        # Encourage empty spaces and smooth merges
        return (
            2.5 * empty_tiles  # More space = better
            + 1.2 * self.monotonicity(board)
            + 0.8 * self.smoothness(board)
            + max_tile  # Reward large tile presence
        )

    def monotonicity(self, board: np.ndarray) -> float:
        """Encourages tiles to be ordered in a consistent direction (either increasing or decreasing)."""
        total_score = 0

        for row in board:
            increasing, decreasing = 0, 0
            for i in range(3):
                increasing += max(0, row[i + 1] - row[i])  # Only count increasing diffs
                decreasing += max(0, row[i] - row[i + 1])  # Only count decreasing diffs
            total_score += min(
                increasing, decreasing
            )  # Pick the better strategy for this row

        for col in board.T:
            increasing, decreasing = 0, 0
            for i in range(3):
                increasing += max(0, col[i + 1] - col[i])
                decreasing += max(0, col[i] - col[i + 1])
            total_score += min(
                increasing, decreasing
            )  # Pick the better strategy for this column

        return -total_score  # Lower penalty means better board

    def smoothness(self, board):
        """Penalizes big jumps in tile values."""
        score = 0
        for row in board:
            for i in range(3):
                score -= abs(row[i] - row[i + 1])  # Large differences are bad
        for col in board.T:
            for i in range(3):
                score -= abs(col[i] - col[i + 1])
        return score  # Higher score is better

    def update_lookup_table(self, board, next_board):
        """Updates the lookup table using TD-Learning."""
        current_value = self.evaluate_board(board)
        next_value = self.evaluate_board(next_board)
        reward = self.evaluate(board)

        td_error = (reward + self.discount_factor * next_value) - current_value

        for t in self.extract_tuples(board):
            self.lookup_table[t] += (
                self.learning_rate * td_error
            )  # Adjust tuple weights

    def get_best_move(self, game: Game2048):
        """Chooses the best move based on lookup table estimates."""
        best_move = None
        best_value = -float("inf")

        for move in game.get_possible_moves():
            new_board, _ = game.execute_player_move(move)
            if np.array_equal(game.board, new_board):
                continue
            value = self.evaluate_board(new_board)

            if value > best_value:
                best_value = value
                best_move = move

        return best_move


def __main__():
    agent = NTupleAgent()
    for epoch in range(5000):
        game = Game2048()
        i = 0
        while action := agent.get_best_move(game):
            prev_board = game.board.copy()
            prev_score = game.score

            # print("Next Player Move: ", action)
            game.execute_player_move(action, save_move=True)
            game.add_random_tile()
            # game.print_board()

            reward = game.score - prev_score  # Reward is score change
            agent.update_lookup_table(prev_board, game.board)  # Train N-Tuple

            i += 1

        if epoch % 100 == 0:
            print("Epoch: ", epoch)
            print("Final Score: ", game.score)
            print("Number of Moves: ", i)
            print("Final Board")
            game.print_board()
            print()


if __name__ == "__main__":
    __main__()
