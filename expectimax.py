from game import Game2048
import numpy as np


class ExpectimaxAgent:
    def __init__(self, depth: int):
        self.depth = depth

    def get_action(self, game: Game2048):
        best_move, _ = self.expectimax(game, self.depth, True)
        return best_move

    def expectimax(
        self, game: Game2048, depth: int, is_player_turn: bool
    ) -> tuple[str | None, float]:
        if depth == 0:
            return None, self.evaluate(game.board)

        if is_player_turn:
            max_eval = float("-inf")
            best_move = None
            for move in game.get_possible_moves():
                new_board, score = game.execute_player_move(move)
                # Skip moves which don't change the board
                if np.array_equal(game.board, new_board):
                    continue
                _, eval = self.expectimax(
                    Game2048(board=new_board, score=score, is_new_game=False),
                    depth - 1,
                    False,
                )
                if eval > max_eval:
                    max_eval = eval
                    best_move = move
            return best_move, max_eval
        else:
            expected_value, possible_moves = 0, game.get_possible_computer_moves()
            for row, col in possible_moves:
                for tile, prob in game.get_tiles_and_probabilities():
                    new_game = Game2048(
                        board=game.board.copy(),
                        score=game.score,
                        is_new_game=False,
                    )
                    new_game.execute_computer_move(tile, row, col)
                    _, eval = self.expectimax(new_game, depth - 1, True)
                    expected_value += prob * (1.0 / len(possible_moves)) * eval
            return None, expected_value

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


def __main__():
    game = Game2048()
    agent = ExpectimaxAgent(depth=4)
    while action := agent.get_action(game):
        print("Next Player Move: ", action)
        game.execute_player_move(action, save_move=True)
        game.add_random_tile()
        game.print_board()


if __name__ == "__main__":
    __main__()
