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
            return None, game.score

        if is_player_turn:
            max_eval = game.score
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

    def evaluate(self, board):
        # Define your heuristic evaluation function here
        pass
