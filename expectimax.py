from game import Game2048


class ExpectimaxAgent:
    def __init__(self, depth: int):
        self.depth = depth

    def get_action(self, game: Game2048):
        best_move, _ = self.expectimax(game, self.depth, True)
        return best_move

    def expectimax(
        self, game: Game2048, depth: int, is_player_turn: bool
    ) -> tuple[str | None, float]:
        if depth == 0 or game.is_game_over():
            return None, game.score

        if is_player_turn:
            max_eval = float("-inf")
            best_move = None
            for move in game.get_possible_moves():
                new_board, score = game.generate_move(move)
                _, eval = self.expectimax(
                    Game2048(new_board, score, False), depth - 1, False
                )
                if eval > max_eval:
                    max_eval = eval
                    best_move = move
            return best_move, max_eval
        else:
            expected_value = 0
            possible_tiles = get_possible_tiles()
            for tile, prob in possible_tiles:
                new_board = add_tile_to_board(board, tile)
                _, eval = self.expectimax(new_board, depth - 1, True)
                expected_value += prob * eval
            return None, expected_value

    def evaluate(self, board):
        # Define your heuristic evaluation function here
        pass
