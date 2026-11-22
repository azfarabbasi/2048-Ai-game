"""Simple 2048 agents and an Expectimax search implementation.

Depth counts player moves. Each player move is followed by one complete
chance node (the random tile spawn), so depth=1 is a one-move search.

The search works on board tuples (see ``game.py``): a move creates a new tuple
instead of copying a whole ``GameState``, and tuples can be cache keys.
"""

import math
import random
from functools import lru_cache

from game import SIZE, can_move, possible_moves


# Chosen by tune.py (grid search on tuning seeds 51-70, best average score at
# depth 2; see results/tuning/). The corner weight came out as 0: monotonicity
# already pushes the largest tile into a corner, so the extra bonus did not help.
DEFAULT_WEIGHTS = {
    "empty": 100.0,
    "monotonicity": 20.0,
    "smoothness": 11.0,
    "corner": 0.0,
}

# Value of a board with no legal moves. Far below any normal heuristic value,
# so even a small chance of losing makes a move look clearly worse.
LOSS_VALUE = -100_000.0


def _log_tile(value):
    return math.log2(value) if value else 0.0


@lru_cache(maxsize=None)
def _line_features(line):
    """Return (empty cells, monotonicity, smoothness) for one row or column.

    The board heuristic is a sum over its 4 rows and 4 columns, so caching each
    line's features makes evaluation fast. Tile values are compared as log2
    (2 -> 1, 4 -> 2, 8 -> 3, ...) so that 1024 vs 2048 counts as one step.
    """
    logs = [_log_tile(value) for value in line]
    empty = sum(value == 0 for value in line)

    # Smoothness: penalize neighbouring tiles whose exponents differ greatly.
    smoothness = 0.0
    for left, right in zip(logs, logs[1:]):
        if left and right:
            smoothness -= abs(left - right)

    # Monotonicity: add up the "uphill" steps and the "downhill" steps.
    # A perfectly ordered line has one of them equal to 0, so no penalty.
    increasing = decreasing = 0.0
    for left, right in zip(logs, logs[1:]):
        if right > left:
            increasing += right - left
        else:
            decreasing += left - right
    monotonicity = -min(increasing, decreasing)

    return empty, monotonicity, smoothness


def evaluate_board(board, weights=None):
    """Return a weighted score for a board; larger values are better.

    ``board`` may be a ``GameState`` or a board tuple.
    """
    if hasattr(board, "to_tuple"):
        board = board.to_tuple()
    weights = weights or DEFAULT_WEIGHTS

    empty = monotonicity = smoothness = 0.0
    for row in board:
        row_empty, row_monotonicity, row_smoothness = _line_features(row)
        empty += row_empty
        monotonicity += row_monotonicity
        smoothness += row_smoothness
    for column in zip(*board):
        # Empty cells were already counted in the rows.
        _, column_monotonicity, column_smoothness = _line_features(column)
        monotonicity += column_monotonicity
        smoothness += column_smoothness

    largest = max(max(row) for row in board)
    corners = (board[0][0], board[0][-1], board[-1][0], board[-1][-1])
    corner = _log_tile(largest) if largest and largest in corners else 0.0

    return (
        weights["empty"] * empty
        + weights["monotonicity"] * monotonicity
        + weights["smoothness"] * smoothness
        + weights["corner"] * corner
    )


def place_tile(board, r, c, tile):
    """Return a new board tuple with ``tile`` placed at row r, column c."""
    row = board[r][:c] + (tile,) + board[r][c + 1:]
    return board[:r] + (row,) + board[r + 1:]


class RandomAgent:
    """Choose a random legal move."""

    def __init__(self, seed=None):
        self.rng = random.Random(seed)

    def choose_move(self, state):
        moves = state.legal_moves()
        return self.rng.choice(moves) if moves else None


class GreedyAgent:
    """Choose the legal move with the best immediate heuristic score."""

    def __init__(self, weights=None):
        self.weights = dict(weights) if weights is not None else None
        self.move_values = {}  # direction -> heuristic value, for display

    def choose_move(self, state):
        self.move_values = {}
        best_move, best_value = None, -math.inf
        for direction, child, _ in possible_moves(state.to_tuple()):
            value = evaluate_board(child, self.weights)
            self.move_values[direction] = value
            if value > best_value:
                best_move, best_value = direction, value
        return best_move


class ExpectimaxAgent:
    """Depth-limited Expectimax with MAX moves, chance spawns, and caching.

    ``nodes_expanded`` counts the MAX and chance nodes actually computed in the
    last decision (answers found in the transposition table are not counted).
    """

    def __init__(self, depth=3, weights=None):
        if not isinstance(depth, int) or depth < 1:
            raise ValueError("depth must be a positive integer")
        self.depth = depth
        self.weights = dict(weights) if weights is not None else None
        self.nodes_expanded = 0
        self.transposition_table = {}
        self.move_values = {}  # direction -> expected value, for display

    def choose_move(self, state):
        """Root MAX node: return the move with the highest expected value."""
        self.nodes_expanded = 0
        self.transposition_table = {}
        self.move_values = {}
        best_move, best_value = None, -math.inf
        for direction, child, _ in possible_moves(state.to_tuple()):
            value = self._chance_value(child, self.depth)
            self.move_values[direction] = value
            if value > best_value:
                best_move, best_value = direction, value
        return best_move

    def _max_value(self, board, depth):
        """MAX node: the player picks the best move (or the search stops)."""
        key = ("max", board, depth)
        if key in self.transposition_table:
            return self.transposition_table[key]
        self.nodes_expanded += 1
        if not can_move(board):
            value = LOSS_VALUE  # game over: the worst possible outcome
        elif depth == 0:
            value = evaluate_board(board, self.weights)  # search limit reached
        else:
            value = max(
                self._chance_value(child, depth) for _, child, _ in possible_moves(board)
            )
        self.transposition_table[key] = value
        return value

    def _spawn_outcomes(self, board):
        """List (board after spawn, probability) for every possible new tile."""
        # After a legal move there is always at least one empty cell.
        empty = [(r, c) for r in range(SIZE) for c in range(SIZE) if board[r][c] == 0]
        outcomes = []
        for r, c in empty:
            for tile, tile_probability in ((2, 0.9), (4, 0.1)):
                probability = (1 / len(empty)) * tile_probability
                outcomes.append((place_tile(board, r, c, tile), probability))
        return outcomes

    def _chance_value(self, board, depth):
        """CHANCE node: probability-weighted average over all tile spawns."""
        key = ("chance", board, depth)
        if key in self.transposition_table:
            return self.transposition_table[key]
        self.nodes_expanded += 1
        value = sum(
            probability * self._max_value(child, depth - 1)
            for child, probability in self._spawn_outcomes(board)
        )
        self.transposition_table[key] = value
        return value
