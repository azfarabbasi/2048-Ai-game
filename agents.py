"""Simple 2048 agents and an Expectimax search implementation.

Depth counts player moves. Each player move is followed by one complete
chance node (the random tile spawn), so depth=1 is a one-move search.
"""

import math
import random

from game import DIRECTIONS, SIZE


def _board_key(state):
    return tuple(value for row in state.grid for value in row)


def _log_tile(value):
    return math.log2(value) if value else 0.0


def evaluate_board(state, weights=None):
    """Return a weighted score for a board; larger values are better."""
    weights = weights or {
        "empty": 270.0,
        "monotonicity": 47.0,
        "smoothness": 11.0,
        "corner": 10.0,
    }
    grid = state.grid
    logs = [[_log_tile(value) for value in row] for row in grid]
    empty_count = sum(value == 0 for row in grid for value in row)

    # Penalize neighboring tiles whose exponents differ greatly.
    smoothness = 0.0
    for r in range(SIZE):
        for c in range(SIZE):
            if logs[r][c] == 0:
                continue
            if c + 1 < SIZE and logs[r][c + 1] != 0:
                smoothness -= abs(logs[r][c] - logs[r][c + 1])
            if r + 1 < SIZE and logs[r + 1][c] != 0:
                smoothness -= abs(logs[r][c] - logs[r + 1][c])

    # For each row and column, reward either increasing or decreasing order.
    monotonicity = 0.0
    for line in logs + [list(column) for column in zip(*logs)]:
        increasing = decreasing = 0.0
        for left, right in zip(line, line[1:]):
            increasing += right - left
            decreasing += left - right
        monotonicity -= max(increasing, decreasing)

    largest = max(max(row) for row in grid)
    corners = (grid[0][0], grid[0][-1], grid[-1][0], grid[-1][-1])
    corner = _log_tile(largest) if largest and largest in corners else 0.0

    return (
        weights["empty"] * empty_count
        + weights["monotonicity"] * monotonicity
        + weights["smoothness"] * smoothness
        + weights["corner"] * corner
    )


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

    def choose_move(self, state):
        best_move, best_value = None, -math.inf
        for direction in DIRECTIONS:
            child = state.copy()
            if child.move(direction):
                value = evaluate_board(child, self.weights)
                if value > best_value:
                    best_move, best_value = direction, value
        return best_move


class ExpectimaxAgent:
    """Depth-limited Expectimax with MAX moves, chance spawns, and caching.

    ``max_chance_outcomes`` is optional. When set, the complete spawn
    distribution is divided into that many equal-probability intervals and
    the midpoint outcome of each interval is evaluated. This is a deterministic
    approximation to the exact chance expectation; the default is exact.
    ``nodes_expanded`` counts MAX and chance nodes actually evaluated, excluding
    results served from the transposition table.
    """

    def __init__(self, depth=3, weights=None, max_chance_outcomes=None):
        if not isinstance(depth, int) or depth < 1:
            raise ValueError("depth must be a positive integer")
        if max_chance_outcomes is not None and max_chance_outcomes < 1:
            raise ValueError("max_chance_outcomes must be at least 1")
        self.depth = depth
        self.weights = dict(weights) if weights is not None else None
        self.max_chance_outcomes = max_chance_outcomes
        self.nodes_expanded = 0
        self.transposition_table = {}

    def choose_move(self, state):
        self.nodes_expanded = 0
        self.transposition_table = {}
        best_move, best_value = None, -math.inf
        for direction in DIRECTIONS:
            child = state.copy()
            if not child.move(direction):
                continue
            value = self._chance_value(child, self.depth)
            if value > best_value:
                best_move, best_value = direction, value
        return best_move

    def _max_value(self, state, depth):
        key = ("max", _board_key(state), depth)
        if key in self.transposition_table:
            return self.transposition_table[key]
        self.nodes_expanded += 1
        moves = state.legal_moves()
        if depth == 0 or not moves:
            value = evaluate_board(state, self.weights)
        else:
            value = -math.inf
            for direction in moves:
                child = state.copy()
                child.move(direction)
                value = max(value, self._chance_value(child, depth))
        self.transposition_table[key] = value
        return value

    def _spawn_outcomes(self, state):
        empty = state.empty_cells()
        if not empty:
            return [(state, 1.0)]
        outcomes = []
        cell_probability = 1.0 / len(empty)
        for r, c in empty:
            for tile, tile_probability in ((2, 0.9), (4, 0.1)):
                child = state.copy()
                child.grid[r][c] = tile
                outcomes.append((child, cell_probability * tile_probability))
        limit = self.max_chance_outcomes
        if limit is None or len(outcomes) <= limit:
            return outcomes

        # Stratified midpoint approximation: each representative stands for
        # one equal-probability slice of the full distribution.
        sampled = []
        for index in range(limit):
            midpoint = (index + 0.5) / limit
            cumulative = 0.0
            for child, probability in outcomes:
                cumulative += probability
                if midpoint <= cumulative:
                    sampled.append((child, 1.0 / limit))
                    break
        return sampled

    def _chance_value(self, state, depth):
        key = ("chance", _board_key(state), depth)
        if key in self.transposition_table:
            return self.transposition_table[key]
        self.nodes_expanded += 1
        value = sum(
            probability * self._max_value(child, depth - 1)
            for child, probability in self._spawn_outcomes(state)
        )
        self.transposition_table[key] = value
        return value
