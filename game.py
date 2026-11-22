"""Core rules for the 2048 puzzle game.

Two ways of holding a board are used:

* ``GameState.grid`` is a list of 4 lists. The real game changes it in place.
* A *board tuple* is a tuple of 4 row tuples. Tuples cannot be changed, so they
  are safe to share and can be used as dictionary keys. The AI searches with
  board tuples because that needs no copying.

Both use the same ``slide_board`` function, so the game and the AI always
follow exactly the same rules.
"""

import random
from functools import lru_cache


SIZE = 4
DIRECTIONS = ("up", "down", "left", "right")


@lru_cache(maxsize=None)
def _slide_row_left(row):
    """Slide one row (a tuple) to the left; return (new row tuple, points).

    ``lru_cache`` remembers every answer. The AI meets the same rows again and
    again, so after the first time each row costs only a dictionary lookup.
    """
    numbers = [n for n in row if n != 0]
    merged = []
    points = 0
    index = 0
    while index < len(numbers):
        if index + 1 < len(numbers) and numbers[index] == numbers[index + 1]:
            value = numbers[index] * 2
            merged.append(value)
            points += value
            index += 2  # skip the partner, so a tile merges at most once
        else:
            merged.append(numbers[index])
            index += 1
    return tuple(merged + [0] * (SIZE - len(merged))), points


def merge_line(line):
    """Slide a row toward its start; return (new row, score earned)."""
    result, points = _slide_row_left(tuple(line))
    return list(result), points


def slide_board(board, direction):
    """Return (new board tuple, points) after a move. Never spawns a tile.

    Every direction reuses the "slide left" rule:
    right = reverse each row, slide left, reverse back;
    up/down = turn columns into rows with ``zip(*board)``, then do the same.
    """
    vertical = direction in ("up", "down")
    backwards = direction in ("right", "down")
    lines = zip(*board) if vertical else board
    new_lines = []
    points = 0
    for line in lines:
        if backwards:
            result, earned = _slide_row_left(line[::-1])
            result = result[::-1]
        else:
            result, earned = _slide_row_left(line)
        new_lines.append(result)
        points += earned
    if vertical:
        return tuple(zip(*new_lines)), points
    return tuple(new_lines), points


def possible_moves(board):
    """Return (direction, new board, points) for each move that changes the board."""
    moves = []
    for direction in DIRECTIONS:
        new_board, points = slide_board(board, direction)
        if new_board != board:
            moves.append((direction, new_board, points))
    return moves


def can_move(board):
    """Fast game-over test: True if the board has an empty cell or equal neighbours."""
    for r in range(SIZE):
        for c in range(SIZE):
            value = board[r][c]
            if value == 0:
                return True
            if c + 1 < SIZE and board[r][c + 1] == value:
                return True
            if r + 1 < SIZE and board[r + 1][c] == value:
                return True
    return False


class GameState:
    """A board and its score; supports deterministic moves and seeded spawning."""

    def __init__(self, grid=None, score=0, seed=None, start_tiles=True):
        self.rng = random.Random(seed)
        self.grid = [list(row) for row in grid] if grid is not None else [[0] * SIZE for _ in range(SIZE)]
        if len(self.grid) != SIZE or any(len(row) != SIZE for row in self.grid):
            raise ValueError("Board must be 4x4")
        if any(not isinstance(n, int) or isinstance(n, bool) or n < 0 or
               (n != 0 and n & (n - 1) != 0) or n == 1
               for row in self.grid for n in row):
            raise ValueError("Tiles must be zero or powers of two starting at 2")
        self.score = score
        if grid is None and start_tiles:
            self.spawn()
            self.spawn()

    def copy(self):
        other = GameState(self.grid, score=self.score, start_tiles=False)
        other.rng.setstate(self.rng.getstate())
        return other

    def to_tuple(self):
        """Return the board as a tuple of row tuples (the AI's search format)."""
        return tuple(tuple(row) for row in self.grid)

    def empty_cells(self):
        return [(r, c) for r in range(SIZE) for c in range(SIZE) if self.grid[r][c] == 0]

    def spawn(self):
        """Add a 2 (90%) or 4 (10%) to an empty cell; return whether one appeared."""
        empty = self.empty_cells()
        if not empty:
            return False
        r, c = self.rng.choice(empty)
        self.grid[r][c] = 2 if self.rng.random() < 0.9 else 4
        return True

    def move(self, direction):
        """Apply a direction without spawning. Return True only if board changed."""
        direction = direction.lower()
        if direction not in DIRECTIONS:
            raise ValueError("Direction must be up, down, left, or right")
        board = self.to_tuple()
        new_board, points = slide_board(board, direction)
        if new_board == board:
            return False
        self.grid = [list(row) for row in new_board]
        self.score += points
        return True

    def play(self, direction):
        """Make a complete game turn: move and spawn only after a legal move."""
        if not self.move(direction):
            return False
        self.spawn()
        return True

    def legal_moves(self):
        return [direction for direction, _, _ in possible_moves(self.to_tuple())]

    def is_terminal(self):
        return not self.legal_moves()

    def highest_tile(self):
        return max(max(row) for row in self.grid)

    def reached_2048(self):
        return self.highest_tile() >= 2048
