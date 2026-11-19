"""Core rules for the 2048 puzzle game."""

import random


SIZE = 4
DIRECTIONS = ("up", "down", "left", "right")


def merge_line(line):
    """Slide a row toward its start; return (new row, score earned)."""
    numbers = [n for n in line if n != 0]
    merged = []
    points = 0
    index = 0
    while index < len(numbers):
        if index + 1 < len(numbers) and numbers[index] == numbers[index + 1]:
            value = numbers[index] * 2
            merged.append(value)
            points += value
            index += 2
        else:
            merged.append(numbers[index])
            index += 1
    return merged + [0] * (SIZE - len(merged)), points


class GameState:
    """A board and its score; supports deterministic moves and seeded spawning."""

    def __init__(self, grid=None, score=0, seed=None, start_tiles=True):
        self.rng = random.Random(seed)
        self.grid = [row[:] for row in grid] if grid is not None else [[0] * SIZE for _ in range(SIZE)]
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

        before = [row[:] for row in self.grid]
        points = 0
        for i in range(SIZE):
            vertical = direction in ("up", "down")
            reversed_line = direction in ("right", "down")
            line = [self.grid[r][i] for r in range(SIZE)] if vertical else self.grid[i][:]
            if reversed_line:
                line.reverse()
            result, earned = merge_line(line)
            points += earned
            if reversed_line:
                result.reverse()
            if vertical:
                for r in range(SIZE):
                    self.grid[r][i] = result[r]
            else:
                self.grid[i] = result

        changed = self.grid != before
        if changed:
            self.score += points
        return changed

    def play(self, direction):
        """Make a complete game turn: move and spawn only after a legal move."""
        if not self.move(direction):
            return False
        self.spawn()
        return True

    def legal_moves(self):
        moves = []
        for direction in DIRECTIONS:
            simulation = self.copy()
            if simulation.move(direction):
                moves.append(direction)
        return moves

    def is_terminal(self):
        return not self.legal_moves()

    def highest_tile(self):
        return max(max(row) for row in self.grid)

    def reached_2048(self):
        return self.highest_tile() >= 2048
