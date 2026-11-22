# 2048 AI with Expectimax Search

Artificial Intelligence Lab project — FAST NUCES.

## Requirements

Python 3.9 or newer. The project uses only the Python standard library.

## Run the game

From this folder:

```bash
python main.py
```

Controls: W = up, A = left, S = down, D = right, Q = quit.

## Run the AI demo

```bash
python demo.py
```

The demo asks the Random, Greedy, and depth-2 Expectimax agents to choose a
move from the same example board.

## Run the tests

From this folder:

```bash
python -m unittest discover -s tests -v
```

## Architecture

- `game.py`: 2048 rules, reproducible tile spawning, legal moves, and game-over checks.
- `agents.py`: Random, one-ply Greedy, and depth-limited Expectimax agents.
- `demo.py`: a small runnable example of the agents.
- `tests/test_game.py` and `tests/test_agents.py`: tests for the engine and agents.

`move(direction)` changes the board without spawning. This lets AI search
hypothetical states without randomness. `play(direction)` performs the real
turn and adds one random tile only after a successful move. Expectimax models
that spawn separately at a chance node.

## Phase 2: implemented

Expectimax uses MAX nodes for player moves and chance nodes for random spawns.
It models a 2 tile with 90% probability and a 4 tile with 10% probability,
uniformly over empty cells. Its depth counts player moves, including the move
being selected. The default evaluates every spawn outcome exactly; an optional
chance limit uses a deterministic approximation.

The board heuristic considers empty cells, monotonicity, smoothness, and the
largest tile in a corner. The agent also has a transposition table and counts
expanded nodes. See `README_PHASE2.md` for a beginner-friendly walkthrough and
viva notes.

## Planned next phases

1. Add a graphical interface.
2. Run seeded experiments and compare the agents.
3. Prepare charts, final documentation, and viva materials.
