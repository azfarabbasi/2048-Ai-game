> **Historical phase notes.** The up-to-date explanation is in `docs/algorithm.md`, `docs/results.md` and `docs/viva_guide.md`. (The heuristic weights mentioned here were later re-tuned and a monotonicity bug was fixed.)

# Phase 2: Random, Greedy, and Expectimax

Copy `agents.py`, `demo.py`, and `tests/test_agents.py` into the matching places
in the existing project. Keep the current `game.py`; the agents use its
`copy()`, `move()`, `legal_moves()`, and `empty_cells()` methods.

## Run

From the project folder:

```bash
python demo.py
python -m unittest discover -s tests -v
```

## Beginner's guide

1. **RandomAgent** asks the game for legal moves and picks one with Python's
   seeded random generator. It returns `None` when the game is over.
2. **GreedyAgent** copies the board for each legal move, makes only that move,
   and chooses the result with the best heuristic. It does not add a random
   tile, so this is a one-ply baseline.
3. **ExpectimaxAgent** considers a move at a MAX node. It then considers every
   random tile spawn at a CHANCE node and averages the future values using the
   spawn probabilities. It repeats this until its depth limit.
4. **evaluate_board** scores empty cells, row/column monotonicity, smooth
   neighboring tiles, and whether a largest tile is in a corner. Default
   weights are simple starting values, not experimentally tuned weights.
5. **Transposition table** stores a result by board, node type, and remaining
   depth, avoiding repeated work in the same move decision. `nodes_expanded`
   counts MAX and chance nodes that were calculated, but not cache hits.

## Depth and chance approximation

Depth counts player moves, including the move being selected. After each move,
one tile spawn is modeled. Thus depth 1 means: make one move, average over its
spawn, then apply the heuristic; it does not mean one MAX layer plus one extra
future move. The full distribution is evaluated by default: each empty cell is
equally likely, and its tile is 2 with probability 0.9 or 4 with probability
0.1.

For speed, `ExpectimaxAgent(max_chance_outcomes=N)` approximates a chance node
by splitting the full cumulative probability distribution into N equal-mass
intervals and evaluating one midpoint outcome per interval. This is a
deterministic estimate, while the default (`None`) is exact. The game engine's
real spawning rule remains unchanged.

## Viva summary

Expectimax is suitable because the game has a known random outcome model. MAX
nodes represent our choices; chance nodes represent tile spawns. A weighted
average models expected value, while minimax would incorrectly assume an
opponent chooses the worst outcome. The heuristic estimates board quality
when the search stops. Search can still be expensive because many cells can
receive a tile, which is why caching and an optional chance limit are useful.
