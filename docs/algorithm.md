# How the project works

This file explains the code from the bottom up: the game engine, the
heuristic, the three agents, and the speed tricks. Every number in the dry
runs below was produced by the real code (you can re-run `python demo.py`).

## 1. Architecture

```text
 gui.py  (Tkinter window)        experiments.py (50 seeded games)     tune.py (grid search)
     |                                   |                                   |
     +------------------+----------------+-----------------------------------+
                        v
                  agents.py   RandomAgent | GreedyAgent | ExpectimaxAgent
                        |     evaluate_board (heuristic)
                        v
                  game.py     GameState (real game) + slide_board / possible_moves (pure rules)
                                                    |
 charts.py  <--- results/final/per_game.csv <-------+
```

* **game.py** holds *all* the rules. Nothing else knows how tiles merge.
* **agents.py** only asks the engine "what happens if I move this way?".
* **gui.py**, **experiments.py** and **tune.py** only use the engine and the agents.
* **charts.py** only reads CSV files, so charts can be redrawn without replaying games.

### Two board formats

| Format | Example | Used by | Why |
|---|---|---|---|
| `GameState.grid` (list of lists) | `[[2, 0, 0, 0], ...]` | the real game, GUI | can be changed in place |
| board tuple (tuple of tuples) | `((2, 0, 0, 0), ...)` | the AI search | cannot change, so no copying needed; can be a dictionary key for caching |

Both formats go through the same `slide_board()` function, so the AI and the
real game always follow identical rules.

## 2. The game engine (`game.py`)

**Sliding one row left** (`_slide_row_left`): remove the zeros, walk left to
right, merge two equal neighbours and *skip the partner* so a tile merges at
most once per move, then pad with zeros.

```text
[4, 4, 8, 0] -> numbers [4, 4, 8] -> 4+4 merge -> 8, skip; then 8 stays
             -> [8, 8, 0, 0], points 8      (the new 8 does NOT merge again)
```

**Other directions reuse the same rule.** Right = reverse the row, slide
left, reverse back. Up/down = `zip(*board)` turns columns into rows, slide,
turn back.

**A move is legal only if it changes the board.** `play()` = `move()` +
`spawn()`; an illegal move never spawns a tile.

**Spawning:** pick a random empty cell (all equally likely), put a 2 with
probability 0.9 or a 4 with probability 0.1. The random generator is seeded,
so seed 7 always produces exactly the same game for the same moves.

## 3. The heuristic (`evaluate_board` in `agents.py`)

The search cannot reach the end of the game, so when it stops it needs a
"how good does this board look?" number. Ours is a weighted sum of four
features. Tiles are compared as log2 (2→1, 4→2, 8→3 …) so that 1024 vs 2048
counts as one step, just like 2 vs 4.

| Feature | Meaning | Why it helps | Weight |
|---|---|---|---|
| empty | number of empty cells | room to move; a full board is close to game over | 100 |
| monotonicity | penalty when a row/column goes up *and* down | ordered lines can be merged in a chain | 20 |
| smoothness | penalty for big jumps between neighbours | similar neighbours can merge soon | 11 |
| corner | log2(largest tile) if it sits in a corner | classic human strategy | 0 (see tuning) |

**Monotonicity of one line:** add up the uphill steps and the downhill steps;
the penalty is the smaller of the two. A perfectly ordered line has penalty 0.

```text
(2, 4, 8, 16) -> logs 1,2,3,4 -> uphill 3, downhill 0 -> penalty -min(3,0) = 0
(2, 16, 4, 8) -> logs 1,4,2,3 -> uphill 3+1=4, downhill 2 -> penalty -2
```

> **Bug fixed during this phase:** the earlier version computed
> `sum(right - left)`, which telescopes to `last - first`, so it only compared
> the two end tiles. Fixing it raised the depth-1 agent's average score a lot.

**Worked example** for this board (weights 100 / 20 / 11 / 0):

```text
 4   8  16   2      rows:    mono -2, 0, 0, 0     smooth -5, -3, -3, -3
 4   8  16  32      columns: mono  0, 0, 0, 0     smooth -2, -2, -2, -6
 8  16  32  64      empty = 0, monotonicity = -2, smoothness = -26
16  32  64 128      value = 100*0 + 20*(-2) + 11*(-26) = -326
```

Because each feature is a sum over the 4 rows and 4 columns, the features of
each *line* are cached (`@lru_cache` on `_line_features`). After the first
time, a line costs one dictionary lookup.

**Game over** is not scored by the heuristic at all: a board with no legal
move gets `LOSS_VALUE = -100000`, far below any normal value. So even a 10%
chance of losing makes a move look clearly bad.

## 4. The agents

* **RandomAgent** – picks any legal move. The "no intelligence" baseline.
* **GreedyAgent** – tries each legal move once and keeps the board with the
  best heuristic. It looks one move ahead and **ignores the random tile**.
* **ExpectimaxAgent** – looks `depth` moves ahead and **averages over every
  possible random tile**.

### Expectimax

Two kinds of node alternate:

```text
MAX node     (our turn)    value = the best child value
CHANCE node  (game's turn) value = sum over every empty cell and tile of
                                   P(cell) * P(tile) * value of the child
                                   P(cell) = 1 / number of empty cells
                                   P(tile) = 0.9 for a 2, 0.1 for a 4
Leaf                       value = heuristic (or LOSS_VALUE if no move is possible)
```

Depth counts *our* moves. Depth 1 = make a move, average over the spawn,
then evaluate.

```python
def _max_value(self, board, depth):          # simplified from agents.py
    if not can_move(board): return LOSS_VALUE
    if depth == 0:          return evaluate_board(board)
    return max(self._chance_value(child, depth) for _, child, _ in possible_moves(board))

def _chance_value(self, board, depth):
    return sum(p * self._max_value(child, depth - 1)
               for child, p in self._spawn_outcomes(board))
```

### Dry run: why the chance node matters

```text
 2   2   8  16
 4   8  16  32       Legal moves: left and right (no vertical merges)
 8  16  32  64
16  32  64 128
```

**Greedy** evaluates the boards right after the move: left = −149,
right = −142, so it picks **right**.

**Expectimax depth 1** also looks at the tile that will appear:

```text
LEFT  -> 4 8 16 [ ]   one empty cell (0,3)
         spawn 2 (p=0.9): heuristic -326
         spawn 4 (p=0.1): heuristic -304
         chance value = 0.9*(-326) + 0.1*(-304) = -323.8

RIGHT -> [ ] 4 8 16   one empty cell (0,0)
         spawn 2 (p=0.9): board is full, no equal neighbours -> GAME OVER -> -100000
         spawn 4 (p=0.1): heuristic -242
         chance value = 0.9*(-100000) + 0.1*(-242) = -90024.2

MAX: left (-323.8) > right (-90024.2)  ->  choose LEFT
```

Greedy's move loses the game 90% of the time. Expectimax avoids it because it
weighs every random outcome by its probability. This is the whole idea of the
project in one example. (6 nodes were expanded: root's 2 chance nodes + 4 leaves.)

### Why Expectimax and not Minimax?

Minimax assumes an opponent picks the *worst* tile for us. In 2048 the tile is
random, not hostile; assuming the worst makes the agent far too cautious. The
spawn probabilities are known, so the correct thing is the expected
(probability-weighted average) value.

## 5. Making Python fast enough

The original version needed 385 ms per move at depth 2 (168 s for one game)
because every search step copied a full `GameState`, including its random
number generator. Profiling showed 65% of the time was spent copying.

| Change | Idea in one sentence |
|---|---|
| Board tuples in search | A move builds a new small tuple instead of copying a whole object. |
| `lru_cache` on row sliding | There are few distinct rows, so each is computed once and then looked up. |
| `lru_cache` on line features | Same trick for the heuristic. |
| `can_move()` | A board with an empty cell can always move, so most leaves skip trying all 4 moves. |
| Transposition table | Inside one decision, the same board can be reached in different orders (left-then-up = up-then-left); its value is stored by (node type, board, depth) and reused. `nodes_expanded` counts only real computations, not cache hits. |
| `--workers` in experiments | Games are independent, so 7 run at once on different CPU cores. Seeds make the results identical to running them one by one. |

Result: depth 2 went from **385 ms to about 17 ms per move** with identical
decisions, and depth 3 became practical. A test (`test_caching_does_not_change_values`)
checks the cached search against a plain textbook Expectimax with no cache.

### Complexity

With *e* empty cells, a chance node has 2*e* children and a MAX node up to 4.
So one level of lookahead multiplies the work by roughly 4 × 2*e* (about 50–100
in mid-game). In `demo.py` the node count grows 73 → 1,928 → 29,679 for depths
1 → 2 → 3. That is why depth 4 is impractical in pure Python and why caching
matters.

### Optional chance-node limit

`ExpectimaxAgent(max_chance_outcomes=N)` (and `--max-chance-outcomes` in
experiments) replaces the full spawn list by N representative outcomes, one per
equal-probability slice. It is an approximation for speed; by default it is
off and the search is exact. The final results do not use it.
