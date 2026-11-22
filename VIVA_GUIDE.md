# Viva guide: how the 2048 AI works

Read this top to bottom once and you can explain the whole project. Every
number in the examples was produced by the real code.

1. [The big picture](#1-the-big-picture)
2. [Game engine](#2-game-engine-gamepy)
3. [Heuristic](#3-heuristic-evaluate_board-in-agentspy)
4. [The three agents and Expectimax](#4-the-three-agents)
5. [Dry run: why chance nodes matter](#5-dry-run-why-chance-nodes-matter)
6. [Speed](#6-making-python-fast-enough)
7. [Experiments](#7-experiments)
8. [Demo plan](#8-demo-plan-5-minutes)
9. [Questions and answers](#9-questions-and-answers)

## 1. The big picture

Five Python files, each with one job, in layers. A file only uses the files
below it:

```text
   main.py           experiments.py          charts.py
   (window)          (play many games,       (CSV -> pictures)
      |               write CSV)                  ^
      |                    |                      |
      +--------+-----------+          results/final/per_game.csv
               v
          agents.py      "which move is best?"   Random, Greedy, Expectimax + heuristic
               |
               v
           game.py       "what happens if I move?"   all the rules of 2048
```

* **game.py** knows the rules. Nothing else knows how tiles merge.
* **agents.py** decides moves. It only asks game.py "what happens if...".
* **main.py** shows the board and calls an agent. It has no rules of its own.
* **experiments.py** plays hundreds of games with no window and saves numbers.
* **charts.py** draws charts only from the saved CSV, so nothing is typed by hand.
* **tests.py** checks all of the above.

One game turn: `main.py` asks the chosen agent for a move →
`agents.py` searches possible futures using `game.py` → `main.py` calls
`state.play(move)` → the board is redrawn.

## 2. Game engine (`game.py`)

**Two board formats.** The real game keeps `GameState.grid`, a list of lists
that it changes in place. The AI searches with *board tuples*
(`((2, 0, 0, 0), ...)`): tuples cannot change, so they never need copying and
can be used as dictionary keys for caching. Both go through the same
`slide_board()`, so the AI and the game always follow identical rules.

**Sliding one row left** (`_slide_row_left`): remove zeros, merge equal
neighbours from left to right, skip the partner so a tile merges at most once,
pad with zeros.

```text
[4, 4, 8, 0] -> 4+4 = 8, skip; 8 stays -> [8, 8, 0, 0], +8 points
               (the new 8 does NOT merge again in the same move)
```

**Other directions reuse that rule.** Right = reverse, slide left, reverse
back. Up/down = `zip(*board)` turns columns into rows, slide, turn back.

**Turns.** `move()` slides without spawning and returns False if nothing
changed (illegal move). `play()` = `move()` + `spawn()`, so an illegal move never
adds a tile. `spawn()` picks a random empty cell and places a 2 (90%) or 4
(10%). The random generator is seeded: seed 7 always gives the same game for
the same moves.

## 3. Heuristic (`evaluate_board` in `agents.py`)

The search cannot reach the end of the game, so at the depth limit it asks:
"how good does this board look?" The answer is a weighted sum of features.
Tiles are compared as log2 (2→1, 4→2, 8→3 …) so 1024 vs 2048 is one step, like 2 vs 4.

| Feature | Meaning | Why it helps | Weight |
|---|---|---|---|
| empty | number of empty cells | room to move | 100 |
| monotonicity | penalty when a row/column goes both up and down | ordered lines merge in a chain (2-4-8-16) | 20 |
| smoothness | penalty for big jumps between neighbours | similar neighbours merge soon | 11 |
| corner | log2(largest tile) if it is in a corner | classic human strategy | 0 |

**Monotonicity of a line:** add the uphill steps and the downhill steps; the
penalty is the smaller of the two, so a perfectly ordered line gets 0.

```text
(2, 4, 8, 16) -> logs 1,2,3,4 -> uphill 3, downhill 0 -> penalty 0
(2, 16, 4, 8) -> logs 1,4,2,3 -> uphill 3+1 = 4, downhill 2 -> penalty -2
```

**Worked example** (weights 100 / 20 / 11 / 0):

```text
 4   8  16   2      rows:    monotonicity -2, 0, 0, 0   smoothness -5, -3, -3, -3
 4   8  16  32      columns: monotonicity  0, 0, 0, 0   smoothness -2, -2, -2, -6
 8  16  32  64      empty 0, monotonicity -2, smoothness -26
16  32  64 128      value = 100*0 + 20*(-2) + 11*(-26) = -326
```

**Game over** is not scored by the heuristic: a board with no legal move gets
`LOSS_VALUE = -100000`, far below any normal value, so even a small chance of
losing makes a move look bad.

**How the weights were chosen:** a grid search (`python experiments.py --tune`)
tried 81 combinations at depth 1 on seeds 51–70, then re-tested the best 5 plus
the old defaults at depth 2. The depth-1 winner was *not* the depth-2 winner,
so depth 2 decided. The corner weight came out as 0: monotonicity already pushes
the largest tile into a corner, so the extra bonus did not help.

> **Bug found and fixed:** the earlier monotonicity code computed
> `sum(right - left)`, which simplifies to `last - first`, so it only compared
> the two end tiles of a line. Fixing it greatly improved play.

## 4. The three agents

* **RandomAgent** – any legal move. The "no intelligence" baseline.
* **GreedyAgent** – tries each move once, keeps the best-looking board. Looks one
  move ahead and *ignores the random tile*.
* **ExpectimaxAgent** – looks `depth` moves ahead and *averages over every
  possible random tile*.

### Expectimax

```text
MAX node     (our turn)    value = best child value
CHANCE node  (game's turn) value = sum of P(outcome) * value(outcome)
                           over every empty cell x {2, 4}
                           P = 1/(number of empty cells) * 0.9 for a 2, * 0.1 for a 4
Leaf                       value = heuristic, or LOSS_VALUE if no move is possible
```

Depth counts *our* moves. Depth 2 = our move, random tile, our move, random
tile, evaluate. The code (simplified from `agents.py`):

```python
def _max_value(self, board, depth):              # MAX node
    if not can_move(board): return LOSS_VALUE
    if depth == 0:          return evaluate_board(board)
    return max(self._chance_value(child, depth) for _, child, _ in possible_moves(board))

def _chance_value(self, board, depth):           # CHANCE node
    return sum(p * self._max_value(child, depth - 1)
               for child, p in self._spawn_outcomes(board))
```

`choose_move` runs `_chance_value` for each legal move from the current board
and picks the highest. It also saves each move's value in `move_values`, which
the window shows under the score.

## 5. Dry run: why chance nodes matter

```text
 2   2   8  16
 4   8  16  32      Legal moves: left and right (nothing can merge vertically)
 8  16  32  64
16  32  64 128
```

**Greedy** scores the boards right after the move: left −149, right −142 →
picks **right**.

**Expectimax, depth 1** also considers the tile that will appear:

```text
LEFT  -> 4 8 16 [ ]  one empty cell
         spawn 2 (p = 0.9): heuristic -326
         spawn 4 (p = 0.1): heuristic -304
         chance value = 0.9*(-326) + 0.1*(-304) = -323.8

RIGHT -> [ ] 4 8 16  one empty cell
         spawn 2 (p = 0.9): full board, no equal neighbours -> GAME OVER -> -100000
         spawn 4 (p = 0.1): heuristic -242
         chance value = 0.9*(-100000) + 0.1*(-242) = -90024.2

MAX: left (-323.8) > right (-90024.2)  ->  LEFT
```

Greedy's move loses 90% of the time; Expectimax sees it because it weighs every
random outcome by its probability. This is the whole project in one example
(it is also a test: `test_expectimax_avoids_losing_move_that_greedy_takes`).

## 6. Making Python fast enough

The first version needed 385 ms per move at depth 2 (168 s per game). Profiling
showed 65% of the time went into copying `GameState` objects. Now:

| Trick | One-sentence idea |
|---|---|
| Board tuples | A move builds a small new tuple instead of copying a whole object. |
| `@lru_cache` on row sliding | Rows repeat constantly, so each row is computed once, then looked up. |
| `@lru_cache` on line features | Same idea for the heuristic (it is a sum over 4 rows + 4 columns). |
| `can_move()` | A board with an empty cell can always move, so most leaves skip trying 4 moves. |
| Transposition table | The same board can be reached by different move orders (left-then-up = up-then-left); its value is stored by (node type, board, depth) and reused within one decision. Leaves (depth 0) are not stored: they are the most common nodes and cheap to evaluate, and storing them once made the computer run out of memory during the 50-game experiment. |
| Parallel games | Experiments run 7 games at once on different CPU cores; seeds keep results identical. |

Depth 2 is now about **17 ms per move** with exactly the same decisions. A test
compares the cached search against a plain Expectimax with no cache.

**Exponential growth.** On the board `[[2,2,4,0],[4,8,0,0],[0,0,0,0],[0,0,0,0]]`
the agent expands 73 → 2,074 → 32,511 nodes at depth 1 → 2 → 3. With *e* empty
cells, each level multiplies the tree by about 4 moves × 2*e* spawns, i.e. the work
is roughly O((8e)^d). That is why depth 4 is impractical in pure Python. Check it:

```bash
python -c "from agents import *; from game import *; s=GameState([[2,2,4,0],[4,8,0,0],[0]*4,[0]*4]); [print(d, (a:=ExpectimaxAgent(d)).choose_move(s), a.nodes_expanded) for d in (1,2,3)]"
```

## 7. Experiments

`python experiments.py` plays Random, Greedy, and Expectimax at depths 1, 2, 3
(plus Expectimax without monotonicity, the *ablation*) on the same 50 seeds,
writes `results/final/per_game.csv` and `summary.csv`. `python charts.py` turns
them into the charts in `results/charts/`. The results and their meaning are in
the README.

**Why it is fair:** same seeds for every agent; a fresh agent per game; only
decision time is timed; weights tuned on different seeds (51–70) than the
evaluation (1–50).

## 8. Demo plan (5 minutes)

1. `python main.py` → Expectimax, depth 2, **Auto Play**. Point at the line under
   the score: each move's expected value and the number of nodes searched.
2. Stop, choose **Greedy**, press **AI Move**: instant, but no lookahead.
3. Switch Expectimax depth 1 → 2 → 3 and press **AI Move**: the node count
   jumps each time (exponential growth).
4. Open `results/charts/` and the README results table.
5. `python -m unittest tests -v` → all tests pass.

## 9. Questions and answers

**What is 2048?** A 4×4 sliding puzzle: equal tiles merge into their sum; after
every move a 2 (90%) or 4 (10%) appears in a random empty cell. Goal: a 2048
tile. The game ends when no move changes the board.

**What is an AI agent? PEAS?** Something that perceives its environment and acts
toward a goal. Performance: score, highest tile, % reaching 2048. Environment:
the board with random spawns. Actuators: up/down/left/right. Sensors: the full
board and score.

**Environment type?** Fully observable, stochastic, static, discrete,
single-agent. Our agent is utility-based: it maximises expected utility.

**What is a search algorithm?** A systematic exploration of possible future
states (a tree of "if I do this, then that happens") to choose an action.

**What is Expectimax?** A game-tree search with MAX nodes (our choice: take the
best child) and CHANCE nodes (random event: take the probability-weighted
average of the children).

**Minimax vs Expectimax?** Minimax has MIN nodes: an opponent choosing the worst
outcome for us. 2048 has no opponent, only random tiles with known
probabilities, so we average. Minimax would be far too pessimistic.

**MAX node / chance node?** See section 4. MAX = `_max_value`, chance =
`_chance_value` and `_spawn_outcomes`.

**Why does 2048 need chance nodes?** The next tile is random; a move that looks
good can lose if an unlucky tile lands in the wrong cell (section 5).

**What is a heuristic function?** An estimate of how good a board is, used when
the search stops before the game ends.

**Monotonicity? Smoothness? Empty cells? Corner?** Section 3. Short version:
ordered lines merge in chains; similar neighbours merge soon; empty cells are
room to move; a big tile in a corner does not block merges. Tuning set the
corner weight to 0 because monotonicity already does that job.

**What is search depth? Why does it cost so much?** How many of our moves the
agent looks ahead. Each level multiplies the work by about 4 × 2·(empty cells),
so cost grows exponentially (73 → 2,074 → 32,511 nodes).

**Is deeper always better?** Usually it plays better, but it costs much more
time. See the depth charts: the question is whether the gain is worth the time.

**Transposition table? Caching?** Caching = remembering a function's answer so
it is never computed twice. The transposition table caches searched boards
within one decision; `lru_cache` caches row moves and line features. Caching
does not change the result (tested against an uncached version).

**Computational complexity?** Time roughly O((8e)^d) nodes for depth d with e
empty cells; memory small (depth-first) plus the cache for one decision.

**How do you compare agents fairly?** Same 50 seeds for all, fresh agent per
game, only decision time measured, tuning on separate seeds, charts drawn only
from the CSV.

**Why does the GUI not freeze?** The AI runs in a background thread; it puts its
answer in a queue, and the main thread checks the queue every 30 ms and redraws.
Tkinter widgets are only touched by the main thread.

**Limitations?** Pure Python limits depth (depth 4 too slow for 50-game runs);
weights tuned on only 20 games with a small grid; 50 games still leave visible
variance (see the box plot); no learning (RL/neural networks are out of scope).
