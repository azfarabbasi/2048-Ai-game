# Viva preparation

Short answers first; say them in your own words. The file and function to
point at is given for each, so you can open the code during the viva.

## Demo plan (5 minutes)

1. `python gui.py` → choose Expectimax, depth 2, press **Auto Play**. Point at
   the line under the score: it shows the expected value of each move and the
   number of nodes searched.
2. Stop, switch to **Greedy**, press **AI Move** a few times: no node count,
   decisions are instant but short-sighted.
3. `python demo.py` → node counts 73 → 1,928 → 29,679 for depths 1, 2, 3:
   exponential growth.
4. Open `results/charts/` and `docs/results.md`: the measured comparison.
5. `python -m unittest discover -s tests -v` → all tests pass.

## Basics

**What is 2048?** A 4×4 sliding puzzle. Equal tiles merge into their sum;
after every move a 2 (90%) or 4 (10%) appears in a random empty cell. The goal
is a 2048 tile; the game ends when no move changes the board.

**What is an AI agent?** Something that perceives its environment and acts to
achieve a goal. PEAS for us: *Performance* = score, highest tile, win rate;
*Environment* = the board + random spawns; *Actuators* = up/down/left/right;
*Sensors* = the full board and score.

**Environment type?** Fully observable, stochastic, static, discrete,
single-agent. Our agent is *utility-based*: it picks the move with the highest
expected utility.

**What is a search algorithm?** A systematic way to explore possible future
states (a tree of "if I do this, then that happens") to choose an action.

## Expectimax

**What is Expectimax?** A game-tree search with two node types. MAX nodes are
our choices and take the best child. CHANCE nodes are random events and take
the probability-weighted average of their children. (`ExpectimaxAgent` in
`agents.py`.)

**Minimax vs Expectimax?** Minimax has MIN nodes: an opponent who picks the
worst outcome for us. 2048 has no opponent, only random tiles with known
probabilities, so we average instead of taking the minimum. Minimax would be
too pessimistic and play badly.

**What is a MAX node?** Our turn. Try each legal move, return the highest
value. (`_max_value`)

**What is a chance node?** The game's turn. For every empty cell and both tile
values, value × probability, summed. Probability = 1/(empty cells) × 0.9 or 0.1.
(`_chance_value`, `_spawn_outcomes`)

**Why does 2048 need chance nodes?** Because the next tile is random. A move can
look great but lose if an unlucky tile lands in the wrong cell. Show the dry run
in `docs/algorithm.md`: Greedy picks "right", which loses 90% of the time;
Expectimax sees that and picks "left".

**What is search depth?** How many of our moves the agent looks ahead. Depth 2
= our move, random tile, our move, random tile, then evaluate.

**Why does increasing depth increase computation?** Each extra level multiplies
the tree by about 4 moves × 2·(empty cells) spawns, so the work grows
exponentially: about 15–25× per level in our measurements.

**What is computational complexity here?** Roughly O((4 · 2e)^d) nodes for depth
d with e empty cells. Memory is small because we search depth-first; the cache
adds memory proportional to the number of distinct boards visited in one decision.

**Is deeper always better?** No. It usually plays better but costs much more
time; see the depth charts. At some point the extra time is not worth it.

## Heuristic

**What is a heuristic function?** An estimate of how good a board is, used
when the search stops before the game ends. (`evaluate_board`)

**Monotonicity?** Tiles in each row/column should be in order (increasing or
decreasing). Ordered lines can merge in a chain, like 2-4-8-16. We add up the
uphill steps and downhill steps of a line; the smaller one is the penalty.

**Smoothness?** Neighbouring tiles should be similar (in log2 terms) because
similar tiles merge soon. Big jumps like 2 next to 512 are penalised.

**Why are empty cells important?** They are room to move. Fewer empty cells
means closer to game over, and new tiles have fewer places to go.

**Why should the largest tile stay in a corner?** A big tile in the middle
blocks merges in two directions. Interesting finding: our weight tuning gave the
corner bonus weight 0, because monotonicity already pushes the largest tile to
a corner (an ordered row and column both end in a corner).

**Why log2 of tiles?** So that 1024 vs 2048 counts as one step, the same as 2
vs 4. Otherwise big tiles would dominate every feature.

**How were the weights chosen?** Grid search (`tune.py`): 81 weight
combinations at depth 1 on seeds 51–70, then the best 5 plus the old defaults
re-tested at depth 2. The depth-1 winner was *not* the depth-2 winner, so we
picked by the depth-2 test. We never tuned on seeds 1–50, which are used for
the final evaluation.

## Optimisation

**What is a transposition table?** A dictionary that stores the value of a
board already searched during the current decision, keyed by (node type,
board, remaining depth). Different move orders can reach the same board (left
then up = up then left), so the second time it is a lookup.

**What is caching / memoization?** Remembering the answer of a function for an
input so it is never computed twice. We use it for the transposition table, for
sliding a row (`@lru_cache` on `_slide_row_left`) and for per-line heuristic
features (`_line_features`).

**Does caching change the result?** No. `test_caching_does_not_change_values`
compares our agent against a plain recursive Expectimax with no cache and the
values are equal.

**What made the code faster?** Profiling showed most of the time went into
copying game objects. The search now uses immutable tuples, cached row moves,
cached heuristic features, a quick `can_move()` check, and the transposition
table: depth 2 went from 385 ms to ~17 ms per move.

## Experiments

**How do you compare agents fairly?** Same 50 seeds (1–50) for every agent, so
every agent starts from the same boards; a new agent object for each game, so
nothing carries over; only `choose_move` is timed; weights were tuned on
different seeds (51–70). Results are written to CSV and charts are drawn only
from that CSV.

**Is a game fully reproducible?** Yes for the board and moves: the seed fixes
every random tile given the same moves. Only the timing columns change between
runs.

**What is the ablation study?** We switch one heuristic feature off
(monotonicity weight = 0) and rerun. The drop in score shows how much that
feature contributes.

## Limitations you should mention yourself

* Pure Python is slow, so depth 4 is too expensive for full experiments.
* The heuristic weights are tuned on 20 games with a small grid; more tuning
  could help.
* 50 games per agent gives averages with noticeable variance; the box plot
  shows the spread.
* No learning (RL/neural networks) by design: out of scope in the proposal.
