# 2048 AI with Expectimax Search

Artificial Intelligence Lab project (AL-2002), FAST NUCES Karachi.

An AI that plays the 2048 puzzle. Its core is **Expectimax search**: MAX nodes
for our moves, CHANCE nodes for the random 2/4 tile, and a heuristic that scores
boards at the depth limit. It is compared with a **Random** and a **Greedy**
agent on the same 50 seeded games.

![The game window with Expectimax playing](screenshots/gui_autoplay.png)

## How to run

Python 3.9+. Everything except the charts uses only the standard library.

```bash
python main.py                      # open the game: play with arrow keys/WASD or let an AI play
python -m unittest tests -v         # run all tests
python experiments.py --quick       # 1-minute check that experiments work
python experiments.py               # full comparison on seeds 1-50 (about 1.5 hours on 8 cores)
python experiments.py --tune        # grid search for heuristic weights (seeds 51-70)
pip install -r requirements.txt     # matplotlib, needed only for charts
python charts.py                    # draw charts from results/final/per_game.csv
```

## Architecture

Five program files in layers; each file only uses the files below it.

```text
   main.py           experiments.py          charts.py
   (window)          (play many games,       (CSV -> pictures)
      |               write CSV)                  ^
      +--------+-----------+                      |
               v                       results/final/per_game.csv
          agents.py      which move is best?    Random, Greedy, Expectimax + heuristic
               |
               v
           game.py       what happens if I move?    all the rules of 2048
```

| File | Job | Main pieces |
|---|---|---|
| `game.py` | The rules of 2048 | `slide_board` (move without spawning), `possible_moves`, `can_move`, `GameState` (`play`, `spawn`, `legal_moves`, `is_terminal`) |
| `agents.py` | Choosing moves | `evaluate_board` (heuristic), `RandomAgent`, `GreedyAgent`, `ExpectimaxAgent` (`_max_value`, `_chance_value`, transposition table) |
| `main.py` | Tkinter window | Board, score, agent/depth/speed choice, AI Move, Auto Play, Restart. The AI runs in a background thread so the window never freezes. |
| `experiments.py` | Measuring | `--tune` grid search; the 50-seed comparison and ablation → `per_game.csv`, `summary.csv` |
| `charts.py` | Pictures | Six PNG charts in `results/charts/`, computed only from the CSV |
| `tests.py` | Checking | 22 tests: rules, agents, Expectimax maths, experiments, window |

**How one AI move flows:** `main.py` gives the current board to the chosen agent →
the agent explores future boards with `game.py`'s `slide_board` (no randomness,
no copying of the real game) and Expectimax averages over every possible new
tile → it returns the best direction → `main.py` calls `state.play(direction)`,
which moves and spawns a real random tile → the board is redrawn.

**Key design decisions**

* `move()` changes the board without spawning; `play()` = move + spawn. The AI
  can explore "what if" boards with no randomness, and Expectimax models the
  random tile explicitly at chance nodes.
* The AI searches with immutable board tuples, so nothing is copied and boards
  can be cache keys. Depth 2 went from 385 ms to about 17 ms per move.
* A board with no legal move scores −100000, so the agent avoids even a small
  chance of losing.
* Weights (empty 100, monotonicity 20, smoothness 11, corner 0) were tuned on
  seeds 51–70; the evaluation uses seeds 1–50.

**[VIVA_GUIDE.md](VIVA_GUIDE.md)** explains every part step by step, with a dry
run of Expectimax beating Greedy and answers to likely viva questions.

## Results

50 games per agent on seeds 1–50 (the same 50 starting boards for every agent).
Numbers come from `results/final/summary.csv`; charts are in `results/charts/`.

| Agent | Average score | Median score | Reached 2048 | Reached 4096 | Time per move | Nodes per move |
|---|---:|---:|---:|---:|---:|---:|
| Random | 1,098 | 1,024 | 0% | 0% | 0.04 ms | – |
| Greedy | 10,860 | 8,744 | 6% | 0% | 0.08 ms | – |
| Expectimax depth 1 | 16,539 | 14,390 | 24% | 0% | 0.9 ms | 42 |
| Expectimax depth 2 | 30,070 | 29,916 | 64% | 16% | 20 ms | 928 |
| Expectimax depth 3 | **40,656** | **36,262** | **86%** | **32%** | 267 ms | 13,978 |

![Highest tile reached](results/charts/highest_tile_distribution.png)

**What the results show**

1. **Any heuristic beats random play by about 10×.** Random never gets past a
   256 tile; Greedy's average score is 10× higher.
2. **Modelling the random tile is what matters.** Greedy and Expectimax depth 1
   both look exactly one move ahead. The only difference is that Expectimax
   averages over every possible new tile and avoids moves that can lose
   (see the dry run in VIVA_GUIDE.md). That alone raises the 2048 rate from 6% to 24%.
3. **Deeper search plays better:** 24% → 64% → 86% of games reach 2048 at
   depths 1 → 2 → 3, and a third of depth-3 games reach 4096.
4. **But the cost grows exponentially.** Each extra level expands about 15–22×
   more nodes and takes about 14–21× longer per move (0.9 → 20 → 267 ms). Depth 4
   would be roughly 4 s per move, over 2 hours per game in Python, which is why
   it was not included.
5. **Results vary a lot between games** (see the box plot): an unlucky tile
   early on can end a game. That is why 50 games per agent were used instead of a few.

![Score distribution](results/charts/score_boxplot.png)
![Decision time and nodes vs depth](results/charts/depth_cost.png)

**Ablation: is monotonicity useful?** Expectimax was rerun with the monotonicity
weight set to 0 (everything else unchanged):

| Depth | Average score with → without | Reached 2048 with → without |
|---|---|---|
| 1 | 16,539 → 6,363 | 24% → 0% |
| 2 | 30,070 → 13,300 | 64% → 8% |
| 3 | 40,656 → 18,948 (39 of 50 games so far) | 86% → 31% |

Without monotonicity the score roughly halves at every depth, and even depth 3
without it is worse than depth 2 with it. A good heuristic matters as much as
deeper search.

![Ablation](results/charts/ablation.png)

**Limitations.** Pure Python limits the search to depth 3. The weights were
tuned on 20 games with a small grid. There is no learning (reinforcement
learning and neural networks were out of scope).
