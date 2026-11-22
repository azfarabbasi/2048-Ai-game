# 2048 AI with Expectimax Search

Artificial Intelligence Lab project (AL-2002), FAST NUCES Karachi.

An AI that plays the 2048 puzzle. Its core is **Expectimax search**: MAX nodes
for our moves, CHANCE nodes for the random 2/4 tile, and a heuristic that scores
boards at the depth limit. It is compared with a **Random** and a **Greedy**
baseline on the same 50 seeded games.

![GUI](screenshots/gui_autoplay.png)

## Requirements

Python 3.9 or newer. The game, agents, GUI and experiments use only the
standard library (Tkinter is included with Python on Windows). Charts need
matplotlib:

```bash
pip install -r requirements.txt
```

## Run

| Command | What it does |
|---|---|
| `python gui.py` | Window: play with arrow keys/WASD, or let an agent play (choose agent, depth, speed) |
| `python main.py` | Play in the terminal |
| `python demo.py` | Every agent decides on one example board; shows move values and node counts |
| `python -m unittest discover -s tests -v` | Run all tests |
| `python experiments.py --seeds 1-50 --depths 1 2 3 --ablate monotonicity --workers 7 --output-dir results/final` | The full experiment (about 1 hour on 8 cores) |
| `python charts.py` | Draw the report charts from `results/final/per_game.csv` |
| `python tune.py` | Grid-search the heuristic weights on tuning seeds 51-70 |

A quick check that experiments work: `python experiments.py --seeds 1-3 --depths 1 --output-dir results/smoke`.

## Project structure

| File | Purpose |
|---|---|
| `game.py` | All 2048 rules: sliding/merging, scoring, seeded tile spawning, legal moves, game over |
| `agents.py` | `RandomAgent`, `GreedyAgent`, `ExpectimaxAgent`, and the heuristic `evaluate_board` |
| `gui.py` | Tkinter interface; the AI runs in a background thread so the window never freezes |
| `experiments.py` | Plays seeded games for each agent; writes `per_game.csv` and `summary.csv` |
| `charts.py` | Turns the CSV into PNG charts in `results/charts/` |
| `tune.py` | Grid search for heuristic weights (separate seeds from the evaluation) |
| `main.py`, `demo.py` | Terminal game and a small agent demonstration |
| `tests/` | Engine, agent, experiment and GUI tests |
| `docs/algorithm.md` | **How it works**: architecture, heuristic, Expectimax with a dry run, optimisations |
| `docs/results.md` | **Measured results** and what they mean |
| `docs/viva_guide.md` | Answers to likely viva questions and a demo plan |

## Key design points

* `move()` changes the board without spawning; `play()` = move + spawn. This
  lets the AI explore "what if" boards with no randomness, while Expectimax
  models the spawn explicitly at chance nodes.
* Depth counts our moves; each move is followed by one chance node.
* Losing boards get a very large negative value, so the agent avoids moves with
  even a small chance of ending the game.
* Speed: immutable board tuples, cached row moves, cached heuristic features,
  and a transposition table. Depth 2 went from 385 ms to about 17 ms per move.
* Heuristic weights (empty 100, monotonicity 20, smoothness 11, corner 0) were
  chosen by `tune.py` on seeds 51-70; evaluation uses seeds 1-50.

See [docs/results.md](docs/results.md) for the experiment results.
