"""Tune the heuristic weights with a simple grid search.

Every combination of the candidate weights below is played on the tuning
seeds, and combinations are ranked by average score. Tuning uses seeds 51-70,
never the evaluation seeds 1-50, so the final comparison stays fair: the
weights were not chosen by looking at the games they are tested on.

Two stages keep it fast:
1. Try every combination with cheap depth-1 Expectimax.
2. Re-test only the best few combinations at depth 2 (``--top-from``).
"""

import argparse
import csv
import itertools
import multiprocessing
import statistics
from pathlib import Path

from agents import DEFAULT_WEIGHTS, ExpectimaxAgent
from experiments import parse_seeds
from game import GameState


GRID = {
    "empty": (100.0, 270.0, 500.0),
    "monotonicity": (20.0, 47.0, 100.0),
    "smoothness": (0.0, 11.0, 30.0),
    "corner": (0.0, 10.0, 100.0),
}
FEATURES = tuple(GRID)


def play(job):
    """Play one game to the end; return (weights, seed, score, highest tile)."""
    weights, depth, seed = job
    state = GameState(seed=seed)
    agent = ExpectimaxAgent(depth, weights=dict(zip(FEATURES, weights)))
    while True:
        move = agent.choose_move(state)
        if move is None:
            break
        state.play(move)
    return weights, seed, state.score, state.highest_tile()


def load_top(path, count):
    """Read the best ``count`` weight combinations from an earlier tuning CSV."""
    with Path(path).open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    return [tuple(float(row[name]) for name in FEATURES) for row in rows[:count]]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--depth", type=int, default=1)
    parser.add_argument("--seeds", nargs="+", default=["51-70"])
    parser.add_argument("--workers", type=int, default=multiprocessing.cpu_count())
    parser.add_argument("--top-from", help="CSV from an earlier run; re-test only its best rows")
    parser.add_argument("--top", type=int, default=5)
    parser.add_argument("--output", type=Path, default=Path("results/tuning/tuning_depth1.csv"))
    args = parser.parse_args()

    seeds = parse_seeds(args.seeds)
    if args.top_from:
        candidates = load_top(args.top_from, args.top)
        default = tuple(DEFAULT_WEIGHTS[name] for name in FEATURES)
        if default not in candidates:
            candidates.append(default)  # always compare against the current default
    else:
        candidates = list(itertools.product(*GRID.values()))

    jobs = [(weights, args.depth, seed) for weights in candidates for seed in seeds]
    print(f"Playing {len(jobs)} games ({len(candidates)} weight sets x {len(seeds)} seeds)...")
    with multiprocessing.Pool(args.workers) as pool:
        results = pool.map(play, jobs)

    table = []
    for weights in candidates:
        games = [r for r in results if r[0] == weights]
        table.append({
            **dict(zip(FEATURES, weights)),
            "depth": args.depth,
            "games": len(games),
            "average_score": round(statistics.fmean(r[2] for r in games), 1),
            "reached_2048": sum(r[3] >= 2048 for r in games),
            "average_highest_tile": round(statistics.fmean(r[3] for r in games), 1),
        })
    table.sort(key=lambda row: row["average_score"], reverse=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)

    print(f"Saved {args.output}. Best weight sets:")
    for row in table[:5]:
        print(row)


if __name__ == "__main__":
    main()
