"""Measure the agents: tune the heuristic weights, then compare all agents.

    python experiments.py            final comparison on seeds 1-50 -> results/final/
    python experiments.py --quick    3 short games per agent, to check that it works
    python experiments.py --tune     grid search for heuristic weights on seeds 51-70

Fairness rules:
* Every agent plays the same seeds, so every agent gets the same starting board.
* Weights are tuned on seeds 51-70 and tested on seeds 1-50, never the same games.
* A new agent is created for every game, so nothing carries over between games.
* Only ``choose_move`` is timed (not drawing or tile spawning).
"""

import argparse
import csv
import itertools
import multiprocessing
import statistics
import time
from pathlib import Path

from agents import DEFAULT_WEIGHTS, ExpectimaxAgent, GreedyAgent, RandomAgent
from game import GameState


EVALUATION_SEEDS = list(range(1, 51))
TUNING_SEEDS = list(range(51, 71))

# Candidate values for the grid search (3 x 3 x 3 x 3 = 81 combinations).
WEIGHT_GRID = {
    "empty": (100.0, 270.0, 500.0),
    "monotonicity": (20.0, 47.0, 100.0),
    "smoothness": (0.0, 11.0, 30.0),
    "corner": (0.0, 10.0, 100.0),
}


def agent_config(agent, depth=None, ablated_feature=None, weights=None):
    """Describe one agent setting. ``ablated_feature`` sets that weight to 0."""
    label = agent.title() if agent != "expectimax" else f"Expectimax depth {depth}"
    if ablated_feature:
        label += f" (no {ablated_feature})"
        weights = {**DEFAULT_WEIGHTS, ablated_feature: 0.0}
    return {"label": label, "agent": agent, "depth": depth,
            "ablated_feature": ablated_feature, "weights": weights}


def make_agent(config, seed):
    if config["agent"] == "random":
        return RandomAgent(seed=seed)
    if config["agent"] == "greedy":
        return GreedyAgent(config["weights"])
    return ExpectimaxAgent(config["depth"], config["weights"])


def play_game(job):
    """Play one seeded game until no move is left; return one CSV row."""
    config, seed, max_moves = job
    state = GameState(seed=seed)
    agent = make_agent(config, seed)
    moves, thinking_seconds, nodes = 0, 0.0, 0

    while not state.is_terminal() and (max_moves is None or moves < max_moves):
        started = time.perf_counter()
        move = agent.choose_move(state)
        thinking_seconds += time.perf_counter() - started
        nodes += getattr(agent, "nodes_expanded", 0)  # Random/Greedy have no search tree
        state.play(move)
        moves += 1

    return {
        "agent_label": config["label"],
        "agent": config["agent"],
        "depth": config["depth"] or "",
        "ablated_feature": config["ablated_feature"] or "",
        "seed": seed,
        "score": state.score,
        "highest_tile": state.highest_tile(),
        "moves": moves,
        "reached_2048": int(state.reached_2048()),
        "mean_decision_ms": round(1000 * thinking_seconds / max(moves, 1), 4),
        "mean_nodes_expanded": round(nodes / max(moves, 1), 1),
    }


def run_games(configs, seeds, workers, max_moves=None, save_to=None):
    """Play every config on every seed, several games at once on different CPU cores.

    Each game depends only on its seed, so running in parallel gives exactly
    the same results as running one by one; it is just faster.

    With ``save_to``, each game is added to that CSV the moment it finishes and
    games already in the file are skipped. A crash then loses nothing: run the
    same command again and it continues where it stopped.
    """
    done = {}
    if save_to is not None and save_to.exists():
        done = {(row["agent_label"], row["seed"]): row for row in read_csv(save_to)}
    jobs = [(config, seed, max_moves) for config in configs for seed in seeds
            if (config["label"], seed) not in done]
    finished = dict(done)
    with multiprocessing.Pool(workers) as pool:
        for row in pool.imap_unordered(play_game, jobs):  # results arrive as games finish
            finished[(row["agent_label"], row["seed"])] = row
            if save_to is not None:
                append_csv(save_to, row)
            print(f"  {len(finished) - len(done)}/{len(jobs)}  {row['agent_label']}, seed {row['seed']}: "
                  f"score {row['score']}, highest tile {row['highest_tile']}", flush=True)
    # Return the rows in a fixed order: config by config, seed by seed.
    return [finished[(config["label"], seed)] for config in configs for seed in seeds]


def summarize(rows):
    """One summary row per agent label: averages over its games."""
    labels = list(dict.fromkeys(row["agent_label"] for row in rows))
    summary = []
    for label in labels:
        games = [row for row in rows if row["agent_label"] == label]
        summary.append({
            "agent_label": label,
            "games": len(games),
            "reached_2048": sum(g["reached_2048"] for g in games),
            "win_rate_percent": round(100 * statistics.fmean(g["reached_2048"] for g in games), 1),
            "average_score": round(statistics.fmean(g["score"] for g in games)),
            "median_score": statistics.median(g["score"] for g in games),
            "average_highest_tile": round(statistics.fmean(g["highest_tile"] for g in games)),
            "average_moves": round(statistics.fmean(g["moves"] for g in games)),
            "average_decision_ms": round(statistics.fmean(g["mean_decision_ms"] for g in games), 3),
            "average_nodes_per_move": round(statistics.fmean(g["mean_nodes_expanded"] for g in games)),
        })
    return summary


NUMBER_COLUMNS = {"seed": int, "score": int, "highest_tile": int, "moves": int,
                  "reached_2048": int, "mean_decision_ms": float, "mean_nodes_expanded": float}


def read_csv(path):
    """Read per-game rows back from a CSV, turning number columns into numbers."""
    with path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    for row in rows:
        for column, convert in NUMBER_COLUMNS.items():
            row[column] = convert(row[column])
    return rows


def append_csv(path, row):
    """Add one row to a CSV, writing the header first if the file is new."""
    new_file = not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(row))
        if new_file:
            writer.writeheader()
        writer.writerow(row)


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("saved", path)


def compare_agents(depths, seeds, output_dir, workers, max_moves=None):
    """Random, Greedy, Expectimax at each depth, plus the no-monotonicity ablation."""
    configs = [agent_config("random"), agent_config("greedy")]
    configs += [agent_config("expectimax", depth) for depth in depths]
    configs += [agent_config("expectimax", depth, "monotonicity") for depth in depths]
    rows = run_games(configs, seeds, workers, max_moves, save_to=output_dir / "per_game.csv")
    summary = summarize(rows)
    write_csv(output_dir / "per_game.csv", rows)  # rewrite in a tidy, fixed order
    write_csv(output_dir / "summary.csv", summary)
    for line in summary:
        print(f"{line['agent_label']:<40} average score {line['average_score']:>7}, "
              f"2048 in {line['win_rate_percent']}% of games, {line['average_decision_ms']} ms/move")


def tune_weights(workers):
    """Two-stage grid search on the tuning seeds.

    Stage 1: all 81 weight combinations with cheap depth-1 search.
    Stage 2: the best 5 (plus the old default) again at depth 2; best average score wins.
    """
    def score_candidates(candidates, depth):
        configs = [agent_config("expectimax", depth, weights=dict(zip(WEIGHT_GRID, c)))
                   for c in candidates]
        rows = run_games(configs, TUNING_SEEDS, workers)
        n = len(TUNING_SEEDS)
        table = []
        for index, candidate in enumerate(candidates):
            games = rows[index * n:(index + 1) * n]  # rows come back in job order
            table.append({**dict(zip(WEIGHT_GRID, candidate)), "depth": depth,
                          "average_score": round(statistics.fmean(g["score"] for g in games)),
                          "reached_2048": sum(g["reached_2048"] for g in games)})
        table.sort(key=lambda row: row["average_score"], reverse=True)
        write_csv(Path(f"results/tuning/tuning_depth{depth}.csv"), table)
        return table

    stage1 = score_candidates(list(itertools.product(*WEIGHT_GRID.values())), depth=1)
    best = [tuple(row[name] for name in WEIGHT_GRID) for row in stage1[:5]]
    old_default = (270.0, 47.0, 11.0, 10.0)
    if old_default not in best:
        best.append(old_default)
    stage2 = score_candidates(best, depth=2)
    print("Best weights:", {name: stage2[0][name] for name in WEIGHT_GRID})


def main():
    parser = argparse.ArgumentParser(description="Tune and compare the 2048 agents.")
    parser.add_argument("--tune", action="store_true", help="grid-search the heuristic weights")
    parser.add_argument("--quick", action="store_true", help="a fast check: 3 seeds, 100 moves")
    parser.add_argument("--depths", type=int, nargs="+", default=[1, 2, 3])
    # Leave some CPU cores and memory free for the rest of the computer.
    parser.add_argument("--workers", type=int, default=max(1, multiprocessing.cpu_count() // 2))
    parser.add_argument("--output", type=Path, default=Path("results/final"))
    args = parser.parse_args()

    if args.tune:
        tune_weights(args.workers)
    elif args.quick:
        compare_agents([1, 2], [1, 2, 3], Path("results/quick"), args.workers, max_moves=100)
    else:
        compare_agents(args.depths, EVALUATION_SEEDS, args.output, args.workers)


if __name__ == "__main__":
    main()
