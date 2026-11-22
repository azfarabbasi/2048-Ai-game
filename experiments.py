"""Run reproducible comparisons between the 2048 AI agents.

The runner gives every agent the same list of starting seeds.  A result row is
written for every completed game and a second CSV contains aggregate metrics
that are useful in the project report and viva.
"""

import argparse
import csv
import multiprocessing
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from agents import DEFAULT_WEIGHTS, ExpectimaxAgent, GreedyAgent, RandomAgent
from game import GameState


PER_GAME_FIELDS = (
    "agent",
    "agent_label",
    "depth",
    "chance_mode",
    "max_chance_outcomes",
    "ablated_feature",
    "seed",
    "score",
    "highest_tile",
    "moves",
    "reached_2048",
    "decision_time_seconds",
    "mean_decision_ms",
    "total_nodes_expanded",
    "mean_nodes_expanded",
    "stop_reason",
)

SUMMARY_FIELDS = (
    "agent",
    "agent_label",
    "depth",
    "chance_mode",
    "max_chance_outcomes",
    "ablated_feature",
    "games",
    "wins",
    "win_rate",
    "average_score",
    "median_score",
    "average_highest_tile",
    "average_moves",
    "average_decision_ms",
    "average_total_nodes_expanded",
    "average_nodes_expanded_per_move",
)


@dataclass(frozen=True)
class AgentConfig:
    """The settings needed to create a fresh agent for one game."""

    name: str
    depth: Optional[int] = None
    max_chance_outcomes: Optional[int] = None
    ablated_feature: Optional[str] = None  # heuristic feature switched off

    @property
    def label(self):
        if self.name == "expectimax":
            label = f"Expectimax depth {self.depth}"
            if self.ablated_feature:
                label += f" (no {self.ablated_feature})"
            return label
        return self.name.title()

    @property
    def weights(self):
        """Heuristic weights, with the ablated feature's weight set to zero."""
        if self.ablated_feature is None:
            return None  # the agent then uses DEFAULT_WEIGHTS
        return {**DEFAULT_WEIGHTS, self.ablated_feature: 0.0}

    @property
    def chance_mode(self):
        if self.name != "expectimax":
            return "not_applicable"
        return "exact" if self.max_chance_outcomes is None else "stratified_approximation"

    def metadata(self):
        return {
            "agent": self.name,
            "agent_label": self.label,
            "depth": "" if self.depth is None else self.depth,
            "chance_mode": self.chance_mode,
            "max_chance_outcomes": "" if self.max_chance_outcomes is None else self.max_chance_outcomes,
            "ablated_feature": self.ablated_feature or "",
        }


def parse_seeds(values):
    """Parse individual seeds and inclusive ranges such as ``1-50``."""
    seeds = []
    for value in values:
        if "-" in value:
            start_text, end_text = value.split("-", 1)
            start, end = int(start_text), int(end_text)
            if start > end:
                raise ValueError(f"Seed range must increase: {value}")
            seeds.extend(range(start, end + 1))
        else:
            seeds.append(int(value))
    if not seeds:
        raise ValueError("Provide at least one seed")
    return list(dict.fromkeys(seeds))


def build_configs(agent_names, depths, max_chance_outcomes=None, ablate=()):
    """Build the comparison set requested on the command line.

    ``ablate`` lists heuristic features (such as ``monotonicity``). For each
    one, an extra Expectimax run is added with that feature's weight set to 0,
    which shows how much the feature helps.
    """
    names = tuple(agent_names)
    unknown = set(names) - {"random", "greedy", "expectimax"}
    if unknown:
        raise ValueError(f"Unknown agent name(s): {', '.join(sorted(unknown))}")
    if not names:
        raise ValueError("Choose at least one agent")
    unknown_features = set(ablate) - set(DEFAULT_WEIGHTS)
    if unknown_features:
        raise ValueError(f"Unknown heuristic feature(s): {', '.join(sorted(unknown_features))}")

    configs = []
    for name in names:
        if name == "expectimax":
            for depth in depths:
                configs.append(AgentConfig(name, depth, max_chance_outcomes))
                for feature in ablate:
                    configs.append(AgentConfig(name, depth, max_chance_outcomes, feature))
        else:
            configs.append(AgentConfig(name))
    return configs


def create_agent(config, seed):
    """Create a new agent so games do not share random or search state."""
    if config.name == "random":
        return RandomAgent(seed=seed)
    if config.name == "greedy":
        return GreedyAgent()
    if config.name == "expectimax":
        return ExpectimaxAgent(
            config.depth, weights=config.weights, max_chance_outcomes=config.max_chance_outcomes
        )
    raise ValueError(f"Unsupported agent: {config.name}")


def play_game(config, seed, max_moves=None):
    """Play one seeded game and return the metrics needed for a CSV row.

    Decision time measures only ``choose_move``.  It does not include the
    engine's tile spawning, so it fairly describes the cost of the AI decision.
    """
    if max_moves is not None and max_moves < 1:
        raise ValueError("max_moves must be at least 1 when supplied")

    state = GameState(seed=seed)
    agent = create_agent(config, seed)
    moves = 0
    decision_seconds = 0.0
    total_nodes = 0
    stop_reason = "terminal"

    while not state.is_terminal():
        if max_moves is not None and moves >= max_moves:
            stop_reason = "move_limit"
            break

        started = time.perf_counter()
        move = agent.choose_move(state)
        decision_seconds += time.perf_counter() - started
        total_nodes += getattr(agent, "nodes_expanded", 0)

        if move is None:
            stop_reason = "agent_returned_no_move"
            break
        if not state.play(move):
            stop_reason = "agent_returned_illegal_move"
            break
        moves += 1

    metadata = config.metadata()
    return {
        **metadata,
        "seed": seed,
        "score": state.score,
        "highest_tile": state.highest_tile(),
        "moves": moves,
        "reached_2048": int(state.reached_2048()),
        "decision_time_seconds": round(decision_seconds, 6),
        "mean_decision_ms": round(1000 * decision_seconds / moves, 6) if moves else 0.0,
        "total_nodes_expanded": total_nodes,
        "mean_nodes_expanded": round(total_nodes / moves, 6) if moves else 0.0,
        "stop_reason": stop_reason,
    }


def summarize(rows, configs):
    """Aggregate result rows in the same order as the requested configurations."""
    summaries = []
    for config in configs:
        metadata = config.metadata()
        group = [
            row for row in rows
            if all(row[key] == value for key, value in metadata.items())
        ]
        if not group:
            continue

        scores = [row["score"] for row in group]
        summaries.append({
            **metadata,
            "games": len(group),
            "wins": sum(row["reached_2048"] for row in group),
            "win_rate": round(sum(row["reached_2048"] for row in group) / len(group), 6),
            "average_score": round(statistics.fmean(scores), 6),
            "median_score": statistics.median(scores),
            "average_highest_tile": round(statistics.fmean(row["highest_tile"] for row in group), 6),
            "average_moves": round(statistics.fmean(row["moves"] for row in group), 6),
            "average_decision_ms": round(statistics.fmean(row["mean_decision_ms"] for row in group), 6),
            "average_total_nodes_expanded": round(
                statistics.fmean(row["total_nodes_expanded"] for row in group), 6
            ),
            "average_nodes_expanded_per_move": round(
                statistics.fmean(row["mean_nodes_expanded"] for row in group), 6
            ),
        })
    return summaries


def write_csv(path, fieldnames, rows):
    """Write result dictionaries as a UTF-8 CSV file."""
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _play_job(job):
    """Helper for the process pool: unpack one (config, seed, max_moves) job."""
    config, seed, max_moves = job
    return play_game(config, seed, max_moves=max_moves)


def run_experiment(seeds, configs, output_dir, max_moves=None, workers=1):
    """Run every configuration on every seed and write the two report files.

    Games are independent, so with ``workers > 1`` several games run at the
    same time on different CPU cores. Every game is still fully determined by
    its seed, so the results are identical; only the wall-clock time changes.
    """
    if not seeds:
        raise ValueError("Provide at least one seed")
    if not configs:
        raise ValueError("Provide at least one agent configuration")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    jobs = [(config, seed, max_moves) for config in configs for seed in seeds]
    if workers > 1:
        with multiprocessing.Pool(workers) as pool:
            rows = []
            for row in pool.imap(_play_job, jobs):  # imap keeps the job order
                rows.append(row)
                print(f"  finished {len(rows)}/{len(jobs)}: {row['agent_label']} seed {row['seed']}", flush=True)
    else:
        rows = [_play_job(job) for job in jobs]
    summaries = summarize(rows, configs)
    write_csv(output_dir / "per_game.csv", PER_GAME_FIELDS, rows)
    write_csv(output_dir / "summary.csv", SUMMARY_FIELDS, summaries)
    return rows, summaries


def build_parser():
    parser = argparse.ArgumentParser(
        description="Compare 2048 agents on identical seeded games.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        default=["1-3"],
        help="Individual seeds or inclusive ranges, for example: 1-50 (default: 1-3)",
    )
    parser.add_argument(
        "--agents",
        nargs="+",
        choices=("random", "greedy", "expectimax"),
        default=("random", "greedy", "expectimax"),
        help="Agents to compare (default: all three)",
    )
    parser.add_argument(
        "--depths",
        nargs="+",
        type=int,
        default=(2,),
        help="Expectimax depths to compare (default: 2)",
    )
    parser.add_argument(
        "--max-chance-outcomes",
        type=int,
        default=None,
        help="Optional deterministic chance-node limit; omit for exact branching.",
    )
    parser.add_argument(
        "--ablate",
        nargs="+",
        choices=tuple(DEFAULT_WEIGHTS),
        default=(),
        help="Also run Expectimax with these heuristic features switched off.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of games to run in parallel (default: 1).",
    )
    parser.add_argument(
        "--max-moves",
        type=int,
        default=None,
        help="Optional safety cap per game; omit to play until terminal.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results"),
        help="Folder for per_game.csv and summary.csv (default: results)",
    )
    return parser


def main():
    args = build_parser().parse_args()
    seeds = parse_seeds(args.seeds)
    configs = build_configs(args.agents, args.depths, args.max_chance_outcomes, args.ablate)
    rows, summaries = run_experiment(
        seeds, configs, args.output_dir, args.max_moves, workers=args.workers
    )

    print(f"Completed {len(rows)} games. Results saved to {args.output_dir}.")
    for summary in summaries:
        print(
            f"{summary['agent_label']}: "
            f"average score {summary['average_score']:.1f}, "
            f"win rate {summary['win_rate']:.1%}, "
            f"average decision {summary['average_decision_ms']:.3f} ms"
        )


if __name__ == "__main__":
    main()
