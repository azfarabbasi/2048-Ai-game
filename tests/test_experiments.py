import csv
import tempfile
import unittest
from pathlib import Path

from experiments import AgentConfig, build_configs, parse_seeds, play_game, run_experiment


class ExperimentTests(unittest.TestCase):
    def test_parse_seeds_accepts_numbers_and_ranges(self):
        self.assertEqual(parse_seeds(["1-3", "7", "3"]), [1, 2, 3, 7])
        with self.assertRaises(ValueError):
            parse_seeds(["4-1"])

    def test_random_games_are_reproducible_except_for_timing(self):
        config = AgentConfig("random")
        first = play_game(config, seed=11, max_moves=20)
        second = play_game(config, seed=11, max_moves=20)
        for key in first:
            if "time" not in key and "ms" not in key:
                self.assertEqual(first[key], second[key])

    def test_run_experiment_writes_per_game_and_summary_csv(self):
        configs = build_configs(("random", "greedy"), depths=(2,))
        with tempfile.TemporaryDirectory() as temporary_directory:
            rows, summaries = run_experiment(
                seeds=[1, 2],
                configs=configs,
                output_dir=temporary_directory,
                max_moves=3,
            )
            self.assertEqual(len(rows), 4)
            self.assertEqual(len(summaries), 2)

            with (Path(temporary_directory) / "per_game.csv").open(newline="", encoding="utf-8") as file:
                csv_rows = list(csv.DictReader(file))
            with (Path(temporary_directory) / "summary.csv").open(newline="", encoding="utf-8") as file:
                summary_rows = list(csv.DictReader(file))

            self.assertEqual(len(csv_rows), 4)
            self.assertEqual(len(summary_rows), 2)
            self.assertEqual({row["stop_reason"] for row in csv_rows}, {"move_limit"})

    def test_expectimax_records_expanded_nodes(self):
        config = AgentConfig("expectimax", depth=1, max_chance_outcomes=1)
        row = play_game(config, seed=1, max_moves=1)
        self.assertEqual(row["chance_mode"], "stratified_approximation")
        self.assertGreater(row["total_nodes_expanded"], 0)
        self.assertEqual(row["stop_reason"], "move_limit")

    def test_ablation_adds_config_with_zero_weight(self):
        configs = build_configs(("expectimax",), depths=(2,), ablate=("monotonicity",))
        self.assertEqual([c.label for c in configs],
                         ["Expectimax depth 2", "Expectimax depth 2 (no monotonicity)"])
        self.assertEqual(configs[1].weights["monotonicity"], 0.0)
        with self.assertRaises(ValueError):
            build_configs(("expectimax",), depths=(2,), ablate=("colour",))


if __name__ == "__main__":
    unittest.main()
