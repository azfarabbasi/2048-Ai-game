"""All tests for the project.   Run:  python -m unittest tests -v"""

import csv
import math
import random
import tempfile
import time
import unittest
from pathlib import Path

from agents import (
    DEFAULT_WEIGHTS, LOSS_VALUE, ExpectimaxAgent, GreedyAgent, RandomAgent,
    _line_features, evaluate_board, place_tile,
)
from experiments import agent_config, play_game, read_csv, run_games, summarize, write_csv
from game import SIZE, GameState, can_move, merge_line, possible_moves, slide_board

LOCKED = ((2, 4, 2, 4), (4, 2, 4, 2), (2, 4, 2, 4), (4, 2, 4, 2))  # full, no merges


def plain_expectimax(board, depth, is_max):
    """Textbook Expectimax with no cache, used to check the real agent."""
    if is_max:
        if not can_move(board):
            return LOSS_VALUE
        if depth == 0:
            return evaluate_board(board)
        return max(plain_expectimax(child, depth, False) for _, child, _ in possible_moves(board))
    empty = [(r, c) for r in range(SIZE) for c in range(SIZE) if board[r][c] == 0]
    return sum(
        p / len(empty) * plain_expectimax(place_tile(board, r, c, tile), depth - 1, True)
        for r, c in empty for tile, p in ((2, 0.9), (4, 0.1))
    )


class GameTests(unittest.TestCase):
    def test_merge_once(self):
        self.assertEqual(merge_line([2, 2, 2, 2]), ([4, 4, 0, 0], 8))
        self.assertEqual(merge_line([0, 2, 0, 2]), ([4, 0, 0, 0], 4))
        # The new 8 must not merge again with the existing 8 in the same move.
        self.assertEqual(merge_line([4, 4, 8, 0]), ([8, 8, 0, 0], 8))

    def test_all_four_directions(self):
        board = ((2, 0, 0, 2), (0, 0, 0, 0), (0, 0, 0, 0), (2, 0, 0, 0))
        self.assertEqual(slide_board(board, "left")[0][0], (4, 0, 0, 0))
        self.assertEqual(slide_board(board, "right")[0][0], (0, 0, 0, 4))
        self.assertEqual([row[0] for row in slide_board(board, "up")[0]], [4, 0, 0, 0])
        self.assertEqual([row[0] for row in slide_board(board, "down")[0]], [0, 0, 0, 4])

    def test_move_updates_score(self):
        state = GameState([[2, 0, 2, 4], [0] * 4, [0] * 4, [0] * 4])
        self.assertTrue(state.move("right"))
        self.assertEqual(state.grid[0], [0, 0, 4, 4])
        self.assertEqual(state.score, 4)

    def test_illegal_move_does_not_spawn(self):
        state = GameState([[2, 0, 0, 0], [0] * 4, [0] * 4, [0] * 4], seed=10)
        self.assertFalse(state.play("left"))
        self.assertEqual(len(state.empty_cells()), 15)
        self.assertTrue(state.play("right"))
        self.assertEqual(len(state.empty_cells()), 14)

    def test_legal_moves_and_game_over(self):
        state = GameState([[2, 2, 8, 16], [32, 64, 128, 256], [2, 4, 8, 16], [32, 64, 128, 256]])
        self.assertEqual(state.legal_moves(), ["left", "right"])
        self.assertTrue(GameState([list(row) for row in LOCKED]).is_terminal())
        self.assertFalse(can_move(LOCKED))
        self.assertTrue(can_move(((0, 2, 4, 8),) + LOCKED[1:]))

    def test_spawn_probabilities_are_about_90_10(self):
        rng = random.Random(0)
        fours = 0
        for _ in range(5000):
            state = GameState(start_tiles=False, seed=rng.random())
            state.spawn()
            fours += 4 in [n for row in state.grid for n in row]
        self.assertAlmostEqual(fours / 5000, 0.1, delta=0.02)

    def test_seed_is_reproducible_and_copy_is_independent(self):
        self.assertEqual(GameState(seed=42).grid, GameState(seed=42).grid)
        state = GameState(seed=42)
        clone = state.copy()
        clone.grid[0][0] = 128
        self.assertNotEqual(state.grid, clone.grid)


class AgentTests(unittest.TestCase):
    def setUp(self):
        self.state = GameState([[2, 2, 4, 0], [4, 8, 0, 0], [0] * 4, [0] * 4], start_tiles=False)

    def test_every_agent_returns_a_legal_move(self):
        for agent in (RandomAgent(seed=3), GreedyAgent(), ExpectimaxAgent(depth=2)):
            self.assertIn(agent.choose_move(self.state), self.state.legal_moves())
        self.assertIsNone(RandomAgent().choose_move(GameState([list(r) for r in LOCKED])))

    def test_agents_do_not_change_the_real_game(self):
        before = self.state.to_tuple()
        ExpectimaxAgent(depth=2).choose_move(self.state)
        self.assertEqual(self.state.to_tuple(), before)

    def test_heuristic_features(self):
        self.assertEqual(_line_features((2, 4, 8, 16))[1], 0)   # ordered: no penalty
        self.assertEqual(_line_features((16, 8, 4, 2))[1], 0)
        self.assertLess(_line_features((2, 16, 4, 8))[1], 0)
        empty = ((2, 0, 0, 0),) + ((0,) * 4,) * 3
        crowded = ((2, 4, 8, 16), (32, 64, 128, 256), (512, 1024, 2, 4), (8, 16, 32, 64))
        self.assertGreater(evaluate_board(empty), evaluate_board(crowded))

    def test_chance_node_is_weighted_average(self):
        # Exactly one empty cell: value = 0.9 * (spawn 2) + 0.1 * (spawn 4).
        board = ((2, 2, 8, 16), (4, 8, 16, 32), (8, 16, 32, 64), (16, 32, 64, 0))
        with_two = evaluate_board(place_tile(board, 3, 3, 2))
        with_four = evaluate_board(place_tile(board, 3, 3, 4))
        value = ExpectimaxAgent(depth=1)._chance_value(board, 1)
        self.assertAlmostEqual(value, 0.9 * with_two + 0.1 * with_four)

    def test_spawn_probabilities_sum_to_one(self):
        board = ((2, 0, 0, 0),) + ((0,) * 4,) * 3
        outcomes = ExpectimaxAgent(depth=1)._spawn_outcomes(board)
        self.assertEqual(len(outcomes), 30)  # 15 empty cells x (2 or 4)
        self.assertAlmostEqual(sum(p for _, p in outcomes), 1.0)

    def test_lost_board_gets_loss_value(self):
        self.assertEqual(ExpectimaxAgent(depth=2)._max_value(LOCKED, 2), LOSS_VALUE)

    def test_expectimax_avoids_losing_move_that_greedy_takes(self):
        # The dry run from VIVA_GUIDE.md: "right" loses 90% of the time.
        state = GameState([[2, 2, 8, 16], [4, 8, 16, 32], [8, 16, 32, 64], [16, 32, 64, 128]])
        self.assertEqual(GreedyAgent().choose_move(state), "right")
        self.assertEqual(ExpectimaxAgent(depth=1).choose_move(state), "left")

    def test_caching_does_not_change_values(self):
        agent = ExpectimaxAgent(depth=2)
        agent.choose_move(self.state)
        for direction, child, _ in possible_moves(self.state.to_tuple()):
            self.assertTrue(math.isclose(
                agent.move_values[direction], plain_expectimax(child, 2, False), rel_tol=1e-9))

    def test_deeper_search_expands_more_nodes(self):
        counts = []
        for depth in (1, 2):
            agent = ExpectimaxAgent(depth=depth)
            agent.choose_move(self.state)
            counts.append(agent.nodes_expanded)
        self.assertGreater(counts[0], 0)
        self.assertLess(counts[0], counts[1])

    def test_invalid_depth_is_rejected(self):
        with self.assertRaises(ValueError):
            ExpectimaxAgent(depth=0)


class ExperimentTests(unittest.TestCase):
    def test_games_are_reproducible_except_timing(self):
        job = (agent_config("expectimax", 1), 11, 30)
        first, second = play_game(job), play_game(job)
        for key in first:
            if key != "mean_decision_ms":
                self.assertEqual(first[key], second[key])
        self.assertEqual(first["moves"], 30)

    def test_ablation_sets_weight_to_zero(self):
        config = agent_config("expectimax", 2, "monotonicity")
        self.assertEqual(config["label"], "Expectimax depth 2 (no monotonicity)")
        self.assertEqual(config["weights"]["monotonicity"], 0.0)
        self.assertEqual(config["weights"]["empty"], DEFAULT_WEIGHTS["empty"])

    def test_summary_and_csv(self):
        rows = [play_game((agent_config(name), seed, 20)) for name in ("random", "greedy")
                for seed in (1, 2)]
        summary = summarize(rows)
        self.assertEqual([s["agent_label"] for s in summary], ["Random", "Greedy"])
        self.assertEqual(summary[0]["games"], 2)
        with tempfile.TemporaryDirectory() as folder:
            write_csv(Path(folder) / "per_game.csv", rows)
            with (Path(folder) / "per_game.csv").open(newline="", encoding="utf-8") as file:
                self.assertEqual(len(list(csv.DictReader(file))), 4)


    def test_saved_games_are_skipped_when_resuming(self):
        configs = [agent_config("random")]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "per_game.csv"
            first = run_games(configs, [1, 2], workers=1, max_moves=10, save_to=path)
            self.assertEqual(len(read_csv(path)), 2)
            again = run_games(configs, [1, 2, 3], workers=1, max_moves=10, save_to=path)
            self.assertEqual(len(read_csv(path)), 3)  # only seed 3 was played
            self.assertEqual(again[:2], first)


class GuiTests(unittest.TestCase):
    def setUp(self):
        try:
            import tkinter as tk
            from main import Game2048App
            self.root = tk.Tk()
        except Exception:  # no Tkinter or no screen (for example on a CI server)
            self.skipTest("Tkinter window not available")
        self.root.withdraw()
        self.app = Game2048App(self.root)

    def tearDown(self):
        self.root.destroy()

    def test_ai_move_updates_board_and_score(self):
        self.app.agent_var.set("Expectimax")
        self.app.depth_var.set("1")
        before = self.app.state.to_tuple()
        self.app.ai_move()
        end = time.time() + 10
        while self.app.thinking and time.time() < end:
            self.root.update()
            time.sleep(0.01)
        self.assertNotEqual(self.app.state.to_tuple(), before)
        self.assertIn(str(self.app.state.score), self.app.score_label.cget("text"))

    def test_restart_resets_game(self):
        self.app.state.score = 500
        self.app.restart()
        self.assertEqual(self.app.state.score, 0)
        self.assertEqual(len(self.app.state.empty_cells()), 14)


if __name__ == "__main__":
    unittest.main()
