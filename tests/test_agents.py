import math
import unittest

from agents import (
    LOSS_VALUE, ExpectimaxAgent, GreedyAgent, RandomAgent, _line_features, evaluate_board,
    place_tile,
)
from game import SIZE, GameState, can_move, possible_moves


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


class AgentTests(unittest.TestCase):
    def setUp(self):
        self.state = GameState(
            [[2, 2, 4, 0], [4, 8, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
            start_tiles=False,
        )

    def test_random_agent_returns_only_legal_moves(self):
        agent = RandomAgent(seed=3)
        for _ in range(10):
            self.assertIn(agent.choose_move(self.state), self.state.legal_moves())
        full = GameState([[2, 4, 2, 4], [4, 2, 4, 2], [2, 4, 2, 4], [4, 2, 4, 2]])
        self.assertIsNone(agent.choose_move(full))

    def test_greedy_returns_legal_move_and_does_not_change_state(self):
        before = [row[:] for row in self.state.grid]
        move = GreedyAgent().choose_move(self.state)
        self.assertIn(move, self.state.legal_moves())
        self.assertEqual(self.state.grid, before)

    def test_heuristic_rewards_empty_cells_and_corner_largest(self):
        empty = GameState([[2, 0, 0, 0], [0] * 4, [0] * 4, [0] * 4], start_tiles=False)
        crowded = GameState([[2, 4, 8, 16], [32, 64, 128, 256], [512, 1024, 2, 4], [8, 16, 32, 64]], start_tiles=False)
        self.assertGreater(evaluate_board(empty), evaluate_board(crowded))
        corner = GameState([[128, 2, 4, 8], [0] * 4, [0] * 4, [0] * 4], start_tiles=False)
        middle = GameState([[2, 0, 0, 0], [0, 128, 0, 0], [0] * 4, [0] * 4], start_tiles=False)
        # Compare with only the corner feature active, isolating that component.
        weights = {"empty": 0, "monotonicity": 0, "smoothness": 0, "corner": 1}
        self.assertGreater(evaluate_board(corner, weights), evaluate_board(middle, weights))

    def test_expectimax_depth_one_and_node_count(self):
        agent = ExpectimaxAgent(depth=1)
        move = agent.choose_move(self.state)
        self.assertIn(move, self.state.legal_moves())
        self.assertGreater(agent.nodes_expanded, 0)

    def test_chance_outcomes_match_spawn_probabilities(self):
        state = GameState([[2, 0, 0, 0], [0] * 4, [0] * 4, [0] * 4], start_tiles=False)
        agent = ExpectimaxAgent(depth=1)
        outcomes = agent._spawn_outcomes(state.to_tuple())
        self.assertEqual(len(outcomes), 30)  # 15 empty cells, each with 2 or 4
        self.assertAlmostEqual(sum(probability for _, probability in outcomes), 1.0)
        original_total = sum(sum(row) for row in state.grid)
        twos = sum(
            probability for child, probability in outcomes
            if sum(sum(row) for row in child) == original_total + 2
        )
        self.assertAlmostEqual(twos, 0.9)

    def test_chance_limit_is_deterministic_and_normalized(self):
        state = GameState([[2, 0, 0, 0], [0] * 4, [0] * 4, [0] * 4], start_tiles=False)
        agent = ExpectimaxAgent(depth=1, max_chance_outcomes=5)
        first = agent._spawn_outcomes(state.to_tuple())
        second = agent._spawn_outcomes(state.to_tuple())
        self.assertEqual(len(first), 5)
        self.assertEqual([p for _, p in first], [p for _, p in second])
        self.assertAlmostEqual(sum(p for _, p in first), 1.0)

    def test_expectimax_rejects_invalid_depth(self):
        with self.assertRaises(ValueError):
            ExpectimaxAgent(depth=0)

    def test_monotonic_line_has_no_penalty(self):
        self.assertEqual(_line_features((2, 4, 8, 16))[1], 0)
        self.assertEqual(_line_features((16, 8, 4, 2))[1], 0)
        self.assertLess(_line_features((2, 16, 4, 8))[1], 0)

    def test_chance_node_is_weighted_average(self):
        # Exactly one empty cell: value = 0.9 * (spawn 2) + 0.1 * (spawn 4).
        board = ((2, 2, 8, 16), (4, 8, 16, 32), (8, 16, 32, 64), (16, 32, 64, 0))
        agent = ExpectimaxAgent(depth=1)
        with_two = evaluate_board(place_tile(board, 3, 3, 2))
        with_four = evaluate_board(place_tile(board, 3, 3, 4))
        self.assertAlmostEqual(agent._chance_value(board, 1), 0.9 * with_two + 0.1 * with_four)

    def test_lost_board_gets_loss_value(self):
        locked = ((2, 4, 2, 4), (4, 2, 4, 2), (2, 4, 2, 4), (4, 2, 4, 2))
        self.assertEqual(ExpectimaxAgent(depth=2)._max_value(locked, 2), LOSS_VALUE)

    def test_caching_does_not_change_values(self):
        board = self.state.to_tuple()
        agent = ExpectimaxAgent(depth=2)
        agent.choose_move(self.state)
        for direction, child, _ in possible_moves(board):
            self.assertTrue(math.isclose(
                agent.move_values[direction], plain_expectimax(child, 2, False), rel_tol=1e-9))

    def test_deeper_search_expands_more_nodes(self):
        counts = []
        for depth in (1, 2):
            agent = ExpectimaxAgent(depth=depth)
            agent.choose_move(self.state)
            counts.append(agent.nodes_expanded)
        self.assertLess(counts[0], counts[1])


if __name__ == "__main__":
    unittest.main()
