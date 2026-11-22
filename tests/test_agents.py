import unittest

from agents import ExpectimaxAgent, GreedyAgent, RandomAgent, evaluate_board
from game import GameState


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
        outcomes = agent._spawn_outcomes(state)
        self.assertEqual(len(outcomes), 30)  # 15 empty cells, each with 2 or 4
        self.assertAlmostEqual(sum(probability for _, probability in outcomes), 1.0)
        original_total = sum(sum(row) for row in state.grid)
        twos = sum(
            probability for child, probability in outcomes
            if sum(sum(row) for row in child.grid) == original_total + 2
        )
        self.assertAlmostEqual(twos, 0.9)

    def test_chance_limit_is_deterministic_and_normalized(self):
        state = GameState([[2, 0, 0, 0], [0] * 4, [0] * 4, [0] * 4], start_tiles=False)
        agent = ExpectimaxAgent(depth=1, max_chance_outcomes=5)
        first = agent._spawn_outcomes(state)
        second = agent._spawn_outcomes(state)
        self.assertEqual(len(first), 5)
        self.assertEqual([p for _, p in first], [p for _, p in second])
        self.assertAlmostEqual(sum(p for _, p in first), 1.0)

    def test_expectimax_rejects_invalid_depth(self):
        with self.assertRaises(ValueError):
            ExpectimaxAgent(depth=0)


if __name__ == "__main__":
    unittest.main()
