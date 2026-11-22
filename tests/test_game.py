import unittest

import random

from game import GameState, can_move, merge_line, possible_moves, slide_board


class GameTests(unittest.TestCase):
    def test_merge_once(self):
        self.assertEqual(merge_line([2, 2, 2, 2]), ([4, 4, 0, 0], 8))
        self.assertEqual(merge_line([2, 2, 4, 0]), ([4, 4, 0, 0], 4))
        self.assertEqual(merge_line([0, 2, 0, 2]), ([4, 0, 0, 0], 4))

    def test_right_move_and_score(self):
        state = GameState([[2, 0, 2, 4], [0] * 4, [0] * 4, [0] * 4])
        self.assertTrue(state.move("right"))
        self.assertEqual(state.grid[0], [0, 0, 4, 4])
        self.assertEqual(state.score, 4)

    def test_vertical_merge(self):
        state = GameState([[2, 0, 0, 0], [2, 0, 0, 0], [4, 0, 0, 0], [0, 0, 0, 0]])
        self.assertTrue(state.move("up"))
        self.assertEqual([row[0] for row in state.grid], [4, 4, 0, 0])

    def test_illegal_move_does_not_spawn(self):
        state = GameState([[2, 0, 0, 0], [0] * 4, [0] * 4, [0] * 4], seed=10)
        self.assertFalse(state.play("left"))
        self.assertEqual(len(state.empty_cells()), 15)

    def test_legal_move_spawns_one_tile(self):
        state = GameState([[2, 0, 0, 0], [0] * 4, [0] * 4, [0] * 4], seed=10)
        self.assertTrue(state.play("right"))
        self.assertEqual(len(state.empty_cells()), 14)

    def test_no_legal_moves(self):
        state = GameState([[2, 4, 2, 4], [4, 2, 4, 2], [2, 4, 2, 4], [4, 2, 4, 2]])
        self.assertTrue(state.is_terminal())
        self.assertEqual(state.legal_moves(), [])

    def test_seed_is_reproducible(self):
        self.assertEqual(GameState(seed=42).grid, GameState(seed=42).grid)

    def test_copy_is_independent(self):
        state = GameState(seed=42)
        clone = state.copy()
        clone.grid[0][0] = 128
        self.assertNotEqual(state.grid, clone.grid)

    def test_tile_merges_only_once_per_move(self):
        # The new 8 must not merge again with the existing 8 in the same move.
        self.assertEqual(merge_line([4, 4, 8, 0]), ([8, 8, 0, 0], 8))

    def test_all_four_directions(self):
        board = ((2, 0, 0, 2), (0, 0, 0, 0), (0, 0, 0, 0), (2, 0, 0, 0))
        self.assertEqual(slide_board(board, "left")[0][0], (4, 0, 0, 0))
        self.assertEqual(slide_board(board, "right")[0][0], (0, 0, 0, 4))
        self.assertEqual([row[0] for row in slide_board(board, "up")[0]], [4, 0, 0, 0])
        self.assertEqual([row[0] for row in slide_board(board, "down")[0]], [0, 0, 0, 4])

    def test_possible_moves_match_legal_moves(self):
        state = GameState([[2, 2, 8, 16], [32, 64, 128, 256], [2, 4, 8, 16], [32, 64, 128, 256]])
        self.assertEqual(state.legal_moves(), ["left", "right"])
        self.assertEqual([d for d, _, _ in possible_moves(state.to_tuple())], ["left", "right"])

    def test_can_move(self):
        full_locked = ((2, 4, 2, 4), (4, 2, 4, 2), (2, 4, 2, 4), (4, 2, 4, 2))
        full_with_pair = ((2, 2, 4, 8), (4, 8, 16, 32), (8, 16, 32, 64), (16, 32, 64, 128))
        self.assertFalse(can_move(full_locked))
        self.assertTrue(can_move(full_with_pair))
        self.assertTrue(can_move(((0, 2, 4, 8),) + full_locked[1:]))

    def test_spawn_probabilities_are_about_90_10(self):
        rng = random.Random(0)
        fours = 0
        for _ in range(5000):
            state = GameState(start_tiles=False)
            state.rng = random.Random(rng.random())
            state.spawn()
            fours += 4 in [n for row in state.grid for n in row]
        self.assertAlmostEqual(fours / 5000, 0.1, delta=0.02)


if __name__ == "__main__":
    unittest.main()
