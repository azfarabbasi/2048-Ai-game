import unittest

from game import GameState, merge_line


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


if __name__ == "__main__":
    unittest.main()
