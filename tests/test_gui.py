import time
import unittest

try:
    import tkinter as tk
    from gui import Game2048App
except ImportError:  # Python built without Tkinter
    tk = None


class GuiTests(unittest.TestCase):
    def setUp(self):
        if tk is None:
            self.skipTest("Tkinter is not installed")
        try:
            self.root = tk.Tk()
        except tk.TclError:
            self.skipTest("No display available")
        self.root.withdraw()
        self.app = Game2048App(self.root)

    def tearDown(self):
        self.root.destroy()

    def wait_for_ai(self, seconds=10):
        end = time.time() + seconds
        while self.app.thinking and time.time() < end:
            self.root.update()
            time.sleep(0.01)

    def test_ai_move_updates_board_and_score_display(self):
        self.app.agent_var.set("Expectimax")
        self.app.depth_var.set("1")
        before = self.app.state.to_tuple()
        self.app.ai_move()
        self.wait_for_ai()
        self.assertNotEqual(self.app.state.to_tuple(), before)
        self.assertIn(str(self.app.state.score), self.app.score_label.cget("text"))

    def test_restart_resets_game(self):
        self.app.state.score = 500
        self.app.restart()
        self.assertEqual(self.app.state.score, 0)
        self.assertEqual(len(self.app.state.empty_cells()), 14)


if __name__ == "__main__":
    unittest.main()
