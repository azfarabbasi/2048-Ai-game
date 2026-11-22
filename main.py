"""2048 AI: play the game yourself or watch an AI agent play (Tkinter window).

Run with:  python main.py

The GUI contains no game rules. It only draws ``GameState`` and calls the
agents from ``agents.py``.

Why a background thread? Expectimax can need a second or more per move. If
that work ran on Tkinter's main thread the window would freeze. So the AI
runs in a worker thread, puts its answer in a queue, and the main thread
checks the queue every 30 ms and updates the board. (Tkinter widgets must only
be touched from the main thread.)
"""

import queue
import threading
import time
import tkinter as tk
from tkinter import ttk

from agents import ExpectimaxAgent, GreedyAgent, RandomAgent
from game import SIZE, GameState


# Our own colour ramp: light blue for small tiles through purple to gold.
TILE_COLOURS = {
    0: "#d6dde6", 2: "#f0f4f8", 4: "#d9e8f5", 8: "#a9d1f0", 16: "#7ab8e6",
    32: "#4f9fd9", 64: "#2f7fc4", 128: "#6c63d9", 256: "#8a4fd1",
    512: "#b03fc0", 1024: "#d9367f", 2048: "#f0a020",
}
BIG_TILE_COLOUR = "#2b2d42"
DARK_TEXT, LIGHT_TEXT = "#243447", "#ffffff"
BOARD_COLOUR = "#8a9bb0"

KEYS = {
    "Up": "up", "Down": "down", "Left": "left", "Right": "right",
    "w": "up", "s": "down", "a": "left", "d": "right",
}
AGENTS = ("Expectimax", "Greedy", "Random")


def make_agent(name, depth):
    if name == "Expectimax":
        return ExpectimaxAgent(depth=depth)
    if name == "Greedy":
        return GreedyAgent()
    return RandomAgent()


class Game2048App:
    def __init__(self, root):
        self.root = root
        root.title("2048 AI - Expectimax")
        root.configure(padx=16, pady=12)
        root.resizable(False, False)

        self.state = GameState()
        self.game_id = 0          # changes on restart, so old AI answers are ignored
        self.thinking = False     # True while the worker thread is searching
        self.auto_play = False
        self.celebrated = False
        self.results = queue.Queue()

        self._build_controls()
        self._build_board()
        self._build_buttons()
        root.bind("<Key>", self.on_key)
        self.redraw()
        self.root.after(30, self.check_results)

    # ----- layout -------------------------------------------------------
    def _build_controls(self):
        tk.Label(self.root, text="2048 AI", font=("Segoe UI", 24, "bold")).pack()

        options = tk.Frame(self.root)
        options.pack(pady=(6, 4))
        tk.Label(options, text="Agent:").grid(row=0, column=0, sticky="e")
        self.agent_var = tk.StringVar(value="Expectimax")
        ttk.Combobox(options, textvariable=self.agent_var, values=AGENTS,
                     state="readonly", width=12).grid(row=0, column=1, padx=(4, 12))
        tk.Label(options, text="Depth:").grid(row=0, column=2, sticky="e")
        self.depth_var = tk.StringVar(value="2")
        ttk.Combobox(options, textvariable=self.depth_var, values=("1", "2", "3", "4"),
                     state="readonly", width=3).grid(row=0, column=3, padx=(4, 12))
        tk.Label(options, text="Delay (ms):").grid(row=1, column=0, sticky="e")
        self.delay_var = tk.IntVar(value=100)
        tk.Scale(options, from_=0, to=1000, resolution=50, orient="horizontal",
                 variable=self.delay_var, length=200, showvalue=True).grid(
                     row=1, column=1, columnspan=3, sticky="w")

        stats = tk.Frame(self.root)
        stats.pack(pady=4)
        self.score_label = tk.Label(stats, font=("Segoe UI", 13, "bold"))
        self.score_label.pack()
        self.info_label = tk.Label(stats, font=("Segoe UI", 9), fg="#4a5568")
        self.info_label.pack()

    def _build_board(self):
        board = tk.Frame(self.root, bg=BOARD_COLOUR, padx=6, pady=6)
        board.pack(pady=6)
        self.cells = []
        for r in range(SIZE):
            row = []
            for c in range(SIZE):
                cell = tk.Label(board, width=5, height=2, font=("Segoe UI", 22, "bold"))
                cell.grid(row=r, column=c, padx=4, pady=4)
                row.append(cell)
            self.cells.append(row)

    def _build_buttons(self):
        buttons = tk.Frame(self.root)
        buttons.pack(pady=(6, 2))
        tk.Button(buttons, text="Restart", width=10, command=self.restart).pack(side="left", padx=4)
        self.step_button = tk.Button(buttons, text="AI Move", width=10, command=self.ai_move)
        self.step_button.pack(side="left", padx=4)
        self.auto_button = tk.Button(buttons, text="Auto Play", width=10, command=self.toggle_auto)
        self.auto_button.pack(side="left", padx=4)
        self.status = tk.Label(self.root, text="Use the arrow keys or W/A/S/D to play yourself.",
                               font=("Segoe UI", 10), wraplength=360)
        self.status.pack(pady=(6, 0))

    # ----- drawing ------------------------------------------------------
    def redraw(self):
        for r in range(SIZE):
            for c in range(SIZE):
                value = self.state.grid[r][c]
                self.cells[r][c].configure(
                    text=str(value) if value else "",
                    bg=TILE_COLOURS.get(value, BIG_TILE_COLOUR),
                    fg=DARK_TEXT if value <= 8 else LIGHT_TEXT,
                    font=("Segoe UI", 22 if value < 1024 else 18, "bold"),
                )
        self.score_label.configure(
            text=f"Score: {self.state.score}     Highest: {self.state.highest_tile()}")
        if self.state.reached_2048() and not self.celebrated:
            self.celebrated = True
            self.status.configure(text="2048 reached! The game continues.", fg="#2f855a")
        if self.state.is_terminal():
            self.auto_play = False
            self.auto_button.configure(text="Auto Play")
            self.status.configure(
                text=f"Game over! Final score {self.state.score}, "
                     f"highest tile {self.state.highest_tile()}. Press Restart.",
                fg="#c53030")

    # ----- actions ------------------------------------------------------
    def on_key(self, event):
        direction = KEYS.get(event.keysym) or KEYS.get(event.keysym.lower())
        if direction is None or self.thinking or self.auto_play:
            return
        if self.state.play(direction):
            self.info_label.configure(text=f"You moved {direction}.")
        self.redraw()

    def restart(self):
        self.game_id += 1
        self.auto_play = False
        self.thinking = False
        self.celebrated = False
        self.state = GameState()
        self.auto_button.configure(text="Auto Play")
        self.status.configure(text="New game. Play with the keys or let the AI move.", fg="black")
        self.info_label.configure(text="")
        self.redraw()

    def toggle_auto(self):
        self.auto_play = not self.auto_play
        self.auto_button.configure(text="Stop" if self.auto_play else "Auto Play")
        if self.auto_play:
            self.ai_move()

    def ai_move(self):
        """Start one AI decision in a background thread."""
        if self.thinking or self.state.is_terminal():
            return
        self.thinking = True
        self.step_button.configure(state="disabled")
        agent = make_agent(self.agent_var.get(), int(self.depth_var.get()))
        snapshot = self.state.copy()  # the worker never touches the live game
        game_id = self.game_id

        def work():
            started = time.perf_counter()
            move = agent.choose_move(snapshot)
            seconds = time.perf_counter() - started
            self.results.put((game_id, agent, move, seconds))

        threading.Thread(target=work, daemon=True).start()

    def check_results(self):
        """Runs every 30 ms on the main thread: apply any finished AI move."""
        try:
            game_id, agent, move, seconds = self.results.get_nowait()
        except queue.Empty:
            pass
        else:
            if game_id == self.game_id:  # ignore answers from before a restart
                self.apply_ai_move(agent, move, seconds)
        self.root.after(30, self.check_results)

    def apply_ai_move(self, agent, move, seconds):
        self.thinking = False
        self.step_button.configure(state="normal")
        if move is None:
            self.redraw()
            return
        self.state.play(move)
        details = f"AI chose {move} in {seconds * 1000:.0f} ms"
        if isinstance(agent, ExpectimaxAgent):
            details += f", {agent.nodes_expanded} nodes searched"
        values = getattr(agent, "move_values", {})
        if values:
            details += "\n" + "   ".join(f"{d}: {v:,.0f}" for d, v in values.items())
        self.info_label.configure(text=details)
        self.redraw()
        if self.auto_play and not self.state.is_terminal():
            self.root.after(self.delay_var.get(), self.ai_move)


def main():
    try:  # sharp text on high-resolution Windows screens
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except (ImportError, AttributeError, OSError):
        pass
    root = tk.Tk()
    Game2048App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
