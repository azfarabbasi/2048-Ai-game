"""Play Phase 1 of the 2048 project from a terminal."""

from game import GameState


def show(state):
    print("\nScore:", state.score, "| Highest tile:", state.highest_tile())
    for row in state.grid:
        print(" ".join(f"{value or '.':>5}" for value in row))


def main():
    game = GameState()
    print("2048 — W/A/S/D to move; Q to quit")
    celebrated = False
    while True:
        show(game)
        if game.is_terminal():
            print("Game over! No legal moves remain.")
            break
        choice = input("Your move: ").strip().lower()
        if choice == "q":
            break
        direction = {"w": "up", "a": "left", "s": "down", "d": "right"}.get(choice)
        if direction is None:
            print("Use W, A, S, D, or Q.")
            continue
        if not game.play(direction):
            print("That move did not change the board. Try another direction.")
        if game.reached_2048() and not celebrated:
            print("You reached 2048! You can keep playing.")
            celebrated = True


if __name__ == "__main__":
    main()
