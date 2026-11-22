"""Show how each agent decides on the same small example board."""

from agents import ExpectimaxAgent, GreedyAgent, RandomAgent
from game import GameState


def main():
    state = GameState(
        [[2, 2, 4, 0], [4, 8, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
        start_tiles=False,
    )
    agents = (
        ("Random", RandomAgent(seed=7)),
        ("Greedy", GreedyAgent()),
        ("Expectimax depth 1", ExpectimaxAgent(depth=1)),
        ("Expectimax depth 2", ExpectimaxAgent(depth=2)),
        ("Expectimax depth 3", ExpectimaxAgent(depth=3)),
    )
    print("Example board:")
    for row in state.grid:
        print(row)
    print("Legal moves:", state.legal_moves())
    for name, agent in agents:
        move = agent.choose_move(state)
        print(f"\n{name} chooses: {move}")
        if isinstance(agent, ExpectimaxAgent):
            print(f"  nodes expanded: {agent.nodes_expanded}")
        for direction, value in getattr(agent, "move_values", {}).items():
            print(f"  value of {direction:<5}: {value:10.1f}")


if __name__ == "__main__":
    main()
