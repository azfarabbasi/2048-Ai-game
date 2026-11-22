"""Run each Phase 2 agent on the same small example board."""

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
        ("Expectimax depth 2", ExpectimaxAgent(depth=2)),
    )
    print("Example board:")
    for row in state.grid:
        print(row)
    print("Legal moves:", state.legal_moves())
    for name, agent in agents:
        move = agent.choose_move(state)
        detail = f"; nodes expanded: {agent.nodes_expanded}" if isinstance(agent, ExpectimaxAgent) else ""
        print(f"{name} chooses: {move}{detail}")


if __name__ == "__main__":
    main()
