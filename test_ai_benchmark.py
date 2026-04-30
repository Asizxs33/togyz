from ai_benchmark import battle, choose_move
from game_logic import TogyzkumalakState


def test_benchmark_smoke():
    result = battle("random", "greedy", games=2, seed=7, max_time_seconds=0.01)
    assert result.games == 2
    assert result.wins_agent1 + result.wins_agent2 + result.draws == 2
    assert result.avg_plies > 0


def test_agent_moves_are_legal():
    state = TogyzkumalakState()
    move = choose_move(state, "greedy", rng=__import__("random").Random(1), max_time_seconds=0.01)
    assert move in state.getPossibleMoves(state.currentPlayer)


if __name__ == "__main__":
    test_benchmark_smoke()
    test_agent_moves_are_legal()
    print("ai benchmark ok")

