"""Benchmark Togyzkumalak agents against each other.

Inspired by the battleground/evaluation scripts in Kalah/Mancala projects,
but implemented for this codebase and this game's rules.

Examples:
    python ai_benchmark.py --games 20 --time 0.2
    python ai_benchmark.py --agent1 ai --agent2 greedy --games 50
"""

import argparse
import random
import time
from dataclasses import dataclass

from ai import get_best_move_alphabeta
from game_logic import TogyzkumalakState


AGENTS = ("ai", "greedy", "random")


@dataclass
class BattleResult:
    games: int
    wins_agent1: int = 0
    wins_agent2: int = 0
    draws: int = 0
    invalid: int = 0
    avg_plies: float = 0.0

    @property
    def agent1_win_rate(self):
        return self.wins_agent1 / self.games if self.games else 0.0


def choose_move(state, agent, rng, max_time_seconds):
    moves = state.getPossibleMoves(state.currentPlayer)
    if not moves:
        return -1

    if agent == "random":
        return rng.choice(moves)
    if agent == "greedy":
        return _choose_greedy_move(state, rng)
    if agent == "ai":
        return get_best_move_alphabeta(state, state.currentPlayer, max_time_seconds=max_time_seconds)

    raise ValueError(f"unknown agent: {agent}")


def _choose_greedy_move(state, rng):
    player = state.currentPlayer
    best_gain = None
    best_moves = []

    for move in state.getPossibleMoves(player):
        child = state.clone()
        before = child.kazans[player]
        child.makeMove(move)
        gain = child.kazans[player] - before
        if best_gain is None or gain > best_gain:
            best_gain = gain
            best_moves = [move]
        elif gain == best_gain:
            best_moves.append(move)

    return rng.choice(best_moves)


def play_game(agent1, agent2, seed, max_time_seconds=0.2, max_plies=500):
    rng = random.Random(seed)
    state = TogyzkumalakState()
    plies = 0

    while not state.isGameOver and plies < max_plies:
        agent = agent1 if state.currentPlayer == 0 else agent2
        move = choose_move(state, agent, rng, max_time_seconds)
        if not state.makeMove(move):
            return 1 - state.currentPlayer, plies, True
        plies += 1

    if not state.isGameOver:
        if state.kazans[0] > state.kazans[1]:
            return 0, plies, False
        if state.kazans[1] > state.kazans[0]:
            return 1, plies, False
        return -1, plies, False

    return state.winner, plies, False


def battle(agent1, agent2, games=20, seed=42, max_time_seconds=0.2):
    result = BattleResult(games=games)
    total_plies = 0

    for i in range(games):
        first_agent = agent1 if i % 2 == 0 else agent2
        second_agent = agent2 if i % 2 == 0 else agent1
        winner, plies, invalid = play_game(
            first_agent,
            second_agent,
            seed + i,
            max_time_seconds=max_time_seconds,
        )
        total_plies += plies
        result.invalid += int(invalid)

        if winner == -1:
            result.draws += 1
        elif (winner == 0 and i % 2 == 0) or (winner == 1 and i % 2 == 1):
            result.wins_agent1 += 1
        else:
            result.wins_agent2 += 1

    result.avg_plies = total_plies / games if games else 0.0
    return result


def main():
    parser = argparse.ArgumentParser(description="Benchmark Togyzkumalak AI agents")
    parser.add_argument("--agent1", choices=AGENTS, default="ai")
    parser.add_argument("--agent2", choices=AGENTS, default="greedy")
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--time", type=float, default=0.2, help="seconds per AI move")
    args = parser.parse_args()

    started = time.time()
    result = battle(args.agent1, args.agent2, args.games, args.seed, args.time)
    elapsed = time.time() - started

    print(f"{args.agent1} vs {args.agent2}")
    print(f"games: {result.games}")
    print(f"{args.agent1} wins: {result.wins_agent1} ({result.agent1_win_rate:.1%})")
    print(f"{args.agent2} wins: {result.wins_agent2}")
    print(f"draws: {result.draws}")
    print(f"invalid games: {result.invalid}")
    print(f"avg plies: {result.avg_plies:.1f}")
    print(f"elapsed: {elapsed:.1f}s")


if __name__ == "__main__":
    main()

