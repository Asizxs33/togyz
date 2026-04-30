"""
Self-play trainer for the Togyzkumalak AI.

Runs alpha-beta vs alpha-beta games locally and feeds each finished game
into the production learning store. Two delivery modes:

    HTTP mode (default, no setup):
        python selfplay_trainer.py --games 200 --api https://togyz.onrender.com

    Direct DB mode (much faster, requires DATABASE_URL):
        export DATABASE_URL='postgres://user:pass@host/db'
        python selfplay_trainer.py --games 1000

Tunables:
    --time      seconds per move for alpha-beta (default 0.3)
    --epsilon   exploration rate — fraction of moves picked at random
                so self-play games don't all look identical (default 0.10)
"""
import argparse
import json
import os
import random
import sys
import time
import urllib.request
import urllib.error
from collections import Counter

from game_logic import TogyzkumalakState
from ai import get_best_move_alphabeta, record_learning


def play_one_game(time_per_move: float, epsilon: float, max_moves: int = 400):
    state = TogyzkumalakState()
    samples = []
    moves_played = 0

    while not state.isGameOver and moves_played < max_moves:
        possible = state.getPossibleMoves(state.currentPlayer)
        if not possible:
            break

        if random.random() < epsilon and len(possible) > 1:
            move = random.choice(possible)
        else:
            move = get_best_move_alphabeta(
                state, state.currentPlayer, max_time_seconds=time_per_move
            )
            if move not in possible:
                move = possible[0]

        samples.append({
            "board": state.board[:],
            "kazans": state.kazans[:],
            "tuzdyks": state.tuzdyks[:],
            "player": state.currentPlayer,
            "move": move,
        })
        state.makeMove(move)
        moves_played += 1

    return state, samples


def submit_direct(samples, winner):
    saved = 0
    for player in (0, 1):
        saved += record_learning(samples, winner, ai_player=player) or 0
    return saved


def submit_http(samples, winner, api_url):
    saved = 0
    url = f"{api_url.rstrip('/')}/api/learn"
    for player in (0, 1):
        body = json.dumps({
            "samples": samples,
            "winner": winner,
            "aiPlayer": player,
        }).encode("utf-8")
        req = urllib.request.Request(
            url, data=body,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                payload = json.loads(resp.read())
                saved += int(payload.get("learned", 0))
        except urllib.error.URLError as exc:
            print(f"  [warn] /api/learn failed for player={player}: {exc}")
    return saved


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--games", type=int, default=100, help="number of games to play")
    p.add_argument("--time", type=float, default=0.3, help="seconds per move for alpha-beta")
    p.add_argument("--epsilon", type=float, default=0.10, help="random-move exploration rate")
    p.add_argument("--api", default=None, help="HTTP mode: base URL of the backend (e.g. https://togyz.onrender.com)")
    p.add_argument("--report-every", type=int, default=5, help="print stats every N games")
    p.add_argument("--seed", type=int, default=None, help="RNG seed for reproducibility")
    args = p.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    if args.api:
        print(f"[mode] HTTP -> {args.api}")
        submit = lambda samples, winner: submit_http(samples, winner, args.api)
    else:
        if not os.environ.get("DATABASE_URL"):
            print("[warn] DATABASE_URL is not set — record_learning will fall back to a local JSON file.")
            print("       Set DATABASE_URL or use --api to send to a remote backend.")
        print("[mode] direct (record_learning)")
        submit = submit_direct

    print(f"[config] games={args.games} time/move={args.time}s epsilon={args.epsilon}")

    wins = Counter()
    total_records = 0
    started = time.time()

    try:
        for i in range(1, args.games + 1):
            game_start = time.time()
            state, samples = play_one_game(args.time, args.epsilon)
            wins[state.winner] += 1
            saved = submit(samples, state.winner)
            total_records += saved
            game_time = time.time() - game_start

            if i % args.report_every == 0 or i == args.games:
                elapsed = time.time() - started
                eta = elapsed / i * (args.games - i)
                print(
                    f"  [{i}/{args.games}] "
                    f"P0={wins[0]} P1={wins[1]} draw={wins[-1]} "
                    f"| moves={len(samples)} last={game_time:.1f}s "
                    f"records={total_records} "
                    f"elapsed={elapsed:.0f}s eta={eta:.0f}s"
                )
    except KeyboardInterrupt:
        print("\n[abort] interrupted")
        sys.exit(130)

    print(f"[done] total_records={total_records} elapsed={time.time()-started:.0f}s")


if __name__ == "__main__":
    main()
