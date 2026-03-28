"""
Togyzkumalak Neural Network Training  (v2 — stronger)
======================================================
Architecture: 34 → 256 → 128 → 64 → 1  (3 hidden layers)
Features: 34 (board + kazans + tuzdyk + tactical)

Phase 1: Sample 10 000 diverse positions, label with heuristic eval
Phase 2: Self-play games with alpha-beta (0.25 s/move), label with outcomes
Phase 3: Train with Adam, 120 epochs, save weights

Usage:
    python3 train_ai.py           # default: 300 self-play games (~2 h)
    python3 train_ai.py 100       # faster test run
"""

import numpy as np
import random
import time
import os
import sys

from game_logic import TogyzkumalakState
from ai import get_best_move_alphabeta, _evaluate_heuristic as heuristic_eval

# ── Architecture ───────────────────────────────────────────────────────────────
INPUT_SIZE = 34
H1, H2, H3 = 256, 128, 64
WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), 'model_weights.npz')


def state_to_features(state: TogyzkumalakState) -> np.ndarray:
    f = np.empty(INPUT_SIZE, dtype=np.float32)
    cur = state.currentPlayer
    opp = 1 - cur
    cs, os_ = cur * 9, opp * 9

    # [0..8]  — my pockets
    # [9..17] — opponent pockets
    for i in range(9):
        f[i]     = state.board[cs + i] / 20.0
        f[9 + i] = state.board[os_ + i] / 20.0

    # [18..20] — kazans
    f[18] = state.kazans[cur] / 162.0
    f[19] = state.kazans[opp] / 162.0
    f[20] = (state.kazans[cur] - state.kazans[opp]) / 162.0

    # [21..24] — tuzdyk flags + positions
    f[21] = 1.0 if state.tuzdyks[cur] != -1 else 0.0
    f[22] = 1.0 if state.tuzdyks[opp] != -1 else 0.0
    f[23] = (state.tuzdyks[cur]  % 9) / 8.0 if state.tuzdyks[cur]  != -1 else -1.0
    f[24] = (state.tuzdyks[opp] % 9) / 8.0 if state.tuzdyks[opp] != -1 else -1.0

    # [25] — opponent even-count pockets (capture opportunities for me)
    f[25] = sum(1 for i in range(os_, os_+9)
                if state.board[i] > 0 and state.board[i] % 2 == 0) / 9.0

    # [26] — my even-count pockets (vulnerabilities)
    f[26] = sum(1 for i in range(cs, cs+9)
                if state.board[i] > 0 and state.board[i] % 2 == 0) / 9.0

    # [27] — game phase: 1.0 = full board, 0.0 = empty
    f[27] = sum(state.board) / 162.0

    # [28] — can I create tuzdyk this turn?
    f[28] = 0.0
    if state.tuzdyks[cur] == -1:
        for i in range(os_, os_+9):
            if state.board[i] == 2 and i not in (8, 17):
                if state.tuzdyks[opp] == -1 or state.tuzdyks[opp] % 9 != i % 9:
                    f[28] = 1.0
                    break

    # [29] — can opponent create tuzdyk next turn?
    f[29] = 0.0
    if state.tuzdyks[opp] == -1:
        for i in range(cs, cs+9):
            if state.board[i] == 2 and i not in (8, 17):
                if state.tuzdyks[cur] == -1 or state.tuzdyks[cur] % 9 != i % 9:
                    f[29] = 1.0
                    break

    # [30] — total capturable stones on opponent side / 162
    f[30] = sum(state.board[i] for i in range(os_, os_+9)
                if state.board[i] > 0 and state.board[i] % 2 == 0) / 162.0

    # [31] — total vulnerable stones on my side / 162
    f[31] = sum(state.board[i] for i in range(cs, cs+9)
                if state.board[i] > 0 and state.board[i] % 2 == 0) / 162.0

    # [32] — my total stones on board
    f[32] = sum(state.board[cs:cs+9]) / 162.0

    # [33] — opponent total stones on board
    f[33] = sum(state.board[os_:os_+9]) / 162.0

    return f


# ── Network (pure numpy) ───────────────────────────────────────────────────────

def relu(x):  return np.maximum(0.0, x)
def drelu(x): return (x > 0).astype(np.float32)


def init_weights(seed=42):
    rng = np.random.default_rng(seed)
    return {
        'W1': rng.standard_normal((INPUT_SIZE, H1)).astype(np.float32) * np.sqrt(2.0 / INPUT_SIZE),
        'b1': np.zeros(H1, dtype=np.float32),
        'W2': rng.standard_normal((H1, H2)).astype(np.float32) * np.sqrt(2.0 / H1),
        'b2': np.zeros(H2, dtype=np.float32),
        'W3': rng.standard_normal((H2, H3)).astype(np.float32) * np.sqrt(2.0 / H2),
        'b3': np.zeros(H3, dtype=np.float32),
        'W4': rng.standard_normal((H3, 1)).astype(np.float32) * np.sqrt(2.0 / H3),
        'b4': np.zeros(1, dtype=np.float32),
    }


def nn_forward(w, x):
    h1  = relu(x  @ w['W1'] + w['b1'])
    h2  = relu(h1 @ w['W2'] + w['b2'])
    h3  = relu(h2 @ w['W3'] + w['b3'])
    out = np.tanh(h3 @ w['W4'] + w['b4'])
    return out.reshape(-1), h1, h2, h3


def nn_predict(w, x):
    out, _, _, _ = nn_forward(w, x)
    return float(out)


# ── Data generation ────────────────────────────────────────────────────────────

def sample_heuristic_positions(n=10000):
    """Random rollouts → label with heuristic eval."""
    print(f"  Sampling {n} positions...", flush=True)
    samples = []
    while len(samples) < n:
        state = TogyzkumalakState()
        depth = random.randint(2, 60)
        for _ in range(depth):
            if state.isGameOver:
                break
            moves = state.getPossibleMoves(state.currentPlayer)
            if not moves:
                break
            state.makeMove(random.choice(moves))
        if state.isGameOver:
            continue
        feat  = state_to_features(state)
        label = np.tanh(heuristic_eval(state) / 40.0)
        samples.append((feat, label))
    print(f"  → {len(samples)} heuristic samples", flush=True)
    return samples


def generate_self_play_game(time_per_move=0.25, epsilon=0.06):
    """One alpha-beta self-play game with epsilon-random exploration."""
    state = TogyzkumalakState()
    traj  = []

    while not state.isGameOver:
        player = state.currentPlayer
        traj.append((state_to_features(state), player))

        if random.random() < epsilon:
            moves = state.getPossibleMoves(player)
            move  = random.choice(moves) if moves else -1
        else:
            move  = get_best_move_alphabeta(state, player, max_time_seconds=time_per_move)

        if move == -1:
            break
        state.makeMove(move)

    winner = state.winner
    out = []
    for feat, player in traj:
        if winner == -1:
            label = 0.0
        elif winner == player:
            label = 1.0
        else:
            label = -1.0
        out.append((feat, label))
    return out, winner


# ── Training (mini-batch Adam) ─────────────────────────────────────────────────

def train(w, samples, epochs=120, batch=256, lr=0.001):
    random.shuffle(samples)
    X = np.array([s[0] for s in samples], dtype=np.float32)
    y = np.array([s[1] for s in samples], dtype=np.float32).reshape(-1, 1)
    N = len(X)

    m = {k: np.zeros_like(v) for k, v in w.items()}
    v = {k: np.zeros_like(v) for k, v in w.items()}
    t = 0
    b1, b2, eps = 0.9, 0.999, 1e-8

    for epoch in range(epochs):
        idx      = np.random.permutation(N)
        loss_sum = 0.0
        n_batches = 0

        for start in range(0, N, batch):
            bi      = idx[start:start + batch]
            xb, yb  = X[bi], y[bi]
            n       = len(xb)

            # Forward
            out, h1, h2, h3 = nn_forward(w, xb)
            out = out.reshape(-1, 1)

            # MSE loss
            diff = out - yb
            loss_sum += float(np.mean(diff ** 2))
            n_batches += 1

            # Backprop through tanh output + 3 ReLU layers
            d4  = (2.0 / n) * diff * (1.0 - out ** 2)
            gW4 = h3.T @ d4
            gb4 = d4.sum(axis=0)

            d3  = (d4 @ w['W4'].T) * drelu(h3)
            gW3 = h2.T @ d3
            gb3 = d3.sum(axis=0)

            d2  = (d3 @ w['W3'].T) * drelu(h2)
            gW2 = h1.T @ d2
            gb2 = d2.sum(axis=0)

            d1  = (d2 @ w['W2'].T) * drelu(h1)
            gW1 = xb.T @ d1
            gb1 = d1.sum(axis=0)

            grads = dict(W1=gW1, b1=gb1, W2=gW2, b2=gb2,
                         W3=gW3, b3=gb3, W4=gW4, b4=gb4)

            # Adam update
            t += 1
            for k in w:
                m[k] = b1 * m[k] + (1 - b1) * grads[k]
                v[k] = b2 * v[k] + (1 - b2) * grads[k] ** 2
                mh   = m[k] / (1 - b1 ** t)
                vh   = v[k] / (1 - b2 ** t)
                w[k] -= lr * mh / (np.sqrt(vh) + eps)

        if (epoch + 1) % 20 == 0:
            print(f"    epoch {epoch+1:3d}/{epochs}  loss={loss_sum/n_batches:.4f}", flush=True)

    return w


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    n_games = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    t_start = time.time()

    print("=" * 55)
    print("  Togyzkumalak Neural Network Training  (v2)")
    print(f"  Architecture: {INPUT_SIZE}→{H1}→{H2}→{H3}→1")
    print("=" * 55)

    # Phase 1
    print("\n[Phase 1/3] Heuristic position sampling (10 000)")
    samples = sample_heuristic_positions(n=10000)

    # Phase 2
    print(f"\n[Phase 2/3] Self-play ({n_games} games, 0.25 s/move)")
    wins = {0: 0, 1: 0, -1: 0}
    for i in range(n_games):
        t0 = time.time()
        game_samples, winner = generate_self_play_game(time_per_move=0.25)
        wins[winner] += 1
        samples.extend(game_samples)
        print(f"  game {i+1:3d}/{n_games}  pos={len(game_samples):3d}  "
              f"winner={winner}  ({time.time()-t0:.1f}s)", flush=True)

    print(f"\nTotal samples: {len(samples)}")
    print(f"Wins: P0={wins[0]}  P1={wins[1]}  Draw={wins[-1]}")

    # Phase 3
    print("\n[Phase 3/3] Training (120 epochs, Adam)")
    w = init_weights()
    w = train(w, samples, epochs=120, lr=0.001)

    np.savez(WEIGHTS_PATH, **w)
    print(f"\nSaved → {WEIGHTS_PATH}")
    print(f"Total time: {(time.time()-t_start)/60:.1f} min")
    print("\nAI will now use the new v2 neural network.")
