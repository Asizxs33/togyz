"""
Togyzkumalak Neural Network Training
=====================================
Phase 1: Sample diverse positions, label with handcrafted eval (fast)
Phase 2: Self-play games with alpha-beta, label with game outcomes
Phase 3: Train 3-layer MLP with Adam, save weights

Usage:
    python3 train_ai.py           # default: 40 self-play games
    python3 train_ai.py 80        # 80 self-play games (stronger but slower)
"""

import numpy as np
import random
import time
import os
import sys

from game_logic import TogyzkumalakState
from ai import get_best_move_alphabeta, evaluate as heuristic_eval

# ── Architecture ──────────────────────────────────────────────────────────────
INPUT_SIZE = 25
H1, H2 = 128, 64
WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), 'model_weights.npz')


def state_to_features(state: TogyzkumalakState) -> np.ndarray:
    f = np.empty(INPUT_SIZE, dtype=np.float32)
    cur = state.currentPlayer
    opp = 1 - cur
    cs, os_ = cur * 9, opp * 9

    # Board stones from current player's perspective (rotated so cur is always first)
    for i in range(9):
        f[i]     = state.board[cs + i] / 20.0
        f[9 + i] = state.board[os_ + i] / 20.0

    # Kazans
    f[18] = state.kazans[cur] / 162.0
    f[19] = state.kazans[opp] / 162.0

    # Score difference
    f[20] = (state.kazans[cur] - state.kazans[opp]) / 162.0

    # Tuzdyk flags
    f[21] = 1.0 if state.tuzdyks[cur] != -1 else 0.0
    f[22] = 1.0 if state.tuzdyks[opp] != -1 else 0.0

    # Tuzdyk positions (relative, -1 if none)
    f[23] = (state.tuzdyks[cur]  % 9) / 8.0 if state.tuzdyks[cur]  != -1 else -1.0
    f[24] = (state.tuzdyks[opp] % 9) / 8.0 if state.tuzdyks[opp] != -1 else -1.0

    return f


# ── Network (pure numpy, no deps) ────────────────────────────────────────────

def relu(x):   return np.maximum(0.0, x)
def drelu(x):  return (x > 0).astype(np.float32)

def init_weights(seed=42):
    rng = np.random.default_rng(seed)
    return {
        'W1': rng.standard_normal((INPUT_SIZE, H1)).astype(np.float32) * np.sqrt(2.0 / INPUT_SIZE),
        'b1': np.zeros(H1, dtype=np.float32),
        'W2': rng.standard_normal((H1, H2)).astype(np.float32) * np.sqrt(2.0 / H1),
        'b2': np.zeros(H2, dtype=np.float32),
        'W3': rng.standard_normal((H2, 1)).astype(np.float32) * np.sqrt(2.0 / H2),
        'b3': np.zeros(1, dtype=np.float32),
    }

def nn_forward(w, x):
    h1  = relu(x @ w['W1'] + w['b1'])
    h2  = relu(h1 @ w['W2'] + w['b2'])
    out = np.tanh(h2 @ w['W3'] + w['b3'])
    return out.reshape(-1), h1, h2

def nn_predict(w, x):
    out, _, _ = nn_forward(w, x)
    return float(out)


# ── Data generation ──────────────────────────────────────────────────────────

def sample_heuristic_positions(n=8000):
    """Fast: random rollouts → label with handcrafted eval."""
    print(f"  Sampling {n} positions via random rollouts...", flush=True)
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
        label = np.tanh(heuristic_eval(state) / 40.0)   # squash to [-1,1]
        samples.append((feat, label))
    print(f"  → {len(samples)} heuristic samples", flush=True)
    return samples


def generate_self_play_game(time_per_move=0.12, epsilon=0.08):
    """One alpha-beta self-play game with epsilon-random exploration."""
    state = TogyzkumalakState()
    traj  = []   # (features, player)

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


# ── Training (mini-batch Adam) ────────────────────────────────────────────────

def train(w, samples, epochs=80, batch=256, lr=0.001):
    random.shuffle(samples)
    X = np.array([s[0] for s in samples], dtype=np.float32)
    y = np.array([s[1] for s in samples], dtype=np.float32).reshape(-1, 1)
    N = len(X)

    # Adam state
    m = {k: np.zeros_like(v) for k, v in w.items()}
    v = {k: np.zeros_like(v) for k, v in w.items()}
    t = 0
    b1, b2, eps = 0.9, 0.999, 1e-8

    for epoch in range(epochs):
        idx  = np.random.permutation(N)
        loss_sum = 0.0
        n_batches = 0

        for start in range(0, N, batch):
            bi  = idx[start:start + batch]
            xb, yb = X[bi], y[bi]
            n = len(xb)

            # Forward
            out, h1, h2 = nn_forward(w, xb)
            out = out.reshape(-1, 1)

            # MSE loss + gradient
            diff = out - yb
            loss_sum += float(np.mean(diff ** 2))
            n_batches += 1

            # Backprop
            d3 = (2.0 / n) * diff * (1.0 - out ** 2)   # through tanh
            gW3 = h2.T @ d3
            gb3 = d3.sum(axis=0)
            d2  = (d3 @ w['W3'].T) * drelu(h2)
            gW2 = h1.T @ d2
            gb2 = d2.sum(axis=0)
            d1  = (d2 @ w['W2'].T) * drelu(h1)
            gW1 = xb.T @ d1
            gb1 = d1.sum(axis=0)

            grads = dict(W1=gW1, b1=gb1, W2=gW2, b2=gb2, W3=gW3, b3=gb3)

            # Adam
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


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    n_games = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    t_start = time.time()

    print("=" * 50)
    print("  Togyzkumalak Neural Network Training")
    print("=" * 50)

    # Phase 1 — heuristic labels (fast)
    print("\n[Phase 1/3] Heuristic position sampling")
    samples = sample_heuristic_positions(n=8000)

    # Phase 2 — self-play
    print(f"\n[Phase 2/3] Self-play ({n_games} games, ~0.12s/move)")
    wins = {0: 0, 1: 0, -1: 0}
    for i in range(n_games):
        t0 = time.time()
        game_samples, winner = generate_self_play_game(time_per_move=0.12)
        wins[winner] += 1
        samples.extend(game_samples)
        print(f"  game {i+1:3d}/{n_games}  positions={len(game_samples):3d}  "
              f"winner={winner}  ({time.time()-t0:.1f}s)", flush=True)

    print(f"\nTotal samples: {len(samples)}")
    print(f"Wins: P0={wins[0]}  P1={wins[1]}  Draw={wins[-1]}")

    # Phase 3 — train
    print("\n[Phase 3/3] Training (80 epochs, Adam)")
    w = init_weights()
    w = train(w, samples, epochs=80, lr=0.001)

    # Save
    np.savez(WEIGHTS_PATH, **w)
    print(f"\nSaved → {WEIGHTS_PATH}")
    print(f"Total time: {(time.time()-t_start)/60:.1f} min")
    print("\nAI will now use the neural network for evaluation.")
