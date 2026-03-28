"""
Togyzkumalak  —  Policy + Value Network via Self-Play  (AlphaZero-lite)
========================================================================
Architecture: 34 → 256 → 128 → 64 → {value(1 tanh), policy(9 softmax)}

Policy head  : predicts the probability of each pocket being the best move.
               Used in alpha-beta for move ordering → effectively deeper search.
Value head   : same as before, evaluates position from current player's POV.

Training pipeline
-----------------
Phase 1 — Expert positions  (fast)
    Random game rollouts → label value with heuristic, policy with alpha-beta.

Phase 2 — Self-play episodes
    Both players follow the policy network (temperature softmax).
    Label value with game outcome; policy again from alpha-beta.

Phase 3 — Joint Adam training
    Loss = value_MSE + policy_CrossEntropy + entropy_bonus

Result
------
Saves model_weights.npz with keys:
    W1,b1, W2,b2, W3,b3, Wv,bv (value), Wp,bp (policy)
ai.py will use the policy for move ordering and value for evaluation.

Usage
-----
    python3 train_rl.py          # 200 self-play games (~1.5 h)
    python3 train_rl.py 100      # faster (~50 min)
"""

import numpy as np
import random, time, os, sys

from game_logic import TogyzkumalakState
from ai import get_best_move_alphabeta, _evaluate_heuristic

# ── Hyper-parameters ──────────────────────────────────────────────────────────
INPUT_SIZE = 34
H1, H2, H3 = 256, 128, 64
N_ACTIONS   = 9          # pockets 0-8 from the current player's perspective
WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), 'model_weights.npz')


# ── Feature extraction (identical to train_ai.py v2) ─────────────────────────
def state_to_features(state: TogyzkumalakState) -> np.ndarray:
    f = np.empty(INPUT_SIZE, dtype=np.float32)
    cur = state.currentPlayer
    opp = 1 - cur
    cs, os_ = cur * 9, opp * 9

    for i in range(9):
        f[i]     = state.board[cs + i]  / 20.0
        f[9 + i] = state.board[os_ + i] / 20.0

    f[18] = state.kazans[cur] / 162.0
    f[19] = state.kazans[opp] / 162.0
    f[20] = (state.kazans[cur] - state.kazans[opp]) / 162.0
    f[21] = 1.0 if state.tuzdyks[cur] != -1 else 0.0
    f[22] = 1.0 if state.tuzdyks[opp] != -1 else 0.0
    f[23] = (state.tuzdyks[cur]  % 9) / 8.0 if state.tuzdyks[cur]  != -1 else -1.0
    f[24] = (state.tuzdyks[opp] % 9) / 8.0 if state.tuzdyks[opp] != -1 else -1.0

    f[25] = sum(1 for i in range(os_, os_+9) if state.board[i] > 0 and state.board[i] % 2 == 0) / 9.0
    f[26] = sum(1 for i in range(cs,  cs+9)  if state.board[i] > 0 and state.board[i] % 2 == 0) / 9.0
    f[27] = sum(state.board) / 162.0

    f[28] = 0.0
    if state.tuzdyks[cur] == -1:
        for i in range(os_, os_+9):
            if state.board[i] == 2 and i not in (8, 17):
                if state.tuzdyks[opp] == -1 or state.tuzdyks[opp] % 9 != i % 9:
                    f[28] = 1.0; break

    f[29] = 0.0
    if state.tuzdyks[opp] == -1:
        for i in range(cs, cs+9):
            if state.board[i] == 2 and i not in (8, 17):
                if state.tuzdyks[cur] == -1 or state.tuzdyks[cur] % 9 != i % 9:
                    f[29] = 1.0; break

    f[30] = sum(state.board[i] for i in range(os_, os_+9) if state.board[i] > 0 and state.board[i] % 2 == 0) / 162.0
    f[31] = sum(state.board[i] for i in range(cs,  cs+9)  if state.board[i] > 0 and state.board[i] % 2 == 0) / 162.0
    f[32] = sum(state.board[cs:cs+9])   / 162.0
    f[33] = sum(state.board[os_:os_+9]) / 162.0

    return f


# ── Network ───────────────────────────────────────────────────────────────────
def relu(x):       return np.maximum(0.0, x)
def drelu(x):      return (x > 0).astype(np.float32)

def softmax_fn(x):
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e / (e.sum(axis=-1, keepdims=True) + 1e-9)


def init_weights(seed=42):
    rng = np.random.default_rng(seed)
    def w(a, b): return rng.standard_normal((a, b)).astype(np.float32) * np.sqrt(2.0 / a)
    def z(n):   return np.zeros(n, dtype=np.float32)
    return {
        'W1': w(INPUT_SIZE, H1), 'b1': z(H1),
        'W2': w(H1, H2),         'b2': z(H2),
        'W3': w(H2, H3),         'b3': z(H3),
        'Wv': w(H3, 1),          'bv': z(1),   # value head
        'Wp': w(H3, N_ACTIONS),  'bp': z(N_ACTIONS),  # policy head
    }


def forward(w, X):
    """Batch forward pass.  X: (N, 34)
    Returns value (N,), policy_probs (N,9), h1,h2,h3 intermediates."""
    h1 = relu(X  @ w['W1'] + w['b1'])
    h2 = relu(h1 @ w['W2'] + w['b2'])
    h3 = relu(h2 @ w['W3'] + w['b3'])
    v  = np.tanh(h3 @ w['Wv'] + w['bv']).reshape(-1)    # (N,)
    logits = h3 @ w['Wp'] + w['bp']                       # (N, 9)
    probs  = softmax_fn(logits)                            # (N, 9)
    return v, probs, logits, h1, h2, h3


# ── Policy target: soft labels from alpha-beta ────────────────────────────────
def policy_target(state: TogyzkumalakState, time_budget: float = 0.15) -> np.ndarray:
    """Soft policy label: best move gets 1.0, others 0.0 (one-hot).
    Uses alpha-beta with given time budget."""
    moves = state.getPossibleMoves(state.currentPlayer)
    if not moves:
        return np.ones(N_ACTIONS, dtype=np.float32) / N_ACTIONS

    best = get_best_move_alphabeta(state, state.currentPlayer, time_budget)
    target = np.zeros(N_ACTIONS, dtype=np.float32)
    if best >= 0:
        target[best % 9] = 1.0
    else:
        # Fallback: uniform over legal moves
        for m in moves:
            target[m % 9] = 1.0
        target /= target.sum()
    return target


# ── Data generation ───────────────────────────────────────────────────────────
def sample_expert_positions(n: int = 5000) -> list:
    """Random rollouts + alpha-beta policy labels."""
    print(f"  Sampling {n} expert positions (alpha-beta policy labels)...", flush=True)
    samples = []
    while len(samples) < n:
        state = TogyzkumalakState()
        for _ in range(random.randint(2, 55)):
            if state.isGameOver: break
            moves = state.getPossibleMoves(state.currentPlayer)
            if not moves: break
            state.makeMove(random.choice(moves))
        if state.isGameOver:
            continue

        feat  = state_to_features(state)
        v_lbl = float(np.tanh(_evaluate_heuristic(state) / 40.0))
        p_lbl = policy_target(state, time_budget=0.12)
        samples.append((feat, p_lbl, v_lbl))

        if len(samples) % 200 == 0:
            print(f"    {len(samples)}/{n}", flush=True)

    print(f"  → {len(samples)} expert samples", flush=True)
    return samples


def generate_self_play_game(w_net, time_budget: float = 0.20, epsilon: float = 0.08) -> list:
    """One self-play game using the policy network + alpha-beta labels."""
    state   = TogyzkumalakState()
    samples = []

    while not state.isGameOver:
        moves = state.getPossibleMoves(state.currentPlayer)
        if not moves: break

        feat = state_to_features(state)

        # Choose move: policy network (stochastic) or random
        if random.random() < epsilon or w_net is None:
            move = random.choice(moves)
        else:
            x = feat.reshape(1, -1)
            _, probs, _, _, _, _ = forward(w_net, x)
            probs = probs[0].copy()
            # Mask illegal moves
            legal_idx = {m % 9 for m in moves}
            for i in range(N_ACTIONS):
                if i not in legal_idx:
                    probs[i] = 0.0
            s = probs.sum()
            if s < 1e-9:
                move = random.choice(moves)
            else:
                probs /= s
                chosen_idx = np.random.choice(N_ACTIONS, p=probs)
                legal_map  = {m % 9: m for m in moves}
                move = legal_map.get(chosen_idx, random.choice(moves))

        # Policy label from alpha-beta (independent of network)
        p_lbl = policy_target(state, time_budget=time_budget)
        samples.append((feat, p_lbl, state.currentPlayer))
        state.makeMove(move)

    # Label value with game outcome
    winner = state.winner
    labeled = []
    for feat, p_lbl, player in samples:
        if winner == player:
            v_lbl = 1.0
        elif winner == -1:
            v_lbl = 0.0
        else:
            v_lbl = -1.0
        labeled.append((feat, p_lbl, v_lbl))

    return labeled, winner


# ── Training ──────────────────────────────────────────────────────────────────
def train(w, samples, epochs: int = 100, batch: int = 256, lr: float = 0.001,
          value_coef: float = 1.0, policy_coef: float = 1.5, entropy_coef: float = 0.02):
    """Joint Adam training on value (MSE) + policy (cross-entropy) + entropy bonus."""
    random.shuffle(samples)
    X = np.array([s[0] for s in samples], dtype=np.float32)   # (N, 34)
    P = np.array([s[1] for s in samples], dtype=np.float32)   # (N, 9)  policy targets
    V = np.array([s[2] for s in samples], dtype=np.float32)   # (N,)    value targets

    N = len(X)
    m_adam = {k: np.zeros_like(v) for k, v in w.items()}
    v_adam = {k: np.zeros_like(v) for k, v in w.items()}
    t_adam = 0
    b1, b2, eps_adam = 0.9, 0.999, 1e-8

    for epoch in range(epochs):
        idx = np.random.permutation(N)
        total_vloss = total_ploss = n_batches = 0

        for start in range(0, N, batch):
            bi = idx[start:start + batch]
            xb, pb, vb = X[bi], P[bi], V[bi]
            n = len(xb)

            # ── Forward ──
            v_pred, p_pred, logits, h1, h2, h3 = forward(w, xb)

            # ── Value loss (MSE) ──
            v_diff = v_pred - vb                                 # (n,)
            vloss  = float(np.mean(v_diff ** 2))

            # ── Policy loss (cross-entropy) ──
            log_p  = np.log(p_pred + 1e-9)                      # (n, 9)
            ploss  = -float(np.mean(np.sum(pb * log_p, axis=1)))

            # ── Entropy bonus (encourage exploration) ──
            ent    = -np.sum(p_pred * log_p, axis=1).mean()

            total_vloss += vloss
            total_ploss += ploss
            n_batches   += 1

            # ── Backward through value head ──
            dv = (2.0 / n) * v_diff * (1.0 - v_pred ** 2) * value_coef  # (n,)

            # ── Backward through policy head ──
            # cross-entropy gradient w.r.t. logits = p_pred - p_target
            dp = (p_pred - pb) * policy_coef / n                          # (n, 9)
            # entropy gradient: -(log_p + 1) * entropy_coef
            dp += -(log_p + 1.0) * (-entropy_coef) / n

            # ── Gradients for heads ──
            grads = {}
            grads['Wv'] = h3.T @ dv.reshape(-1, 1)
            grads['bv'] = dv.sum(axis=0, keepdims=True).reshape(1)
            grads['Wp'] = h3.T @ dp
            grads['bp'] = dp.sum(axis=0)

            # ── Backward through shared trunk ──
            dh3 = (dv.reshape(-1, 1) @ w['Wv'].T + dp @ w['Wp'].T) * drelu(h3)
            grads['W3'] = h2.T @ dh3
            grads['b3'] = dh3.sum(axis=0)

            dh2 = (dh3 @ w['W3'].T) * drelu(h2)
            grads['W2'] = h1.T @ dh2
            grads['b2'] = dh2.sum(axis=0)

            dh1 = (dh2 @ w['W2'].T) * drelu(h1)
            grads['W1'] = xb.T @ dh1
            grads['b1'] = dh1.sum(axis=0)

            # ── Adam update ──
            t_adam += 1
            for k in w:
                m_adam[k] = b1 * m_adam[k] + (1 - b1) * grads[k]
                v_adam[k] = b2 * v_adam[k] + (1 - b2) * grads[k] ** 2
                mh = m_adam[k] / (1 - b1 ** t_adam)
                vh = v_adam[k] / (1 - b2 ** t_adam)
                w[k] -= lr * mh / (np.sqrt(vh) + eps_adam)

        if (epoch + 1) % 20 == 0:
            print(f"    epoch {epoch+1:3d}/{epochs}  "
                  f"value_loss={total_vloss/n_batches:.4f}  "
                  f"policy_loss={total_ploss/n_batches:.4f}", flush=True)

    return w


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    n_games   = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    t_start   = time.time()

    print("=" * 60)
    print("  Togyzkumalak  Policy+Value Network  (AlphaZero-lite)")
    print(f"  Architecture: {INPUT_SIZE}→{H1}→{H2}→{H3}→{{value(1), policy(9)}}")
    print("=" * 60)

    # Phase 1 — expert positions
    print("\n[Phase 1/3] Expert positions (alpha-beta policy labels)")
    samples = sample_expert_positions(n=5000)

    # Phase 2 — self-play
    print(f"\n[Phase 2/3] Self-play ({n_games} games, 0.20 s/move alpha-beta labels)")
    w_net = None   # first iteration: no network, use pure alpha-beta
    wins  = {0: 0, 1: 0, -1: 0}

    for i in range(n_games):
        t0 = time.time()
        game_samples, winner = generate_self_play_game(w_net, time_budget=0.20)
        wins[winner] += 1
        samples.extend(game_samples)
        print(f"  game {i+1:3d}/{n_games}  pos={len(game_samples):3d}  "
              f"winner={winner}  ({time.time()-t0:.1f}s)", flush=True)

        # Every 50 games, do a quick mid-training to update the policy network
        # so later games use a smarter policy
        if (i + 1) % 50 == 0 and i < n_games - 1:
            print(f"  [mid-training at game {i+1}]", flush=True)
            w_net = init_weights(seed=i) if w_net is None else w_net
            w_net = train(w_net, samples[-5000:], epochs=30, lr=0.001)

    print(f"\nTotal samples: {len(samples)}")
    print(f"Wins: P0={wins[0]}  P1={wins[1]}  Draw={wins[-1]}")

    # Phase 3 — full training
    print("\n[Phase 3/3] Full joint training (100 epochs)")
    w_final = init_weights(seed=42) if w_net is None else w_net
    w_final = train(w_final, samples, epochs=100, lr=0.0005)

    # Save (same file — ai.py will detect policy head via 'Wp' key)
    np.savez(WEIGHTS_PATH, **w_final)
    print(f"\nSaved → {WEIGHTS_PATH}")
    print(f"Total time: {(time.time()-t_start)/60:.1f} min")
    print("\nAI now uses Policy+Value network (AlphaZero-lite).")
