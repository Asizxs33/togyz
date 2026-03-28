"""
Togyzkumalak  —  Policy + Value Network  (AlphaZero-lite, v2)
================================================================
Architecture: 34 → 256 → 128 → 64 → {value(1 tanh), policy(9 softmax)}

Training strategy (stable two-phase):
  Phase 1  — Train VALUE head with MSE (same proven approach as train_ai.py)
  Phase 2  — Freeze backbone, train POLICY head with cross-entropy only
  Both heads share the same backbone weights.

Usage:
    python3 train_rl.py          # 200 self-play games (~1.5 h)
    python3 train_rl.py 100      # faster test (~50 min)
"""

import numpy as np
import random, time, os, sys

from game_logic import TogyzkumalakState
from ai import get_best_move_alphabeta, _evaluate_heuristic

# ── Config ────────────────────────────────────────────────────────────────────
INPUT_SIZE  = 34
H1, H2, H3  = 256, 128, 64
N_ACTIONS   = 9
WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), 'model_weights.npz')


# ── Features (same as train_ai.py v2) ─────────────────────────────────────────
def state_to_features(state: TogyzkumalakState) -> np.ndarray:
    f = np.empty(INPUT_SIZE, dtype=np.float32)
    cur = state.currentPlayer;  opp = 1 - cur
    cs,  os_ = cur * 9, opp * 9

    for i in range(9):
        f[i]     = state.board[cs  + i] / 20.0
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


# ── Network helpers ───────────────────────────────────────────────────────────
def relu(x):  return np.maximum(0.0, x)
def drelu(x): return (x > 0).astype(np.float32)

def softmax_fn(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)

def init_weights(seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    w = lambda a, b: (rng.standard_normal((a, b)) * np.sqrt(2.0 / a)).astype(np.float32)
    z = lambda n: np.zeros(n, dtype=np.float32)
    return {
        'W1': w(INPUT_SIZE, H1), 'b1': z(H1),
        'W2': w(H1, H2),         'b2': z(H2),
        'W3': w(H2, H3),         'b3': z(H3),
        'Wv': w(H3, 1),          'bv': z(1),          # value head
        'Wp': w(H3, N_ACTIONS),  'bp': z(N_ACTIONS),  # policy head
    }

def forward_value(w: dict, X: np.ndarray):
    """Returns (v_pred, h1, h2, h3).  X: (N, INPUT_SIZE)"""
    h1 = relu(X  @ w['W1'] + w['b1'])
    h2 = relu(h1 @ w['W2'] + w['b2'])
    h3 = relu(h2 @ w['W3'] + w['b3'])
    v  = np.tanh(h3 @ w['Wv'] + w['bv']).reshape(-1)
    return v, h1, h2, h3

def forward_policy(w: dict, X: np.ndarray):
    """Returns (probs, h3).  X: (N, INPUT_SIZE)"""
    h1 = relu(X  @ w['W1'] + w['b1'])
    h2 = relu(h1 @ w['W2'] + w['b2'])
    h3 = relu(h2 @ w['W3'] + w['b3'])
    return softmax_fn(h3 @ w['Wp'] + w['bp']), h3


# ── Data generation ───────────────────────────────────────────────────────────
def sample_positions(n: int = 6000) -> list:
    """Random rollouts → value label (heuristic) + policy label (alpha-beta)."""
    print(f"  Sampling {n} positions...", flush=True)
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
        # Policy: alpha-beta best move → one-hot
        best  = get_best_move_alphabeta(state, state.currentPlayer, max_time_seconds=0.12)
        p_lbl = np.zeros(N_ACTIONS, dtype=np.float32)
        if best >= 0:
            p_lbl[best % 9] = 1.0
        else:
            legal = state.getPossibleMoves(state.currentPlayer)
            for m in legal: p_lbl[m % 9] = 1.0
            p_lbl /= max(p_lbl.sum(), 1e-9)
        samples.append((feat, v_lbl, p_lbl))
        if len(samples) % 500 == 0:
            print(f"    {len(samples)}/{n}", flush=True)
    print(f"  → {len(samples)} samples", flush=True)
    return samples


def generate_self_play_game(time_budget: float = 0.20, epsilon: float = 0.08) -> list:
    """One alpha-beta self-play game.  Returns list of (feat, v_lbl, p_lbl)."""
    state   = TogyzkumalakState()
    traj    = []   # (feat, player, p_lbl)

    while not state.isGameOver:
        moves = state.getPossibleMoves(state.currentPlayer)
        if not moves: break
        feat  = state_to_features(state)
        best  = get_best_move_alphabeta(state, state.currentPlayer, time_budget)
        p_lbl = np.zeros(N_ACTIONS, dtype=np.float32)
        if best >= 0:
            p_lbl[best % 9] = 1.0
            move = best if random.random() > epsilon else random.choice(moves)
        else:
            for m in moves: p_lbl[m % 9] = 1.0
            p_lbl /= max(p_lbl.sum(), 1e-9)
            move = random.choice(moves)
        traj.append((feat, state.currentPlayer, p_lbl))
        state.makeMove(move)

    winner = state.winner
    out = []
    for feat, player, p_lbl in traj:
        v_lbl = 1.0 if winner == player else (0.0 if winner == -1 else -1.0)
        out.append((feat, v_lbl, p_lbl))
    return out, winner


# ── Phase A: Train VALUE backbone + head  (proven Adam / MSE) ─────────────────
def train_value(w: dict, samples: list,
                epochs: int = 100, batch: int = 256, lr: float = 5e-4) -> dict:
    random.shuffle(samples)
    X = np.array([s[0] for s in samples], dtype=np.float32)
    y = np.array([s[1] for s in samples], dtype=np.float32).reshape(-1, 1)
    N = len(X)

    mk = {k: np.zeros_like(v) for k, v in w.items()}
    vk = {k: np.zeros_like(v) for k, v in w.items()}
    t  = 0
    b1, b2, eps = 0.9, 0.999, 1e-8

    for epoch in range(epochs):
        idx  = np.random.permutation(N)
        lsum = 0.0; nb = 0
        for start in range(0, N, batch):
            bi      = idx[start:start+batch]
            xb, yb  = X[bi], y[bi]; n = len(xb)

            # Forward
            out, h1, h2, h3 = forward_value(w, xb)
            out = out.reshape(-1, 1)
            diff = out - yb
            lsum += float(np.mean(diff**2)); nb += 1

            # Backprop (MSE + tanh)
            d4  = (2.0/n) * diff * (1.0 - out**2)
            gWv = h3.T @ d4;  gbv = d4.sum(axis=0)
            dh3 = (d4 @ w['Wv'].T) * drelu(h3)
            gW3 = h2.T @ dh3; gb3 = dh3.sum(axis=0)
            dh2 = (dh3 @ w['W3'].T) * drelu(h2)
            gW2 = h1.T @ dh2; gb2 = dh2.sum(axis=0)
            dh1 = (dh2 @ w['W2'].T) * drelu(h1)
            gW1 = xb.T @ dh1; gb1 = dh1.sum(axis=0)

            grads = dict(W1=gW1,b1=gb1, W2=gW2,b2=gb2, W3=gW3,b3=gb3,
                         Wv=gWv,bv=gbv.reshape(1),
                         Wp=np.zeros_like(w['Wp']), bp=np.zeros_like(w['bp']))

            # Adam
            t += 1
            for k in w:
                mk[k] = b1*mk[k] + (1-b1)*grads[k]
                vk[k] = b2*vk[k] + (1-b2)*grads[k]**2
                mh = mk[k]/(1-b1**t);  vh = vk[k]/(1-b2**t)
                w[k] -= lr * mh / (np.sqrt(vh)+eps)

        if (epoch+1) % 20 == 0:
            print(f"    [value] epoch {epoch+1:3d}/{epochs}  loss={lsum/nb:.4f}", flush=True)
    return w


# ── Phase B: Train POLICY head only (backbone frozen) ─────────────────────────
def train_policy(w: dict, samples: list,
                 epochs: int = 60, batch: int = 256, lr: float = 2e-4) -> dict:
    """Cross-entropy training; only Wp and bp are updated."""
    random.shuffle(samples)
    X = np.array([s[0] for s in samples], dtype=np.float32)
    P = np.array([s[2] for s in samples], dtype=np.float32)   # (N, 9) one-hot
    N = len(X)

    # Adam state for policy head only
    mp = {k: np.zeros_like(w[k]) for k in ('Wp', 'bp')}
    vp = {k: np.zeros_like(w[k]) for k in ('Wp', 'bp')}
    t  = 0
    b1, b2, eps = 0.9, 0.999, 1e-8

    for epoch in range(epochs):
        idx  = np.random.permutation(N)
        lsum = 0.0; nb = 0
        for start in range(0, N, batch):
            bi     = idx[start:start+batch]
            xb, pb = X[bi], P[bi]; n = len(xb)

            # Forward (backbone frozen — no grad through W1/W2/W3)
            probs, h3 = forward_policy(w, xb)

            # Cross-entropy loss
            log_p = np.log(probs + 1e-9)
            loss  = -float(np.mean(np.sum(pb * log_p, axis=1)))
            lsum += loss; nb += 1

            # Gradient w.r.t. logits = (probs - target) / n
            dlogits = (probs - pb) / n              # (n, 9)

            gWp = h3.T @ dlogits                    # (H3, 9)
            gbp = dlogits.sum(axis=0)               # (9,)

            # Gradient clipping (per tensor)
            for g in (gWp, gbp):
                norm = np.linalg.norm(g)
                if norm > 1.0: g *= (1.0 / norm)

            grads = {'Wp': gWp, 'bp': gbp}
            t += 1
            for k in ('Wp', 'bp'):
                mp[k] = b1*mp[k] + (1-b1)*grads[k]
                vp[k] = b2*vp[k] + (1-b2)*grads[k]**2
                mh = mp[k]/(1-b1**t);  vh = vp[k]/(1-b2**t)
                w[k] -= lr * mh / (np.sqrt(vh)+eps)

        if (epoch+1) % 20 == 0:
            print(f"    [policy] epoch {epoch+1:3d}/{epochs}  "
                  f"CE_loss={lsum/nb:.4f}  (target ~{np.log(N_ACTIONS):.2f})", flush=True)
    return w


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    n_games = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    t_start = time.time()

    print("=" * 60)
    print("  Togyzkumalak  Policy + Value  (AlphaZero-lite v2)")
    print(f"  {INPUT_SIZE}→{H1}→{H2}→{H3}→{{value(1), policy(9)}}")
    print("=" * 60)

    # Phase 1 — diverse positions
    print("\n[Phase 1/4] Sampling positions (value + policy labels)")
    samples = sample_positions(n=6000)

    # Phase 2 — self-play
    print(f"\n[Phase 2/4] Self-play ({n_games} games, {0.20}s/move)")
    wins = {0: 0, 1: 0, -1: 0}
    for i in range(n_games):
        t0 = time.time()
        gs, winner = generate_self_play_game(time_budget=0.20)
        wins[winner] += 1
        samples.extend(gs)
        print(f"  game {i+1:3d}/{n_games}  pos={len(gs):3d}  "
              f"winner={winner}  ({time.time()-t0:.1f}s)", flush=True)

    print(f"\nTotal samples: {len(samples)}")
    print(f"Wins: P0={wins[0]}  P1={wins[1]}  Draw={wins[-1]}")

    # Phase 3 — train value (backbone + value head)
    print("\n[Phase 3/4] Training VALUE network (100 epochs, MSE)")
    w = init_weights(seed=42)
    w = train_value(w, samples, epochs=100, lr=5e-4)

    # Phase 4 — train policy (head only, backbone frozen)
    print("\n[Phase 4/4] Training POLICY head (60 epochs, cross-entropy)")
    w = train_policy(w, samples, epochs=60, lr=2e-4)

    np.savez(WEIGHTS_PATH, **w)
    print(f"\nSaved → {WEIGHTS_PATH}")
    print(f"Total time: {(time.time()-t_start)/60:.1f} min")
    print("AI now uses Policy+Value network.")
