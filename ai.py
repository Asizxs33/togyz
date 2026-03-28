import math
import random
import time
import os
from game_logic import TogyzkumalakState

# ─────────────────────────────────────────────────────────────────────────────
# NEURAL NETWORK — loaded once at startup if model_weights.npz exists
# ─────────────────────────────────────────────────────────────────────────────

_nn = None   # dict of numpy weight arrays, or None

def _load_nn():
    global _nn
    path = os.path.join(os.path.dirname(__file__), 'model_weights.npz')
    if not os.path.exists(path):
        return
    try:
        import numpy as np
        data = np.load(path)
        _nn = {k: data[k] for k in data.files}
        print(f"[AI] Neural network loaded ({path})", flush=True)
    except Exception as e:
        print(f"[AI] Could not load neural network: {e}", flush=True)

_load_nn()


def _nn_features(state: TogyzkumalakState):
    """25-dim feature vector from current player's perspective."""
    import numpy as np
    cur, opp = state.currentPlayer, 1 - state.currentPlayer
    cs, os_ = cur * 9, opp * 9
    f = np.empty(25, dtype=np.float32)
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
    return f


def _nn_eval(state: TogyzkumalakState) -> float:
    """Returns value in [-1, 1] from current player's perspective."""
    import numpy as np
    w = _nn
    x = _nn_features(state)
    h1  = np.maximum(0.0, x  @ w['W1'] + w['b1'])
    h2  = np.maximum(0.0, h1 @ w['W2'] + w['b2'])
    out = float(np.tanh(h2 @ w['W3'] + w['b3']))
    return out

# ─────────────────────────────────────────────────────────────────────────────
# CORE HELPER: simulate sowing without modifying state
# Returns (last_board_pocket, donated_to_p0, donated_to_p1)
# last_board_pocket = -1 if last stone was absorbed by a tuzdyk
# ─────────────────────────────────────────────────────────────────────────────

def _simulate_sow(board, tuzdyks, m):
    stones = board[m]
    if stones == 0:
        return -1, 0, 0

    tuz0, tuz1 = tuzdyks[0], tuzdyks[1]
    don = [0, 0]

    if stones == 1:
        idx = (m + 1) % 18
        if idx == tuz0:
            don[0] += 1
            return -1, don[0], don[1]
        if idx == tuz1:
            don[1] += 1
            return -1, don[0], don[1]
        return idx, 0, 0

    # Multi-stone: first stone stays at m (m is never a tuzdyk since player owns it)
    idx = m
    last_board = m
    remaining = stones - 1
    while remaining > 0:
        idx = (idx + 1) % 18
        if idx == tuz0:
            don[0] += 1
        elif idx == tuz1:
            don[1] += 1
        else:
            last_board = idx
        remaining -= 1

    return last_board, don[0], don[1]


# ─────────────────────────────────────────────────────────────────────────────
# EVALUATION
# Returns score from state.currentPlayer's perspective. Higher = better.
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(state: TogyzkumalakState) -> float:
    # If neural network is loaded, blend NN + heuristic
    # (blend keeps tactical sharpness from heuristic while NN improves strategic play)
    if _nn is not None:
        nn_val  = _nn_eval(state)           # [-1, 1]
        hval    = _evaluate_heuristic(state)
        # Scale NN to same range as heuristic, then blend 60/40
        return 0.6 * nn_val * 120.0 + 0.4 * hval

    return _evaluate_heuristic(state)


def _evaluate_heuristic(state: TogyzkumalakState) -> float:
    cur = state.currentPlayer
    opp = 1 - cur
    cs = cur * 9
    os = opp * 9

    # 1. Kazan difference — primary
    ev = (state.kazans[cur] - state.kazans[opp]) * 3.0

    # 2. Tuzdyk advantage
    # A tuzdyk typically collects 5-15 stones over its life → worth ~25-35 eval pts
    my_tuz = state.tuzdyks[cur] != -1
    opp_tuz = state.tuzdyks[opp] != -1
    ev += (my_tuz - opp_tuz) * 30.0

    # 3. Tuzdyk DANGER: how many stones we'd donate to opponent's tuzdyk
    # if we sowed from each of our current pockets (discourages keeping large
    # piles that pass through opponent's tuzdyk)
    opp_tuz_pos = state.tuzdyks[opp]
    if opp_tuz_pos != -1:
        for i in range(cs, cs + 9):
            if state.board[i] > 0:
                _, d0, d1 = _simulate_sow(state.board, state.tuzdyks, i)
                donated = d0 if opp == 0 else d1
                if donated > 0:
                    ev -= donated * 4.0   # each forced-donated stone = -4 pts

    # 4. Immediate capture opportunities (opponent's even pockets)
    for i in range(os, os + 9):
        v = state.board[i]
        if v > 0 and v % 2 == 0:
            ev += v * 0.6
    # Penalty: our own even pockets are capture bait
    for i in range(cs, cs + 9):
        v = state.board[i]
        if v > 0 and v % 2 == 0:
            ev -= v * 0.6

    # 5. Tuzdyk creation readiness: opponent pocket with 2 stones = one step away
    if not my_tuz:
        opp_tuz_idx = state.tuzdyks[opp]
        for i in range(os, os + 9):
            if state.board[i] == 2 and i not in (8, 17):
                if opp_tuz_idx == -1 or opp_tuz_idx % 9 != i % 9:
                    ev += 10.0  # increased from 6 → 10

    # 6. Stone mobility (tie-breaker)
    ev += (sum(state.board[cs:cs+9]) - sum(state.board[os:os+9])) * 0.1

    return ev


# ─────────────────────────────────────────────────────────────────────────────
# MOVE ORDERING — critical for alpha-beta speed and quality
# ─────────────────────────────────────────────────────────────────────────────

def _is_loud_move(board, tuzdyks, cur_player, m):
    """Returns (is_capture, is_tuzdyk_creation) for a given move."""
    opp = 1 - cur_player
    os = opp * 9
    land, _, _ = _simulate_sow(board, tuzdyks, m)
    if land < 0 or land in tuzdyks:
        return False, False
    if not (os <= land < os + 9):
        return False, False
    fut = board[land] + 1
    is_cap = (fut % 2 == 0)
    is_tuz = (fut == 3 and tuzdyks[cur_player] == -1
              and land not in (8, 17)
              and (tuzdyks[opp] == -1 or tuzdyks[opp] % 9 != land % 9))
    return is_cap, is_tuz


def order_moves(state: TogyzkumalakState, moves: list) -> list:
    opp = 1 - state.currentPlayer
    cur = state.currentPlayer

    def priority(m: int) -> int:
        stones = state.board[m]
        if stones == 0:
            return -99999

        land, d0, d1 = _simulate_sow(state.board, state.tuzdyks, m)

        # CRITICAL: heavily penalize donating stones to opponent's tuzdyk
        donated_to_opp = d0 if opp == 0 else d1
        if donated_to_opp > 0:
            return -donated_to_opp * 50   # push these to the very end

        if land < 0 or land in state.tuzdyks:
            return 0

        os = opp * 9
        if os <= land < os + 9:
            fut = state.board[land] + 1
            if fut % 2 == 0:
                return fut * 10            # capture: more stones = higher priority
            if (fut == 3
                    and state.tuzdyks[cur] == -1
                    and land not in (8, 17)
                    and (state.tuzdyks[opp] == -1
                         or state.tuzdyks[opp] % 9 != land % 9)):
                return 100                 # tuzdyk creation = highest non-capture priority

        return 0

    return sorted(moves, key=priority, reverse=True)


# ─────────────────────────────────────────────────────────────────────────────
# QUIESCENCE SEARCH
# When depth=0, keep searching captures and tuzdyk moves to avoid
# "horizon effect" (AI stops just before losing/gaining big material)
# ─────────────────────────────────────────────────────────────────────────────

def _quiescence(state: TogyzkumalakState, alpha: float, beta: float,
                deadline: float, qdepth: int = 4) -> float:
    if time.time() > deadline:
        raise _Timeout()

    stand_pat = evaluate(state)
    if stand_pat >= beta:
        return beta
    if stand_pat > alpha:
        alpha = stand_pat
    if qdepth == 0:
        return alpha

    cur = state.currentPlayer
    moves = state.getPossibleMoves(cur)

    # Only examine "loud" moves: captures and tuzdyk creation
    loud = []
    for m in moves:
        is_cap, is_tuz = _is_loud_move(state.board, state.tuzdyks, cur, m)
        if is_cap or is_tuz:
            loud.append((m, is_tuz))

    # Tuzdyk creation first, then captures by value
    loud.sort(key=lambda x: (x[1], state.board[x[0]]), reverse=True)

    for m, _ in loud:
        child = state.clone()
        child.makeMove(m)
        score = -_quiescence(child, -beta, -alpha, deadline, qdepth - 1)
        if score >= beta:
            return beta
        if score > alpha:
            alpha = score

    return alpha


# ─────────────────────────────────────────────────────────────────────────────
# ALPHA-BETA NEGAMAX with Iterative Deepening
# ─────────────────────────────────────────────────────────────────────────────

class _Timeout(Exception):
    pass


def _negamax(state: TogyzkumalakState, depth: int,
             alpha: float, beta: float, deadline: float) -> float:
    if time.time() > deadline:
        raise _Timeout()

    if state.isGameOver:
        if state.winner == state.currentPlayer:
            return 10000.0 + depth
        elif state.winner == -1:
            return 0.0
        else:
            return -(10000.0 + depth)

    if depth == 0:
        # Quiescence: don't stop at a tactically volatile position
        return _quiescence(state, alpha, beta, deadline)

    moves = state.getPossibleMoves(state.currentPlayer)
    if not moves:
        return -(10000.0 + depth)

    moves = order_moves(state, moves)
    best = float('-inf')

    for move in moves:
        child = state.clone()
        child.makeMove(move)
        score = -_negamax(child, depth - 1, -beta, -alpha, deadline)
        if score > best:
            best = score
        if score > alpha:
            alpha = score
        if alpha >= beta:
            break

    return best


def get_best_move_alphabeta(root_state: TogyzkumalakState, root_player: int,
                             max_time_seconds: float = 3.0) -> int:
    moves = root_state.getPossibleMoves(root_player)
    if not moves:
        return -1
    if len(moves) == 1:
        return moves[0]

    moves = order_moves(root_state, moves)
    best_move = moves[0]
    deadline = time.time() + max_time_seconds * 0.95

    for depth in range(1, 30):
        try:
            best_score = float('-inf')
            alpha = float('-inf')
            current_best = moves[0]

            for move in moves:
                child = root_state.clone()
                child.makeMove(move)
                score = -_negamax(child, depth - 1, -float('inf'), -alpha, deadline)
                if score > best_score:
                    best_score = score
                    current_best = move
                if score > alpha:
                    alpha = score

            best_move = current_best
            moves = [best_move] + [m for m in moves if m != best_move]

            if abs(best_score) > 9000:
                break

        except _Timeout:
            break

    return best_move


# ─────────────────────────────────────────────────────────────────────────────
# MCTS (kept for fast hint calls < 1 second)
# ─────────────────────────────────────────────────────────────────────────────

class _MCTSNode:
    __slots__ = ('state', 'move', 'parent', 'children', 'wins', 'visits', 'untriedMoves')

    def __init__(self, state: TogyzkumalakState, move=None, parent=None):
        self.state = state
        self.move = move
        self.parent = parent
        self.children = []
        self.wins = 0.0
        self.visits = 0
        self.untriedMoves = state.getPossibleMoves(state.currentPlayer)

    def best_child(self, c=1.41):
        best_score = float('-inf')
        best = None
        log_v = math.log(self.visits)
        for child in self.children:
            score = child.wins / child.visits + c * math.sqrt(log_v / child.visits)
            if score > best_score:
                best_score = score
                best = child
        return best


def _mcts(root_state: TogyzkumalakState, root_player: int,
          iterations: int, max_time_seconds: float) -> int:
    root = _MCTSNode(root_state)
    deadline = time.time() + max_time_seconds

    for _ in range(iterations):
        if time.time() > deadline:
            break

        node = root
        state = root_state.clone()

        # Selection
        while not node.untriedMoves and node.children:
            node = node.best_child()
            state.makeMove(node.move)

        # Expansion
        if node.untriedMoves:
            move = random.choice(node.untriedMoves)
            node.untriedMoves.remove(move)
            state.makeMove(move)
            child = _MCTSNode(state.clone(), move, node)
            node.children.append(child)
            node = child

        # Simulation with tuzdyk-aware heuristic
        sim = state.clone()
        for _ in range(150):
            if sim.isGameOver:
                break
            sim_moves = sim.getPossibleMoves(sim.currentPlayer)
            if not sim_moves:
                break
            opp_p = 1 - sim.currentPlayer
            opp_tuz_pos = sim.tuzdyks[opp_p]

            chosen = None
            if random.random() < 0.85:
                best_m, best_score = None, -999
                os_s = opp_p * 9
                for mv in sim_moves:
                    land, d0, d1 = _simulate_sow(sim.board, sim.tuzdyks, mv)
                    donated = d0 if opp_p == 0 else d1
                    if donated > 0:
                        continue  # never donate to opponent tuzdyk
                    if land < 0 or land in sim.tuzdyks:
                        continue
                    sc = 0
                    if os_s <= land < os_s + 9:
                        fut = sim.board[land] + 1
                        if fut % 2 == 0:
                            sc = fut * 10
                        elif fut == 3 and sim.tuzdyks[sim.currentPlayer] == -1:
                            sc = 80
                    if sc > best_score:
                        best_score = sc
                        best_m = mv
                # If any non-donating move exists, prefer best; else allow donating
                if best_m is not None:
                    chosen = best_m
                elif best_score == 0:
                    # Pick any non-donating move randomly
                    safe = [mv for mv in sim_moves
                            if ((_simulate_sow(sim.board, sim.tuzdyks, mv)[1 if opp_p == 0 else 2]) == 0
                                if opp_tuz_pos != -1 else True)]
                    chosen = random.choice(safe) if safe else random.choice(sim_moves)
            if chosen is None:
                chosen = random.choice(sim_moves)
            sim.makeMove(chosen)

        # Evaluate
        if sim.isGameOver:
            result = 1.0 if sim.winner == root_player else (0.5 if sim.winner == -1 else 0.0)
        else:
            my = sim.kazans[root_player]
            op = sim.kazans[1 - root_player]
            diff = (my - op) * 2.0
            diff += (int(sim.tuzdyks[root_player] != -1) - int(sim.tuzdyks[1 - root_player] != -1)) * 25
            result = 1.0 / (1.0 + math.exp(-diff / 15.0))

        # Backpropagation (player-aware)
        bp = node
        while bp is not None:
            bp.visits += 1
            mover = 1 - bp.state.currentPlayer
            bp.wins += result if mover == root_player else (1.0 - result)
            bp = bp.parent

    if not root.children:
        return -1
    return max(root.children, key=lambda c: c.visits).move


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────

OPENING_BOOK = {
    (162, 0): [4, 6, 8],
    (162, 1): [9, 11, 13],
}


def get_best_move_mcts(root_state: TogyzkumalakState, root_player: int,
                        iterations: int = 20000,
                        max_time_seconds: float = 3.0) -> int:
    key = (sum(root_state.board), root_state.currentPlayer)
    if key in OPENING_BOOK:
        return random.choice(OPENING_BOOK[key])

    if max_time_seconds < 1.0:
        return _mcts(root_state, root_player, iterations, max_time_seconds)

    return get_best_move_alphabeta(root_state, root_player, max_time_seconds)
