import math
import random
import time
import os
import json
from game_logic import TogyzkumalakState

try:
    import psycopg
except ImportError:
    psycopg = None

# Load .env for OPENAI_API_KEY
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

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
        weights = {k: data[k] for k in data.files}
        # Check architecture compatibility: W1 must accept 34-dim input
        if weights['W1'].shape[0] != 34:
            print("[AI] Weights are v1 (25-dim), skipping NN - retrain with train_ai.py", flush=True)
            return
        _nn = weights
        arch = f"{weights['W1'].shape[0]}->{weights['W1'].shape[1]}->{weights['W2'].shape[1]}"
        if 'W4' in weights:
            arch += f"->{weights['W3'].shape[1]}->1"
        else:
            arch += "->1"
        print(f"[AI] Neural network loaded: {arch}", flush=True)
    except Exception as e:
        print(f"[AI] Could not load neural network: {e}", flush=True)

_load_nn()

# ─────────────────────────────────────────────────────────────────────────────
# TRANSPOSITION TABLE — cache previously evaluated positions
# Key: compact board state tuple.  Value: (depth, score, flag, best_move)
# ─────────────────────────────────────────────────────────────────────────────
_TT: dict = {}
_TT_EXACT, _TT_LOWER, _TT_UPPER = 0, 1, 2
_TT_MAX = 300_000   # max entries; evict by clearing when full
DATABASE_URL = os.environ.get("DATABASE_URL")
LEARNING_FILE = os.environ.get("TOGYZ_LEARNING_FILE", os.path.join(os.path.dirname(__file__), "ai_learning.json"))
LEARNING_STATS = None
LEARNING_DB_READY = False

def _tt_key(state: TogyzkumalakState):
    return (tuple(state.board),
            state.kazans[0], state.kazans[1],
            state.tuzdyks[0], state.tuzdyks[1],
            state.currentPlayer)


def _state_key(state: TogyzkumalakState) -> str:
    return "|".join((
        ",".join(map(str, state.board)),
        ",".join(map(str, state.kazans)),
        ",".join(map(str, state.tuzdyks)),
        str(state.currentPlayer),
    ))


def _learning_score(visits: int, value: float) -> float:
    visits = max(1, visits)
    average = value / visits
    confidence = min(1.0, visits / 12)
    return max(-18.0, min(18.0, average * confidence * 18))


def _db_enabled() -> bool:
    return bool(DATABASE_URL and psycopg)


def _ensure_learning_table(conn) -> None:
    global LEARNING_DB_READY
    if LEARNING_DB_READY:
        return

    with conn.cursor() as cur:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS ai_learning (
                state_key TEXT NOT NULL,
                move INTEGER NOT NULL,
                visits INTEGER NOT NULL DEFAULT 0,
                value DOUBLE PRECISION NOT NULL DEFAULT 0,
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                PRIMARY KEY (state_key, move)
            )
            """
        )
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_ai_learning_updated_at
            ON ai_learning(updated_at DESC)
            """
        )
    LEARNING_DB_READY = True


def _db_learning_bias(state: TogyzkumalakState, move: int):
    if not _db_enabled():
        return None

    try:
        with psycopg.connect(DATABASE_URL) as conn:
            _ensure_learning_table(conn)
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT visits, value FROM ai_learning WHERE state_key = %s AND move = %s",
                    (_state_key(state), move),
                )
                row = cur.fetchone()
    except Exception as e:
        print(f"[AI] learning DB read fallback: {type(e).__name__}", flush=True)
        return None

    if not row:
        return 0.0
    return _learning_score(row[0], row[1])


def _db_record_learning(updates) -> int:
    if not updates or not _db_enabled():
        return 0

    try:
        with psycopg.connect(DATABASE_URL) as conn:
            _ensure_learning_table(conn)
            with conn.cursor() as cur:
                cur.executemany(
                    """
                    INSERT INTO ai_learning (state_key, move, visits, value)
                    VALUES (%s, %s, 1, %s)
                    ON CONFLICT (state_key, move)
                    DO UPDATE SET
                        visits = ai_learning.visits + 1,
                        value = ai_learning.value + EXCLUDED.value,
                        updated_at = NOW()
                    """,
                    updates,
                )
    except Exception as e:
        print(f"[AI] learning DB write fallback: {type(e).__name__}", flush=True)
        return 0

    return len(updates)


def _load_learning() -> dict:
    global LEARNING_STATS
    if LEARNING_STATS is not None:
        return LEARNING_STATS

    try:
        with open(LEARNING_FILE, "r", encoding="utf-8") as fh:
            data = json.load(fh)
            LEARNING_STATS = data if isinstance(data, dict) else {}
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        LEARNING_STATS = {}

    return LEARNING_STATS


def _save_learning() -> None:
    if LEARNING_STATS is None:
        return

    tmp_file = f"{LEARNING_FILE}.tmp"
    with open(tmp_file, "w", encoding="utf-8") as fh:
        json.dump(LEARNING_STATS, fh, ensure_ascii=False, separators=(",", ":"))
    os.replace(tmp_file, LEARNING_FILE)


def _learning_bias(state: TogyzkumalakState, move: int) -> float:
    db_bias = _db_learning_bias(state, move)
    if db_bias is not None:
        return db_bias

    stats = _load_learning().get(f"{_state_key(state)}->{move}")
    if not stats:
        return 0.0

    return _learning_score(stats.get("visits", 0), stats.get("value", 0.0))


def record_learning(samples, winner, ai_player=1):
    if winner == ai_player:
        result = 1.0
    elif winner == -1 or winner is None:
        result = 0.15
    else:
        result = -1.0

    updates = []
    for sample in samples:
        if sample.get("player") != ai_player:
            continue

        board = sample.get("board")
        kazans = sample.get("kazans")
        tuzdyks = sample.get("tuzdyks")
        move = sample.get("move")
        if not isinstance(board, list) or not isinstance(kazans, list) or not isinstance(tuzdyks, list):
            continue
        if not isinstance(move, int):
            continue

        state = TogyzkumalakState()
        state.board = board[:18]
        state.kazans = kazans[:2]
        state.tuzdyks = [(t if t is not None else -1) for t in tuzdyks[:2]]
        state.currentPlayer = sample.get("player", ai_player)

        updates.append((_state_key(state), move, result))

    learned = _db_record_learning(updates)
    if learned:
        return learned

    stats = _load_learning()
    for state_key, move, value in updates:
        key = f"{state_key}->{move}"
        entry = stats.setdefault(key, {"visits": 0, "value": 0.0})
        entry["visits"] += 1
        entry["value"] += value

    if updates:
        _save_learning()

    return len(updates)


def get_ai_status():
    return {
        "modelLoaded": _nn is not None,
        "modelKeys": sorted(_nn.keys()) if _nn is not None else [],
        "learningBackend": "postgres" if _db_enabled() else "json",
        "databaseConfigured": bool(DATABASE_URL),
        "postgresDriverLoaded": psycopg is not None,
        "gymEnvironment": "togyzkumalak_gym_env.TogyzkumalakGymEnv",
    }

# Killer moves: 2 per ply (distance from root), up to ply 32
_killers: list = [[None, None] for _ in range(32)]


def _nn_features(state: TogyzkumalakState):
    """34-dim feature vector from current player's perspective (v2)."""
    import numpy as np
    cur, opp = state.currentPlayer, 1 - state.currentPlayer
    cs, os_ = cur * 9, opp * 9
    f = np.empty(34, dtype=np.float32)

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

    f[25] = sum(1 for i in range(os_, os_+9)
                if state.board[i] > 0 and state.board[i] % 2 == 0) / 9.0
    f[26] = sum(1 for i in range(cs, cs+9)
                if state.board[i] > 0 and state.board[i] % 2 == 0) / 9.0
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

    f[30] = sum(state.board[i] for i in range(os_, os_+9)
                if state.board[i] > 0 and state.board[i] % 2 == 0) / 162.0
    f[31] = sum(state.board[i] for i in range(cs, cs+9)
                if state.board[i] > 0 and state.board[i] % 2 == 0) / 162.0
    f[32] = sum(state.board[cs:cs+9]) / 162.0
    f[33] = sum(state.board[os_:os_+9]) / 162.0

    return f


def _nn_eval(state: TogyzkumalakState) -> float:
    """Returns value in [-1, 1] from current player's perspective."""
    import numpy as np
    w = _nn
    x  = _nn_features(state)
    h1 = np.maximum(0.0, x  @ w['W1'] + w['b1'])
    h2 = np.maximum(0.0, h1 @ w['W2'] + w['b2'])
    h3 = np.maximum(0.0, h2 @ w['W3'] + w['b3'])
    # Value head: Wv/bv (AlphaZero-lite) or W4/b4 (value-only v2) or W3/b3 (v1)
    if 'Wv' in w:
        out = np.tanh(h3 @ w['Wv'] + w['bv']).item()
    elif 'W4' in w:
        out = np.tanh(h3 @ w['W4'] + w['b4']).item()
    else:
        out = np.tanh(h2 @ w['W3'] + w['b3']).item()
    return out


def _nn_policy(state: TogyzkumalakState) -> 'np.ndarray | None':
    """Returns policy probabilities (9,) over pockets 0-8, or None if no policy head."""
    if _nn is None or 'Wp' not in _nn:
        return None
    import numpy as np
    w = _nn
    x  = _nn_features(state)
    h1 = np.maximum(0.0, x  @ w['W1'] + w['b1'])
    h2 = np.maximum(0.0, h1 @ w['W2'] + w['b2'])
    h3 = np.maximum(0.0, h2 @ w['W3'] + w['b3'])
    logits = h3 @ w['Wp'] + w['bp']
    e = np.exp(logits - logits.max())
    return e / (e.sum() + 1e-9)

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
    return _evaluate_heuristic(state)


def _evaluate_heuristic(state: TogyzkumalakState) -> float:
    cur = state.currentPlayer
    opp = 1 - cur
    cs = cur * 9
    os = opp * 9

    # 1. Kazan difference — scales up as board empties (late-game precision)
    total_on_board = sum(state.board)
    kazan_weight = 4.0 + (1.0 - total_on_board / 162.0) * 5.0  # 4→9 as board empties
    ev = (state.kazans[cur] - state.kazans[opp]) * kazan_weight

    # 2. Tuzdyk advantage
    my_tuz = state.tuzdyks[cur] != -1
    opp_tuz = state.tuzdyks[opp] != -1
    ev += (my_tuz - opp_tuz) * 35.0

    # 2b. Tuzdyk position quality: center tuzdyk (pos 3-5) captures more on average
    if my_tuz:
        tpos = state.tuzdyks[cur] % 9
        ev += (4 - abs(tpos - 4)) * 3.0   # center=+12, edge=0
    if opp_tuz:
        tpos = state.tuzdyks[opp] % 9
        ev -= (4 - abs(tpos - 4)) * 3.0

    # 3. Tuzdyk DANGER: stones we'd donate to opponent's tuzdyk
    opp_tuz_pos = state.tuzdyks[opp]
    if opp_tuz_pos != -1:
        for i in range(cs, cs + 9):
            if state.board[i] > 0:
                _, d0, d1 = _simulate_sow(state.board, state.tuzdyks, i)
                donated = d0 if opp == 0 else d1
                if donated > 0:
                    ev -= donated * 5.0

    # 4. Capture opportunities (opponent's even pockets we can target)
    for i in range(os, os + 9):
        v = state.board[i]
        if v > 0 and v % 2 == 0:
            ev += v * 0.9

    # 5. Our even pockets = opponent capture bait
    for i in range(cs, cs + 9):
        v = state.board[i]
        if v > 0 and v % 2 == 0:
            ev -= v * 0.9

    # 6. THREAT DETECTION: which of our pockets can opponent capture NEXT move?
    # Makes AI proactively defend instead of waiting to be attacked.
    for i in range(os, os + 9):
        if state.board[i] == 0:
            continue
        land, _, _ = _simulate_sow(state.board, state.tuzdyks, i)
        if land < 0:
            continue
        if cs <= land < cs + 9:
            fut = state.board[land] + 1
            if fut % 2 == 0:
                ev -= fut * 1.8  # strongly penalize being under immediate threat

    # 7. SETUP DETECTION: our moves that create future capture opportunities
    # (2-move planning: we set it up now, capture next turn)
    best_setup = 0.0
    for i in range(cs, cs + 9):
        if state.board[i] == 0:
            continue
        land, d0, d1 = _simulate_sow(state.board, state.tuzdyks, i)
        donated_to_opp = d0 if opp == 0 else d1
        if donated_to_opp > 0:
            continue
        if land < 0:
            continue
        if os <= land < os + 9:
            fut = state.board[land] + 1
            if fut % 2 == 0 and fut >= 4:
                best_setup = max(best_setup, fut * 0.5)
    ev += best_setup

    # 8. Tuzdyk creation readiness: opponent pocket with 2 stones = one step away
    if not my_tuz:
        opp_tuz_idx = state.tuzdyks[opp]
        for i in range(os, os + 9):
            if state.board[i] == 2 and i not in (8, 17):
                if opp_tuz_idx == -1 or opp_tuz_idx % 9 != i % 9:
                    ev += 12.0

    # 9. Mobility: more active pockets = more strategic options
    my_active = sum(1 for i in range(cs, cs + 9) if state.board[i] > 0)
    opp_active = sum(1 for i in range(os, os + 9) if state.board[i] > 0)
    ev += (my_active - opp_active) * 1.5

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


def order_moves(state: TogyzkumalakState, moves: list,
                policy_probs: 'np.ndarray | None' = None) -> list:
    opp = 1 - state.currentPlayer
    cur = state.currentPlayer
    cs = cur * 9
    os = opp * 9

    # Pre-compute: which of OUR pockets are immediately threatened by opponent?
    threatened = set()
    for i in range(os, os + 9):
        if state.board[i] == 0:
            continue
        land, _, _ = _simulate_sow(state.board, state.tuzdyks, i)
        if land >= 0 and cs <= land < cs + 9:
            fut = state.board[land] + 1
            if fut % 2 == 0:
                threatened.add(land)

    def priority(m: int) -> int:
        stones = state.board[m]
        if stones == 0:
            return -99999

        land, d0, d1 = _simulate_sow(state.board, state.tuzdyks, m)

        # CRITICAL: heavily penalize donating stones to opponent's tuzdyk
        donated_to_opp = d0 if opp == 0 else d1
        if donated_to_opp > 0:
            return -donated_to_opp * 50

        score = 0

        # Policy network bonus (AlphaZero-lite): boosts moves NN thinks are good
        if policy_probs is not None:
            score += int(policy_probs[m % 9] * 80)
        score += int(_learning_bias(state, m) * 4)

        # Picking up from a threatened pocket = defensive move (neutralizes threat)
        if m in threatened:
            score += 70

        if land < 0 or land in state.tuzdyks:
            return score

        if os <= land < os + 9:
            fut = state.board[land] + 1
            if fut % 2 == 0:
                return score + fut * 10    # capture
            if (fut == 3
                    and state.tuzdyks[cur] == -1
                    and land not in (8, 17)
                    and (state.tuzdyks[opp] == -1
                         or state.tuzdyks[opp] % 9 != land % 9)):
                return score + 100         # tuzdyk creation

            # Setup: landing makes opponent pocket even (future capture)
            if fut % 2 == 0 and fut >= 4:
                score += fut * 3

        return score

    return sorted(moves, key=priority, reverse=True)


# ─────────────────────────────────────────────────────────────────────────────
# QUIESCENCE SEARCH
# When depth=0, keep searching captures and tuzdyk moves to avoid
# "horizon effect" (AI stops just before losing/gaining big material)
# ─────────────────────────────────────────────────────────────────────────────

def _quiescence(state: TogyzkumalakState, alpha: float, beta: float,
                deadline: float, qdepth: int = 6) -> float:
    if time.time() > deadline:
        raise _Timeout()

    # Full evaluate (NN + heuristic blend) for accurate stand-pat
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
             alpha: float, beta: float, deadline: float, ply: int = 0) -> float:
    if time.time() > deadline:
        raise _Timeout()

    orig_alpha = alpha

    # ── Transposition Table lookup ────────────────────────────────────────
    key = _tt_key(state)
    entry = _TT.get(key)
    if entry is not None and entry[0] >= depth:
        tt_d, tt_s, tt_f, tt_mv = entry
        if tt_f == _TT_EXACT:
            return tt_s
        elif tt_f == _TT_LOWER:
            alpha = max(alpha, tt_s)
        elif tt_f == _TT_UPPER:
            beta  = min(beta,  tt_s)
        if alpha >= beta:
            return tt_s

    if state.isGameOver:
        if state.winner == state.currentPlayer:
            return 10000.0 + depth
        elif state.winner == -1:
            return 0.0
        else:
            return -(10000.0 + depth)

    if depth == 0:
        return _quiescence(state, alpha, beta, deadline)

    moves = state.getPossibleMoves(state.currentPlayer)
    if not moves:
        return -(10000.0 + depth)

    moves = order_moves(state, moves)

    # ── Promote killer moves to front ────────────────────────────────────
    if ply < 32:
        k0, k1 = _killers[ply]
        if k1 is not None and k1 in moves:
            moves.remove(k1); moves.insert(0, k1)
        if k0 is not None and k0 in moves:
            moves.remove(k0); moves.insert(0, k0)

    best      = float('-inf')
    best_move = moves[0]

    for move in moves:
        child = state.clone()
        child.makeMove(move)
        score = -_negamax(child, depth - 1, -beta, -alpha, deadline, ply + 1)
        if score > best:
            best = score
            best_move = move
        if score > alpha:
            alpha = score
        if alpha >= beta:
            # Beta cutoff — store as killer move for this ply
            if ply < 32:
                k = _killers[ply]
                if k[0] != move:
                    k[1] = k[0]
                    k[0] = move
            break

    # ── Store in Transposition Table ─────────────────────────────────────
    if len(_TT) < _TT_MAX:
        if best <= orig_alpha:
            flag = _TT_UPPER
        elif best >= beta:
            flag = _TT_LOWER
        else:
            flag = _TT_EXACT
        _TT[key] = (depth, best, flag, best_move)

    return best


def _gpt_pick_move(state: TogyzkumalakState, top_moves: list, scores: dict) -> int:
    """Ask GPT-4o-mini to pick best move from top candidates. Returns move index or -1 on failure."""
    api_key = os.environ.get('OPENAI_API_KEY', '')
    if not api_key:
        return -1
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)

        cur = state.currentPlayer
        opp = 1 - cur
        cs, os_ = cur * 9, opp * 9

        my_pockets  = [state.board[cs + i]  for i in range(9)]
        opp_pockets = [state.board[os_ + i] for i in range(9)]

        # Rank moves by engine score (best first) so GPT sees them in order
        top_moves = sorted(top_moves, key=lambda m: scores.get(m, 0), reverse=True)

        move_lines = []
        for idx, m in enumerate(top_moves):
            land, d0, d1 = _simulate_sow(state.board, state.tuzdyks, m)
            donated = d0 if opp == 0 else d1
            rank = ["BEST", "2ND", "3RD"][idx] if idx < 3 else f"{idx+1}TH"
            desc = f"Move {idx+1} ({rank} by engine): pocket {m % 9} ({state.board[m]} stones)"
            if donated > 0:
                desc += f" — WARNING: gives {donated} stone(s) to opponent tuzdyk!"
            elif land >= 0:
                fut = state.board[land] + 1
                if os_ <= land < os_ + 9 and fut % 2 == 0:
                    desc += f" — CAPTURES {fut} stones from opponent pocket {land % 9}"
                elif os_ <= land < os_ + 9 and fut == 3 and state.tuzdyks[cur] == -1 and land not in (8, 17):
                    desc += f" — CREATES TUZDYK at opponent pocket {land % 9}"
                else:
                    side = "opponent" if os_ <= land < os_ + 9 else "own"
                    desc += f" — lands on {side} pocket {land % 9} (now {fut} stones)"
            move_lines.append(desc)

        tuz_my  = f"pocket {state.tuzdyks[cur] % 9}" if state.tuzdyks[cur]  != -1 else "none"
        tuz_opp = f"pocket {state.tuzdyks[opp] % 9}" if state.tuzdyks[opp] != -1 else "none"

        prompt = (
            "You are an expert Togyzkumalak player (Central Asian mancala game).\n\n"
            "RULES:\n"
            "- Sow stones counter-clockwise; last stone on OPPONENT side with EVEN total = capture all\n"
            "- Last stone making exactly 3 on opponent side (not last pocket) = tuzdyk (permanent tribute)\n"
            "- Tuzdyk drains every passing stone to owner's kazan permanently\n"
            "- NEVER donate stones to opponent's tuzdyk\n"
            "- First to exceed 81 stones wins\n\n"
            f"POSITION (you are Player {cur}):\n"
            f"  Your pockets [0-8]: {my_pockets}  (kazan: {state.kazans[cur]})\n"
            f"  Opp pockets  [0-8]: {opp_pockets}  (kazan: {state.kazans[opp]})\n"
            f"  Your tuzdyk: {tuz_my} | Opponent tuzdyk: {tuz_opp}\n\n"
            "CANDIDATE MOVES (ranked by search engine — Move 1 is engine's top pick):\n"
            + "\n".join(move_lines) + "\n\n"
            "The engine ranking is tactically accurate. Override it only if you see a clear strategic reason "
            "(tuzdyk creation, preventing opponent tuzdyk, large capture). "
            "Otherwise prefer Move 1. Reply with ONLY the move number."
        )

        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=5,
            temperature=0.0,
            timeout=4.0,
        )
        choice = resp.choices[0].message.content.strip()
        pick = int(choice) - 1
        if 0 <= pick < len(top_moves):
            print(f"[GPT] chose move {top_moves[pick] % 9} (option {pick+1})", flush=True)
            return top_moves[pick]
    except Exception as e:
        print(f"[GPT] fallback: {e}", flush=True)
    return -1


def get_best_move_alphabeta(root_state: TogyzkumalakState, root_player: int,
                             max_time_seconds: float = 3.0) -> int:
    global _TT, _killers
    # Fresh search per move — clear TT and killers
    _TT = {}
    _killers = [[None, None] for _ in range(32)]

    moves = root_state.getPossibleMoves(root_player)
    if not moves:
        return -1
    if len(moves) == 1:
        return moves[0]

    # Use policy network for initial move ordering if available
    root_policy = _nn_policy(root_state)
    moves = order_moves(root_state, moves, policy_probs=root_policy)
    best_move   = moves[0]
    final_scores: dict = {}
    deadline    = time.time() + max_time_seconds * 0.95
    prev_score  = 0.0

    for depth in range(1, 32):
        try:
            best_score   = float('-inf')
            current_best = moves[0]
            depth_scores: dict = {}

            # Aspiration window: narrow search centred on previous depth's score
            asp   = 20.0 if depth > 3 else float('inf')
            a_lo  = prev_score - asp
            alpha = a_lo

            retry = False
            for move in moves:
                child = root_state.clone()
                child.makeMove(move)
                score = -_negamax(child, depth - 1, -float('inf'), -alpha, deadline, ply=1)
                score += _learning_bias(root_state, move)
                depth_scores[move] = score
                if score > best_score:
                    best_score   = score
                    current_best = move
                if score > alpha:
                    alpha = score

            # Aspiration failed (score outside window) → re-search full window
            if asp < float('inf') and best_score <= a_lo:
                retry = True

            if retry:
                best_score   = float('-inf')
                alpha        = float('-inf')
                current_best = moves[0]
                for move in moves:
                    child = root_state.clone()
                    child.makeMove(move)
                    score = -_negamax(child, depth - 1, -float('inf'), -alpha, deadline, ply=1)
                    score += _learning_bias(root_state, move)
                    depth_scores[move] = score
                    if score > best_score:
                        best_score   = score
                        current_best = move
                    if score > alpha:
                        alpha = score

            prev_score   = best_score
            best_move    = current_best
            final_scores = depth_scores
            print(f"[AI] depth={depth} move={best_move} score={best_score:.1f}", flush=True)
            # Move best move to front for next iteration (move ordering)
            moves = [best_move] + [m for m in moves if m != best_move]

            if abs(best_score) > 9000:
                break

        except _Timeout:
            print(f"[AI] timeout at depth={depth}", flush=True)
            break

    # Build top-3 candidates for GPT
    if final_scores:
        bs = final_scores.get(best_move, float('-inf'))
        top3 = sorted(
            [m for m, s in final_scores.items() if s >= bs - 10.0],
            key=lambda m: final_scores[m], reverse=True
        )[:3]

        # Ask GPT to pick among top candidates
        if len(top3) > 1:
            gpt_move = _gpt_pick_move(root_state, top3, final_scores)
            if gpt_move != -1:
                return gpt_move

        # Fallback: return best alpha-beta move
        best_move = top3[0]

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
    '9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9|0,0|-1,-1|0': [5],
    '1,10,10,10,10,10,10,10,10,9,9,9,9,9,9,9,9,9|0,0|-1,-1|1': [17],
    '9,1,10,10,10,10,10,10,10,0,9,9,9,9,9,9,9,9|10,0|-1,-1|1': [17],
    '9,9,1,10,10,10,10,10,10,10,0,9,9,9,9,9,9,9|10,0|-1,-1|1': [11],
    '9,9,9,1,10,10,10,10,10,10,10,0,9,9,9,9,9,9|10,0|-1,-1|1': [17],
    '9,9,9,9,1,10,10,10,10,10,10,10,0,9,9,9,9,9|10,0|-1,-1|1': [11],
    '9,9,9,9,9,1,10,10,10,10,10,10,10,0,9,9,9,9|10,0|-1,-1|1': [11],
    '9,9,9,9,9,9,1,10,10,10,10,10,10,10,0,9,9,9|10,0|-1,-1|1': [15],
    '9,9,9,9,9,9,9,1,10,10,10,10,10,10,10,0,9,9|10,0|-1,-1|1': [16],
    '9,9,9,9,9,9,9,9,1,10,10,10,10,10,10,10,0,9|10,0|-1,-1|1': [17],
    '1,10,10,10,10,10,10,10,10,1,10,10,10,10,10,10,10,10|0,0|-1,-1|0': [8],
    '0,10,10,10,10,10,10,10,10,9,1,10,10,10,10,10,10,10|0,2|-1,-1|0': [8],
    '2,11,10,10,10,10,10,10,10,9,9,1,10,10,10,10,10,10|0,0|-1,-1|0': [8],
    '2,11,11,10,10,10,10,10,10,9,9,9,1,10,10,10,10,10|0,0|-1,-1|0': [8],
    '2,11,11,11,10,10,10,10,10,9,9,9,9,1,10,10,10,10|0,0|-1,-1|0': [8],
    '2,11,11,11,11,10,10,10,10,9,9,9,9,9,1,10,10,10|0,0|-1,-1|0': [8],
    '2,11,11,11,11,11,10,10,10,9,9,9,9,9,9,1,10,10|0,0|-1,-1|0': [8],
    '2,11,11,11,11,11,11,10,10,9,9,9,9,9,9,9,1,10|0,0|-1,-1|0': [2],
    '2,11,11,11,11,11,11,11,10,9,9,9,9,9,9,9,9,1|0,0|-1,-1|0': [7],
    '0,1,10,10,10,10,10,10,10,0,1,10,10,10,10,10,10,10|10,10|-1,-1|0': [8],
    '10,0,10,10,10,10,10,10,10,0,9,1,10,10,10,10,10,10|10,2|-1,-1|0': [8],
    '10,2,11,10,10,10,10,10,10,0,9,9,1,10,10,10,10,10|10,0|-1,-1|0': [8],
    '10,2,11,11,10,10,10,10,10,0,9,9,9,1,10,10,10,10|10,0|-1,-1|0': [8],
    '10,2,11,11,11,10,10,10,10,0,9,9,9,9,1,10,10,10|10,0|-1,-1|0': [7],
    '10,2,11,11,11,11,10,10,10,0,9,9,9,9,9,1,10,10|10,0|-1,-1|0': [8],
    '10,2,11,11,11,11,11,10,10,0,9,9,9,9,9,9,1,10|10,0|-1,-1|0': [8],
    '10,2,11,11,11,11,11,11,10,0,9,9,9,9,9,9,9,1|10,0|-1,-1|0': [6],
    '0,9,1,10,10,10,10,10,10,1,1,10,10,10,10,10,10,10|10,10|-1,-1|0': [8],
    '10,0,1,10,10,10,10,10,10,10,0,1,10,10,10,10,10,10|10,10|-1,-1|0': [8],
    '10,10,0,10,10,10,10,10,10,10,0,9,1,10,10,10,10,10|10,2|-1,-1|0': [8],
    '10,10,2,11,10,10,10,10,10,10,0,9,9,1,10,10,10,10|10,0|-1,-1|0': [8],
    '10,10,2,11,11,10,10,10,10,10,0,9,9,9,1,10,10,10|10,0|-1,-1|0': [8],
    '10,10,2,11,11,11,10,10,10,10,0,9,9,9,9,1,10,10|10,0|-1,-1|0': [8],
    '10,10,2,11,11,11,11,10,10,10,0,9,9,9,9,9,1,10|10,0|-1,-1|0': [8],
    '10,10,2,11,11,11,11,11,10,10,0,9,9,9,9,9,9,1|10,0|-1,-1|0': [6],
    '0,9,9,1,10,10,10,10,10,1,11,1,10,10,10,10,10,10|10,10|-1,-1|0': [8],
    '10,0,9,1,10,10,10,10,10,10,1,1,10,10,10,10,10,10|10,10|-1,-1|0': [8],
    '10,10,0,1,10,10,10,10,10,10,10,0,1,10,10,10,10,10|10,10|-1,-1|0': [8],
    '10,10,10,0,10,10,10,10,10,10,10,0,9,1,10,10,10,10|10,2|-1,-1|0': [8],
    '10,10,10,2,11,10,10,10,10,10,10,0,9,9,1,10,10,10|10,0|-1,-1|0': [8],
    '10,10,10,2,11,11,10,10,10,10,10,0,9,9,9,1,10,10|10,0|-1,-1|0': [4],
    '10,10,10,2,11,11,11,10,10,10,10,0,9,9,9,9,1,10|10,0|-1,-1|0': [4],
    '10,10,10,2,11,11,11,11,10,10,10,0,9,9,9,9,9,1|10,0|-1,-1|0': [7],
    '0,9,9,9,1,10,10,10,10,1,11,11,1,10,10,10,10,10|10,10|-1,-1|0': [8],
    '10,0,9,9,1,10,10,10,10,10,1,11,1,10,10,10,10,10|10,10|-1,-1|0': [8],
    '10,10,0,9,1,10,10,10,10,10,10,1,1,10,10,10,10,10|10,10|-1,-1|0': [8],
    '10,10,10,0,1,10,10,10,10,10,10,10,0,1,10,10,10,10|10,10|-1,-1|0': [8],
    '10,10,10,10,0,10,10,10,10,10,10,10,0,9,1,10,10,10|10,2|-1,-1|0': [8],
    '10,10,10,10,2,11,10,10,10,10,10,10,0,9,9,1,10,10|10,0|-1,-1|0': [8],
    '10,10,10,10,2,11,11,10,10,10,10,10,0,9,9,9,1,10|10,0|-1,-1|0': [5],
    '10,10,10,10,2,11,11,11,10,10,10,10,0,9,9,9,9,1|10,0|-1,-1|0': [5],
    '0,9,9,9,9,1,10,10,10,1,11,11,11,1,10,10,10,10|10,10|-1,-1|0': [8],
    '10,0,9,9,9,1,10,10,10,10,1,11,11,1,10,10,10,10|10,10|-1,-1|0': [8],
    '10,10,0,9,9,1,10,10,10,10,10,1,11,1,10,10,10,10|10,10|-1,-1|0': [8],
    '10,10,10,0,9,1,10,10,10,10,10,10,1,1,10,10,10,10|10,10|-1,-1|0': [8],
    '10,10,10,10,0,1,10,10,10,10,10,10,10,0,1,10,10,10|10,10|-1,-1|0': [8],
    '10,10,10,10,10,0,10,10,10,10,10,10,10,0,9,1,10,10|10,2|-1,-1|0': [8],
    '10,10,10,10,10,2,11,10,10,10,10,10,10,0,9,9,1,10|10,0|-1,-1|0': [6],
    '10,10,10,10,10,2,11,11,10,10,10,10,10,0,9,9,9,1|10,0|-1,-1|0': [6],
    '0,9,9,9,9,9,1,10,10,1,11,11,11,11,1,10,10,10|10,10|-1,-1|0': [8],
    '10,0,9,9,9,9,1,10,10,10,1,11,11,11,1,10,10,10|10,10|-1,-1|0': [8],
    '10,10,0,9,9,9,1,10,10,10,10,1,11,11,1,10,10,10|10,10|-1,-1|0': [8],
    '10,10,10,0,9,9,1,10,10,10,10,10,1,11,1,10,10,10|10,10|-1,-1|0': [8],
    '10,10,10,10,0,9,1,10,10,10,10,10,10,1,1,10,10,10|10,10|-1,-1|0': [8],
    '10,10,10,10,10,0,1,10,10,10,10,10,10,10,0,1,10,10|10,10|-1,-1|0': [8],
    '10,10,10,10,10,10,0,10,10,10,10,10,10,10,0,9,1,10|10,2|-1,-1|0': [8],
    '10,10,10,10,10,10,2,11,10,10,10,10,10,10,0,9,9,1|10,0|-1,-1|0': [7],
    '0,9,9,9,9,9,9,1,10,1,11,11,11,11,11,1,10,10|10,10|-1,-1|0': [6],
    '10,0,9,9,9,9,9,1,10,10,1,11,11,11,11,1,10,10|10,10|-1,-1|0': [8],
    '10,10,0,9,9,9,9,1,10,10,10,1,11,11,11,1,10,10|10,10|-1,-1|0': [8],
    '10,10,10,0,9,9,9,1,10,10,10,10,1,11,11,1,10,10|10,10|-1,-1|0': [6],
    '10,10,10,10,0,9,9,1,10,10,10,10,10,1,11,1,10,10|10,10|-1,-1|0': [6],
    '10,10,10,10,10,0,9,1,10,10,10,10,10,10,1,1,10,10|10,10|-1,-1|0': [8],
    '10,10,10,10,10,10,0,1,10,10,10,10,10,10,10,0,1,10|10,10|-1,-1|0': [8],
    '10,10,10,10,10,10,10,0,10,10,10,10,10,10,10,0,9,1|10,2|-1,-1|0': [8],
    '0,9,9,9,9,9,9,9,1,1,11,11,11,11,11,11,1,10|10,10|-1,-1|0': [7],
    '10,0,9,9,9,9,9,9,1,10,1,11,11,11,11,11,1,10|10,10|-1,-1|0': [4],
    '10,10,0,9,9,9,9,9,1,10,10,1,11,11,11,11,1,10|10,10|-1,-1|0': [6],
    '10,10,10,0,9,9,9,9,1,10,10,10,1,11,11,11,1,10|10,10|-1,-1|0': [6],
    '10,10,10,10,0,9,9,9,1,10,10,10,10,1,11,11,1,10|10,10|-1,-1|0': [7],
    '10,10,10,10,10,0,9,9,1,10,10,10,10,10,1,11,1,10|10,10|-1,-1|0': [7],
    '10,10,10,10,10,10,0,9,1,10,10,10,10,10,10,1,1,10|10,10|-1,-1|0': [5],
    '10,10,10,10,10,10,10,0,1,10,10,10,10,10,10,10,0,1|10,10|-1,-1|0': [6],
}


def get_best_move_mcts(root_state: TogyzkumalakState, root_player: int,
                        iterations: int = 20000,
                        max_time_seconds: float = 3.0) -> int:
    moves = OPENING_BOOK.get(_state_key(root_state))
    if moves:
        legal = root_state.getPossibleMoves(root_player)
        candidates = [m for m in moves if m in legal]
        if candidates:
            return random.choice(candidates)

    if max_time_seconds < 1.0:
        return _mcts(root_state, root_player, iterations, max_time_seconds)

    return get_best_move_alphabeta(root_state, root_player, max_time_seconds)
