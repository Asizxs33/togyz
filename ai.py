import math
import json
import os
import random
import time
from game_logic import TogyzkumalakState

try:
    import psycopg
except ImportError:
    psycopg = None

CENTER_RELATIVE_PITS = {3, 4, 5}
DATABASE_URL = os.environ.get("DATABASE_URL")
LEARNING_FILE = os.environ.get("TOGYZ_LEARNING_FILE", os.path.join(os.path.dirname(__file__), "ai_learning.json"))
LEARNING_STATS = None
LEARNING_DB_READY = False


def _side_range(player: int):
    start = player * 9
    return range(start, start + 9)


def _relative_pit(index: int) -> int:
    return index % 9


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
    except Exception:
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
    except Exception:
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
        state.tuzdyks = tuzdyks[:2]
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


def _move_features(state: TogyzkumalakState, move: int, player: int):
    before_kazan = state.kazans[player]
    before_tuzdyk = state.tuzdyks[player]
    before_count = state.board[move]

    child = state.clone()
    child.makeMove(move)

    own_even_after = sum(
        child.board[i]
        for i in _side_range(player)
        if child.board[i] > 0 and child.board[i] % 2 == 0
    )
    active_after = sum(1 for i in _side_range(player) if child.board[i] > 0)

    return {
        "state": child,
        "capture_gain": child.kazans[player] - before_kazan,
        "tuzdyk_created": before_tuzdyk == -1 and child.tuzdyks[player] != -1,
        "own_even_after": own_even_after,
        "active_after": active_after,
        "large_setup": before_count >= 10,
    }


def _immediate_threat_score(state: TogyzkumalakState, player: int) -> float:
    moves = state.getPossibleMoves(player)
    if not moves:
        return 0.0

    best = 0.0
    for move in moves:
        features = _move_features(state, move, player)
        score = features["capture_gain"]
        if features["tuzdyk_created"]:
            score += 35
        best = max(best, score)
    return best


def _tactical_move_score(state: TogyzkumalakState, move: int, player: int) -> float:
    features = _move_features(state, move, player)
    score = 0.0

    # 2. Make even numbers / capture.
    score += features["capture_gain"] * 2.5

    # 3. Do not leave own even pockets exposed.
    score -= features["own_even_after"] * 0.9

    # 4. Take tuzdyk early.
    if features["tuzdyk_created"]:
        score += 100

    # 5. Keep active pockets.
    score += features["active_after"] * 1.5

    # 6. Prefer strong setup pockets with many stones.
    if features["large_setup"]:
        score += min(state.board[move], 18) * 0.35

    # 7. Account for the opponent's next tactical reply.
    score -= _immediate_threat_score(features["state"], 1 - player) * 0.75
    score += _learning_bias(state, move)

    return score


def _evaluate_for_player(state: TogyzkumalakState, player: int) -> float:
    opponent = 1 - player
    total_stones = sum(state.board)
    progress = 1 - total_stones / 162

    # 9. Endgame: kazan difference matters more as the board empties.
    kazan_weight = 4 + progress * 5
    score = (state.kazans[player] - state.kazans[opponent]) * kazan_weight

    # 4 and 8. Tuzdyk value, with center tuzdyks favored.
    for owner, sign in ((player, 1), (opponent, -1)):
        tuzdyk = state.tuzdyks[owner]
        if tuzdyk != -1:
            center_bonus = max(0, 4 - abs(_relative_pit(tuzdyk) - 4)) * 3
            score += sign * (35 + center_bonus)

    # 3, 5, 6, 8. Even danger, mobility, setup pockets, and center control.
    for owner, sign in ((player, 1), (opponent, -1)):
        active = 0
        center_stones = 0
        even_danger = 0
        large_setup = 0

        for index in _side_range(owner):
            stones = state.board[index]
            if stones <= 0:
                continue
            active += 1
            if _relative_pit(index) in CENTER_RELATIVE_PITS:
                center_stones += stones
            if stones % 2 == 0:
                even_danger += stones
            if stones >= 10:
                large_setup += min(stones, 18)

        score += sign * active * 1.5
        score += sign * center_stones * 0.45
        score -= sign * even_danger * 0.9
        score += sign * large_setup * 0.22

    # 7. Threat detection: simulate best immediate tactical reply.
    score -= _immediate_threat_score(state, opponent) * 1.1
    score += _immediate_threat_score(state, player) * 0.45

    return score


def evaluate(state: TogyzkumalakState) -> float:
    return _evaluate_for_player(state, state.currentPlayer)


def order_moves(state: TogyzkumalakState, moves: list) -> list:
    return sorted(
        moves,
        key=lambda move: _tactical_move_score(state, move, state.currentPlayer),
        reverse=True,
    )


class _Timeout(Exception):
    pass


def _negamax(state: TogyzkumalakState, depth: int,
             alpha: float, beta: float, deadline: float) -> float:
    if time.time() > deadline:
        raise _Timeout()

    if state.isGameOver:
        cur = state.currentPlayer
        if state.winner == cur:
            return 10000.0 + depth
        if state.winner == -1:
            return 0.0
        return -(10000.0 + depth)

    if depth == 0:
        return evaluate(state)

    moves = state.getPossibleMoves(state.currentPlayer)
    if not moves:
        return -(10000.0 + depth)

    moves = order_moves(state, moves)
    best = float('-inf')

    for move in moves:
        child = state.clone()
        child.makeMove(move)
        score = -_negamax(child, depth - 1, -beta, -alpha, deadline)
        best = max(best, score)
        alpha = max(alpha, score)
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
                alpha = max(alpha, score)

            best_move = current_best
            moves = [best_move] + [move for move in moves if move != best_move]

            if abs(best_score) > 9000:
                break
        except _Timeout:
            break

    return best_move


class _MCTSNode:
    __slots__ = ('state', 'move', 'parent', 'children', 'wins', 'visits', 'untriedMoves')

    def __init__(self, state: TogyzkumalakState, move=None, parent=None):
        self.state = state
        self.move = move
        self.parent = parent
        self.children = []
        self.wins = 0.0
        self.visits = 0
        self.untriedMoves = order_moves(state, state.getPossibleMoves(state.currentPlayer))

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

        while not node.untriedMoves and node.children:
            node = node.best_child()
            state.makeMove(node.move)

        if node.untriedMoves:
            move = node.untriedMoves.pop(0)
            state.makeMove(move)
            child = _MCTSNode(state.clone(), move, node)
            node.children.append(child)
            node = child

        sim = state.clone()
        for _ in range(150):
            if sim.isGameOver:
                break
            moves = sim.getPossibleMoves(sim.currentPlayer)
            if not moves:
                break
            ordered = order_moves(sim, moves)
            chosen = ordered[0] if random.random() < 0.85 else random.choice(moves)
            sim.makeMove(chosen)

        if sim.isGameOver:
            if sim.winner == root_player:
                result = 1.0
            elif sim.winner == -1:
                result = 0.5
            else:
                result = 0.0
        else:
            diff = _evaluate_for_player(sim, root_player)
            result = 1.0 / (1.0 + math.exp(-diff / 15.0))

        bp = node
        while bp is not None:
            bp.visits += 1
            mover = 1 - bp.state.currentPlayer
            bp.wins += result if mover == root_player else (1.0 - result)
            bp = bp.parent

    if not root.children:
        return -1

    return max(
        root.children,
        key=lambda child: child.visits + _tactical_move_score(root_state, child.move, root_player) * 0.05,
    ).move


OPENING_BOOK = {
    (162, 0): [6, 8],
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
