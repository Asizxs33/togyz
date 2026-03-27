import math
import random
import time
from game_logic import TogyzkumalakState

# ─────────────────────────────────────────────────────────────────────────────
# EVALUATION FUNCTION
# Returns score from state.currentPlayer's perspective. Higher = better for them.
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(state: TogyzkumalakState) -> float:
    cur = state.currentPlayer
    opp = 1 - cur
    cs = cur * 9
    os = opp * 9

    # 1. Kazan difference — primary factor
    ev = (state.kazans[cur] - state.kazans[opp]) * 3.0

    # 2. Tuzdyk advantage (each tuzdyk drains ~1 stone/turn from opponent)
    my_tuz = state.tuzdyks[cur] != -1
    opp_tuz = state.tuzdyks[opp] != -1
    ev += (my_tuz - opp_tuz) * 18.0

    # 3. Capture opportunities: opponent's even-stoned pockets are free targets
    for i in range(os, os + 9):
        v = state.board[i]
        if v > 0 and v % 2 == 0:
            ev += v * 0.5
    # Penalty: our own even pockets are vulnerable
    for i in range(cs, cs + 9):
        v = state.board[i]
        if v > 0 and v % 2 == 0:
            ev -= v * 0.5

    # 4. Tuzdyk opportunity: landing 3 stones on opponent side is huge
    if not my_tuz:
        for i in range(os, os + 9):
            if state.board[i] == 2 and i not in (8, 17):
                # One capture away from making tuzdyk
                opp_relative = i % 9
                opp_tuz_pos = state.tuzdyks[opp]
                if opp_tuz_pos == -1 or opp_tuz_pos % 9 != opp_relative:
                    ev += 6.0

    # 5. Stone mobility (more stones = more options, harder to be stranded)
    ev += (sum(state.board[cs:cs+9]) - sum(state.board[os:os+9])) * 0.1

    return ev


# ─────────────────────────────────────────────────────────────────────────────
# MOVE ORDERING  (critical for alpha-beta efficiency)
# ─────────────────────────────────────────────────────────────────────────────

def _landing_pocket(state: TogyzkumalakState, m: int) -> int:
    """Approximate last stone landing pocket for move m."""
    stones = state.board[m]
    if stones == 0:
        return -1
    return (m + 1) % 18 if stones == 1 else (m + stones - 1) % 18


def order_moves(state: TogyzkumalakState, moves: list) -> list:
    opp = 1 - state.currentPlayer
    os = opp * 9

    def priority(m: int) -> int:
        land = _landing_pocket(state, m)
        if land < 0 or land in state.tuzdyks:
            return 0
        if os <= land < os + 9:
            fut = state.board[land] + 1
            if fut % 2 == 0:
                return fut * 10           # direct capture — highest priority
            if fut == 3 and state.tuzdyks[state.currentPlayer] == -1:
                is_ninth = land in (8, 17)
                if not is_ninth:
                    opp_tuz = state.tuzdyks[opp]
                    if opp_tuz == -1 or opp_tuz % 9 != land % 9:
                        return 80         # tuzdyk creation — very high priority
        return 0

    return sorted(moves, key=priority, reverse=True)


# ─────────────────────────────────────────────────────────────────────────────
# ALPHA-BETA NEGAMAX with Iterative Deepening
# ─────────────────────────────────────────────────────────────────────────────

class _Timeout(Exception):
    pass


def _negamax(state: TogyzkumalakState, depth: int,
             alpha: float, beta: float, deadline: float) -> float:
    """
    Returns score from state.currentPlayer's perspective.
    Uses negamax: score for current player = -score for opponent.
    """
    if time.time() > deadline:
        raise _Timeout()

    if state.isGameOver:
        cur = state.currentPlayer
        if state.winner == cur:
            # Current player already won — shouldn't happen (game ends on prev move),
            # but handle gracefully
            return 10000.0 + depth
        elif state.winner == -1:
            return 0.0
        else:
            # Current player lost (opponent just won on last move)
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
        if score > best:
            best = score
        if score > alpha:
            alpha = score
        if alpha >= beta:
            break  # Beta cut-off

    return best


def get_best_move_alphabeta(root_state: TogyzkumalakState, root_player: int,
                             max_time_seconds: float = 3.0) -> int:
    """
    Iterative deepening alpha-beta search.
    Searches deeper and deeper until the time budget runs out,
    always returning the best move found so far.
    """
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
            # Move best move to front — dramatically improves pruning next iteration
            moves = [best_move] + [m for m in moves if m != best_move]

            # If we found a forced win/loss, no need to go deeper
            if abs(best_score) > 9000:
                break

        except _Timeout:
            break

    return best_move


# ─────────────────────────────────────────────────────────────────────────────
# MCTS (kept for hint generator which needs fast ~0.5s response)
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

        # Simulation
        sim = state.clone()
        for _ in range(150):
            if sim.isGameOver:
                break
            moves = sim.getPossibleMoves(sim.currentPlayer)
            if not moves:
                break
            chosen = None
            if random.random() < 0.8:
                os = (1 - sim.currentPlayer) * 9
                best_m, best_cap = None, -1
                for m in moves:
                    stones = sim.board[m]
                    if not stones:
                        continue
                    land = (m + 1) % 18 if stones == 1 else (m + stones - 1) % 18
                    if land in sim.tuzdyks:
                        continue
                    if os <= land < os + 9:
                        fut = sim.board[land] + 1
                        if fut % 2 == 0 and fut > best_cap:
                            best_cap = fut
                            best_m = m
                chosen = best_m
            if chosen is None:
                chosen = random.choice(moves)
            sim.makeMove(chosen)

        # Evaluate
        if sim.isGameOver:
            if sim.winner == root_player:
                result = 1.0
            elif sim.winner == -1:
                result = 0.5
            else:
                result = 0.0
        else:
            my = sim.kazans[root_player]
            op = sim.kazans[1 - root_player]
            diff = (my - op) * 2.0
            diff += (int(sim.tuzdyks[root_player] != -1) - int(sim.tuzdyks[1 - root_player] != -1)) * 20
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
# PUBLIC API (called by app.py)
# ─────────────────────────────────────────────────────────────────────────────

OPENING_BOOK = {
    # (total_stones, currentPlayer): [best moves]
    (162, 0): [6, 8],
    (162, 1): [9, 11, 13],
}


def get_best_move_mcts(root_state: TogyzkumalakState, root_player: int,
                        iterations: int = 20000,
                        max_time_seconds: float = 3.0) -> int:
    """
    Main entry point. Uses Alpha-Beta for strong play.
    Falls back to MCTS for very fast (hint) calls under 1 second.
    """
    # Opening book
    key = (sum(root_state.board), root_state.currentPlayer)
    if key in OPENING_BOOK:
        return random.choice(OPENING_BOOK[key])

    # Fast hint calls → use MCTS (Alpha-Beta is too slow at <0.5s)
    if max_time_seconds < 1.0:
        return _mcts(root_state, root_player, iterations, max_time_seconds)

    # Full strength → Alpha-Beta
    return get_best_move_alphabeta(root_state, root_player, max_time_seconds)
