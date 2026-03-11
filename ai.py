import math
import random
from game_logic import TogyzkumalakState

class MCTSNode:
    def __init__(self, state: TogyzkumalakState, move=None, parent=None):
        self.state = state
        self.move = move
        self.parent = parent
        self.children = []
        self.wins = 0
        self.visits = 0
        self.untriedMoves = state.getPossibleMoves(state.currentPlayer)

    def get_best_child(self, exploration_param=1.41):
        best_score = float('-inf')
        best_children = []

        for child in self.children:
            exploit = child.wins / child.visits
            explore = exploration_param * math.sqrt(math.log(self.visits) / child.visits)
            score = exploit + explore

            if score > best_score:
                best_score = score
                best_children = [child]
            elif score == best_score:
                best_children.append(child)

        return random.choice(best_children)

def get_best_move_mcts(root_state: TogyzkumalakState, root_player, iterations=3000, max_time_seconds=3.0):
    import time
    root_node = MCTSNode(root_state)

    start_time = time.time()
    iters = 0

    # Stop either when reaching iterations limit or time limit
    while iters < iterations and (time.time() - start_time) < max_time_seconds:
        node = root_node
        state = root_state.clone()

        # 1. Selection
        while len(node.untriedMoves) == 0 and len(node.children) > 0:
            node = node.get_best_child()
            state.makeMove(node.move)

        # 2. Expansion
        if len(node.untriedMoves) > 0:
            move = random.choice(node.untriedMoves)
            node.untriedMoves.remove(move)
            state.makeMove(move)
            child_node = MCTSNode(state.clone(), move, node)
            node.children.append(child_node)
            node = child_node

        # 3. Simulation
        simulation_state = state.clone()
        sim_depth = 0
        while not simulation_state.isGameOver and sim_depth < 100:
            possible_moves = simulation_state.getPossibleMoves(simulation_state.currentPlayer)
            if not possible_moves:
                break
            random_move = random.choice(possible_moves)
            simulation_state.makeMove(random_move)
            sim_depth += 1

        # 4. Backpropagation (with Heuristics)
        if simulation_state.isGameOver:
            if simulation_state.winner == root_player:
                result = 1.0 # Win
            elif simulation_state.winner == -1:
                result = 0.5 # Draw
            else:
                result = 0.0 # Loss
        else:
            # Game didn't finish within random rollout. Use heuristics to estimate win probability.
            my_score = simulation_state.kazans[root_player]
            opp_score = simulation_state.kazans[1 - root_player]
            
            # 81 is winning score. We map the score difference to a probability between 0 and 1
            # If my_score is much higher, result approaches 1.0
            score_diff = my_score - opp_score
            
            # Include tuzdyk advantage: each tuzdyk is worth roughly +5 points
            my_tuzdyks = 1 if simulation_state.tuzdyks[root_player] != -1 else 0
            opp_tuzdyks = 1 if simulation_state.tuzdyks[1 - root_player] != -1 else 0
            score_diff += (my_tuzdyks - opp_tuzdyks) * 5
            
            # Sigmoid-like mapping: Diff of +20 gives high probability (~0.9), 0 gives 0.5
            result = 1.0 / (1.0 + math.exp(-score_diff / 15.0))


        while node is not None:
            node.visits += 1
            node.wins += result
            result = 1.0 - result
            node = node.parent
            
        iters += 1

    if not root_node.children:
        return -1

    # Choose the child with the most visits
    best_child = max(root_node.children, key=lambda c: c.visits)
    return best_child.move
