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

def get_best_move_mcts(root_state: TogyzkumalakState, root_player, iterations=10000):
    root_node = MCTSNode(root_state)

    for i in range(iterations):
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

        # 4. Backpropagation
        won = False
        if simulation_state.winner == root_player:
            won = True
        
        if simulation_state.winner is None or simulation_state.winner == -1:
            result = 0.5
        else:
            result = 1.0 if won else 0.0

        while node is not None:
            node.visits += 1
            node.wins += result
            result = 1.0 - result
            node = node.parent

    if not root_node.children:
        return -1

    # Choose the child with the most visits
    best_child = max(root_node.children, key=lambda c: c.visits)
    return best_child.move
