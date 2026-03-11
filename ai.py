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

        # 3. Simulation (Heuristic Rollout)
        simulation_state = state.clone()
        sim_depth = 0
        while not simulation_state.isGameOver and sim_depth < 100:
            possible_moves = simulation_state.getPossibleMoves(simulation_state.currentPlayer)
            if not possible_moves:
                break
                
            # Heuristic Rollout: 80% chance to pick a move that captures, 20% random
            chosen_move = None
            if random.random() < 0.8:
                # Try to find a capturing move quickly
                best_sim_move = None
                max_cap = -1
                for m in possible_moves:
                    stones = simulation_state.board[m]
                    if stones == 0: continue
                    # Approximate landing pocket:
                    landing = (m + stones) % 18 if stones > 1 else (m + 1) % 18
                    # If landing on opponent side and making it even:
                    opp_start = (1 - simulation_state.currentPlayer) * 9
                    if opp_start <= landing < opp_start + 9:
                        future_stones = simulation_state.board[landing] + 1
                        if future_stones % 2 == 0:
                            if future_stones > max_cap:
                                max_cap = future_stones
                                best_sim_move = m
                if best_sim_move is not None:
                    chosen_move = best_sim_move

            if chosen_move is None:
                chosen_move = random.choice(possible_moves)
                
            simulation_state.makeMove(chosen_move)
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
            
            # Include tuzdyk advantage: each tuzdyk is worth WAY MORE (+50 points)
            # This forces the MCTS to prioritize getting a star over everything else
            my_tuzdyks = 1 if simulation_state.tuzdyks[root_player] != -1 else 0
            opp_tuzdyks = 1 if simulation_state.tuzdyks[1 - root_player] != -1 else 0
            score_diff += (my_tuzdyks - opp_tuzdyks) * 50
            
            # Principles 4, 5, 6: Advanced evaluating
            my_start = root_player * 9
            opp_start = (1 - root_player) * 9
            
            # Principle 4: Vulnerability (minimize opponent's even pockets, maximize own opportunities)
            my_vulnerability = sum(1 for i in range(my_start, my_start + 9) if simulation_state.board[i] > 0 and simulation_state.board[i] % 2 == 0)
            opp_vulnerability = sum(1 for i in range(opp_start, opp_start + 9) if simulation_state.board[i] > 0 and simulation_state.board[i] % 2 == 0)
            score_diff += (opp_vulnerability - my_vulnerability) * 3
            
            # Principle 6: Central control (pockets 4,5,6 -> indices 3,4,5 from start)
            my_center_stones = sum(simulation_state.board[my_start + i] for i in range(3, 6))
            opp_center_stones = sum(simulation_state.board[opp_start + i] for i in range(3, 6))
            score_diff += (my_center_stones - opp_center_stones) * 0.5
            
            # Principle 5: Mobility (Atsyrau)
            my_mobility = sum(simulation_state.board[my_start:my_start+9])
            opp_mobility = sum(simulation_state.board[opp_start:opp_start+9])
            score_diff += (my_mobility - opp_mobility) * 0.5
            
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
