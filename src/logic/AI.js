export function calculateBestMove(state, depth, player, algorithm = 'minimax') {
    if (algorithm === 'mcts') {
        // MCTS uses iterations instead of depth. Let's do 2000 iterations for a JS browser
        return getBestMoveMCTS(state, player, 2000); 
    }

    // --- STANDARD MINIMAX ---
    let maxEval = -Infinity;
    
    const possibleMoves = state.getPossibleMoves(player);
    if (possibleMoves.length === 0) return -1;
    
    const bestMovesArr = [];

    for (let move of possibleMoves) {
        const newState = state.clone();
        newState.makeMove(move);
        
        const isNextMaximizing = newState.currentPlayer === player;
        
        const evalScore = minimax(newState, depth - 1, -Infinity, Infinity, isNextMaximizing, player);
        
        if (evalScore > maxEval) {
            maxEval = evalScore;
            bestMovesArr.length = 0; // clear array
            bestMovesArr.push(move);
        } else if (evalScore === maxEval) {
            bestMovesArr.push(move);
        }
    }
    
    // Pick random among the best moves to add variety
    return bestMovesArr[Math.floor(Math.random() * bestMovesArr.length)];
}

// ==========================================
// MONTE CARLO TREE SEARCH (MCTS)
// ==========================================

class MCTSNode {
    constructor(state, move = null, parent = null) {
        this.state = state;
        this.move = move;
        this.parent = parent;
        this.children = [];
        this.wins = 0;
        this.visits = 0;
        this.untriedMoves = state.getPossibleMoves(state.currentPlayer);
    }

    getBestChild(explorationParam = 1.41) {
        let bestScore = -Infinity;
        let bestChildren = [];

        for (let child of this.children) {
            const exploit = child.wins / child.visits;
            const explore = explorationParam * Math.sqrt(Math.log(this.visits) / child.visits);
            const score = exploit + explore;

            if (score > bestScore) {
                bestScore = score;
                bestChildren = [child];
            } else if (score === bestScore) {
                bestChildren.push(child);
            }
        }
        return bestChildren[Math.floor(Math.random() * bestChildren.length)];
    }
}

function getBestMoveMCTS(rootState, rootPlayer, iterations) {
    const rootNode = new MCTSNode(rootState);

    for (let i = 0; i < iterations; i++) {
        let node = rootNode;
        let state = rootState.clone();

        // 1. Selection
        while (node.untriedMoves.length === 0 && node.children.length > 0) {
            node = node.getBestChild();
            state.makeMove(node.move);
        }

        // 2. Expansion
        if (node.untriedMoves.length > 0) {
            const moveIndex = Math.floor(Math.random() * node.untriedMoves.length);
            const move = node.untriedMoves.splice(moveIndex, 1)[0];
            state.makeMove(move);
            const childNode = new MCTSNode(state.clone(), move, node);
            node.children.push(childNode);
            node = childNode;
        }

        // 3. Simulation (Playout using light heuristics instead of pure random to save time)
        let simulationState = state.clone();
        let simDepth = 0; // limit depth to avoid infinite loops in bad states
        while (!simulationState.isGameOver && simDepth < 100) {
            const possibleMoves = simulationState.getPossibleMoves(simulationState.currentPlayer);
            if (possibleMoves.length === 0) break;
            const randomMove = possibleMoves[Math.floor(Math.random() * possibleMoves.length)];
            simulationState.makeMove(randomMove);
            simDepth++;
        }

        // 4. Backpropagation
        let won = false;
        if (simulationState.winner === rootPlayer) won = true;
        // Draw is counted as 0.5 win
        let result = simulationState.winner === null ? 0.5 : (won ? 1 : 0);

        while (node !== null) {
            node.visits += 1;
            node.wins += result;
            // invert result for opponent's nodes
            result = 1 - result; 
            node = node.parent;
        }
    }

    if (rootNode.children.length === 0) return -1;
    
    // Choose the child with the most visits
    let bestChild = rootNode.children[0];
    for (let child of rootNode.children) {
        if (child.visits > bestChild.visits) {
            bestChild = child;
        }
    }
    
    return bestChild.move;
}

function minimax(state, depth, alpha, beta, isMaximizing, player) {
    if (depth === 0 || state.isGameOver) {
        return evaluateBoard(state, player);
    }
    
    const possibleMoves = state.getPossibleMoves(state.currentPlayer);
    
    if (isMaximizing) {
        let maxEval = -Infinity;
        for (let move of possibleMoves) {
            const newState = state.clone();
            newState.makeMove(move);
            const isNextMaximizing = newState.currentPlayer === player;
            const ev = minimax(newState, depth - 1, alpha, beta, isNextMaximizing, player);
            maxEval = Math.max(maxEval, ev);
            alpha = Math.max(alpha, ev);
            if (beta <= alpha) break;
        }
        return maxEval;
    } else {
        let minEval = Infinity;
        for (let move of possibleMoves) {
            const newState = state.clone();
            newState.makeMove(move);
            const isNextMaximizing = newState.currentPlayer === player;
            const ev = minimax(newState, depth - 1, alpha, beta, isNextMaximizing, player);
            minEval = Math.min(minEval, ev);
            beta = Math.min(beta, ev);
            if (beta <= alpha) break; // Alpha Beta Pruning
        }
        return minEval;
    }
}

function evaluateBoard(state, player) {
    if (state.isGameOver) {
        if (state.winner === player) return 10000;
        if (state.winner === 1 - player) return -10000;
        return 0; // Draw
    }
    
    const opponent = 1 - player;
    
    // Weights for the heuristic function
    const WEIGHT_K = 10;   // Kazan stones (Most important)
    const WEIGHT_T = 500;  // Having a Tuzdyk is extremely powerful! Prioritize making stars.
    const WEIGHT_M = 1;    // Mobility (having stones on your side to make moves)
    const WEIGHT_V = 2;    // Vulnerability (opponent has pockets with even stones that can be captured)
    
    // 1. K (Kazan)
    const kazanScore = (state.kazans[player] - state.kazans[opponent]) * WEIGHT_K;
    
    // 2. T (Tuzdyk)
    let tuzdykScore = 0;
    if (state.tuzdyks[player] !== -1) tuzdykScore += WEIGHT_T;
    if (state.tuzdyks[opponent] !== -1) tuzdykScore -= WEIGHT_T;
    
    // 3. M (Mobility)
    let myMobility = 0;
    const myStart = player === 0 ? 0 : 9;
    const myEnd = player === 0 ? 8 : 17;
    for (let i = myStart; i <= myEnd; i++) {
        myMobility += state.board[i];
    }
    
    let oppMobility = 0;
    const oppStart = opponent === 0 ? 0 : 9;
    const oppEnd = opponent === 0 ? 8 : 17;
    for (let i = oppStart; i <= oppEnd; i++) {
        oppMobility += state.board[i];
    }
    const mobilityScore = (myMobility - oppMobility) * WEIGHT_M;
    
    // 4. V (Vulnerability - count my even pockets vs their even pockets)
    // A pocket on my side with an even number is vulnerable to the opponent.
    let myVulnerability = 0;
    for (let i = myStart; i <= myEnd; i++) {
        if (state.board[i] > 0 && state.board[i] % 2 === 0) {
            myVulnerability++;
        }
    }
    
    let oppVulnerability = 0;
    for (let i = oppStart; i <= oppEnd; i++) {
        if (state.board[i] > 0 && state.board[i] % 2 === 0) {
            oppVulnerability++;
        }
    }
    
    // If opponent has more vulnerable pockets, it's good for me.
    const vulnerabilityScore = (oppVulnerability - myVulnerability) * WEIGHT_V;
    
    // Total Heuristic Evaluation
    return kazanScore + tuzdykScore + mobilityScore + vulnerabilityScore;
}
