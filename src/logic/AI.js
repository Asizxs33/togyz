export function calculateBestMove(state, depth, player, algorithm = 'minimax') {
    if (algorithm === 'mcts') {
        // MCTS uses iterations instead of depth. Let's do 2000 iterations for a JS browser
        return getBestMoveMCTS(state, player, 2000); 
    }

    // --- STANDARD MINIMAX WITH ITERATIVE DEEPENING & TRANSPOSITION TABLE ---
    let possibleMoves = state.getPossibleMoves(player);
    if (possibleMoves.length === 0) return -1;
    
    let totalStones = 0;
    for (let i = 0; i < 18; i++) totalStones += state.board[i];
    
    // 1. OPENING BOOK: If it's the very first move of the game, play standard theoretical openings
    if (totalStones === 162 && state.currentPlayer === 0) {
        const standardOpenings = [6, 8]; // Pit 7 or Pit 9 (0-indexed) are best first moves in theory
        return standardOpenings[Math.floor(Math.random() * standardOpenings.length)];
    }
    
    // 2. ITERATIVE DEEPENING TIME CONTROL (IDTC)
    const startTime = Date.now();
    const timeLimitMs = 1500; // 1.5 seconds max limit to avoid freezing the React UI
    
    let activeDepth = parseInt(depth);
    if (activeDepth >= 4) {
        if (totalStones < 50) activeDepth += 1;
        if (totalStones < 30) activeDepth += 2;
        if (totalStones < 15) activeDepth += 5; // Search extremely deep in endgame
    }
    
    const tt = new Map(); // Transposition Table for caching evaluated positions
    let bestMovesArr = possibleMoves; // Default safety fallback
    let absoluteBestEval = -Infinity;
    
    let currentDepth = 1;

    // Move Ordering: Evaluate pockets with the most stones first to maximize alpha-beta pruning
    possibleMoves.sort((a, b) => state.board[b] - state.board[a]);
    
    while (currentDepth <= activeDepth) {
        let maxEval = -Infinity;
        let currentDepthBestMoves = [];
        
        for (let move of possibleMoves) {
            if (Date.now() - startTime >= timeLimitMs && currentDepth > 3) {
                break; // Out of time, rely on previous depth's fully completed results
            }
            
            const newState = state.clone();
            newState.makeMove(move);
            const isNextMaximizing = newState.currentPlayer === player;
            
            // Call Minimax and pass down TT and Time limits
            const evalScore = minimax(newState, currentDepth - 1, -Infinity, Infinity, isNextMaximizing, player, tt, startTime, timeLimitMs);
            
            if (evalScore > maxEval) {
                maxEval = evalScore;
                currentDepthBestMoves = [move];
            } else if (evalScore === maxEval) {
                currentDepthBestMoves.push(move);
            }
        }
        
        if (currentDepthBestMoves.length > 0) {
            bestMovesArr = currentDepthBestMoves;
            absoluteBestEval = maxEval;
            
            // Re-order the best moves to the front for the next iteration step (Principal Variation ordering logic)
            const bestMoveSet = new Set(bestMovesArr);
            possibleMoves.sort((a, b) => (bestMoveSet.has(b) ? 1 : 0) - (bestMoveSet.has(a) ? 1 : 0));
        }
        
        if (absoluteBestEval > 9000) break; // Found a winning path!
        if (Date.now() - startTime >= timeLimitMs) break; // Out of time, stop iterating deeper
        
        currentDepth++;
    }
    
    // Pick randomly among the absolute best moves to add variety
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

// Hashing function for Transposition Tables
function hashState(state, isMaximizing) {
    // Unique string representing the game board state. 
    return state.board.join(',') + '|' + state.kazans.join(',') + '|' + state.tuzdyks.join(',') + '|' + isMaximizing;
}

function minimax(state, depth, alpha, beta, isMaximizing, player, tt, startTime, timeLimitMs) {
    const hash = hashState(state, isMaximizing);
    
    // Transposition Table Lookup
    if (tt.has(hash)) {
        const entry = tt.get(hash);
        if (entry.depth >= depth) {
            if (entry.flag === 'EXACT') return entry.value;
            if (entry.flag === 'LOWERBOUND') alpha = Math.max(alpha, entry.value);
            if (entry.flag === 'UPPERBOUND') beta = Math.min(beta, entry.value);
            if (alpha >= beta) return entry.value;
        }
    }

    if (depth === 0 || state.isGameOver) {
        const val = evaluateBoard(state, player);
        // Cache leaf nodes
        tt.set(hash, { value: val, depth, flag: 'EXACT' });
        return val;
    }
    
    // Time constraint bailout
    if (Date.now() - startTime > timeLimitMs) {
        return evaluateBoard(state, player);
    }
    
    // Move Ordering: Evaluate pockets with the most stones first to maximize alpha-beta pruning
    const possibleMoves = state.getPossibleMoves(state.currentPlayer);
    possibleMoves.sort((a, b) => state.board[b] - state.board[a]);
    
    let bestVal = isMaximizing ? -Infinity : Infinity;
    let originalAlpha = alpha;

    for (let move of possibleMoves) {
        const newState = state.clone();
        newState.makeMove(move);
        const isNextMaximizing = newState.currentPlayer === player;
        const ev = minimax(newState, depth - 1, alpha, beta, isNextMaximizing, player, tt, startTime, timeLimitMs);
        
        if (isMaximizing) {
            bestVal = Math.max(bestVal, ev);
            alpha = Math.max(alpha, bestVal);
        } else {
            bestVal = Math.min(bestVal, ev);
            beta = Math.min(beta, bestVal);
        }
        if (beta <= alpha) break; // Alpha Beta Pruning
    }
    
    // Save evaluated node to Transposition Table
    let flag = 'EXACT';
    if (bestVal <= originalAlpha) flag = 'UPPERBOUND';
    else if (bestVal >= beta) flag = 'LOWERBOUND';
    
    tt.set(hash, { value: bestVal, depth, flag });
    
    return bestVal;
}

function evaluateBoard(state, player) {
    if (state.isGameOver) {
        if (state.winner === player) return 10000;
        if (state.winner === 1 - player) return -10000;
        return 0; // Draw
    }
    
    const opponent = 1 - player;
    
    // Weights corresponding to Togyzkumalak principles
    const WEIGHT_K = 10;   // Principle 1: Maximize Kazans
    const WEIGHT_T = 500;  // Principle 2: Tuzdyk (Extremely powerful)
    const WEIGHT_V = 3;    // Principle 4: Vulnerability (Limit opponent's even pockets)
    const WEIGHT_M = 0.5;  // Principle 5: Mobility (Avoid Atsyrau)
    const WEIGHT_C = 0.5;  // Principle 6: Central control (pockets 4,5,6)
    
    // 1. K (Kazan)
    const kazanScore = (state.kazans[player] - state.kazans[opponent]) * WEIGHT_K;
    
    // 2. T (Tuzdyk)
    let tuzdykScore = 0;
    if (state.tuzdyks[player] !== -1) tuzdykScore += WEIGHT_T;
    if (state.tuzdyks[opponent] !== -1) tuzdykScore -= WEIGHT_T;
    
    // 3. M & C (Mobility & Center Control)
    let myMobility = 0;
    let myCenter = 0;
    const myStart = player === 0 ? 0 : 9;
    const myEnd = player === 0 ? 8 : 17;
    for (let i = myStart; i <= myEnd; i++) {
        myMobility += state.board[i];
        if (i >= myStart + 3 && i <= myStart + 5) {
            myCenter += state.board[i];
        }
    }
    
    let oppMobility = 0;
    let oppCenter = 0;
    const oppStart = opponent === 0 ? 0 : 9;
    const oppEnd = opponent === 0 ? 8 : 17;
    for (let i = oppStart; i <= oppEnd; i++) {
        oppMobility += state.board[i];
        if (i >= oppStart + 3 && i <= oppStart + 5) {
            oppCenter += state.board[i];
        }
    }
    const mobilityScore = (myMobility - oppMobility) * WEIGHT_M;
    const centerScore = (myCenter - oppCenter) * WEIGHT_C;
    
    // 4. V (Vulnerability - count my even pockets vs their even pockets)
    // Principle 4: limit opponent win probability by limiting evens
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
    return kazanScore + tuzdykScore + mobilityScore + vulnerabilityScore + centerScore;
}
