from flask import Flask, request, jsonify
from flask_cors import CORS
from game_logic import TogyzkumalakState
from ai import get_best_move_mcts
from hint_generator import generate_hint

app = Flask(__name__)
# Enable CORS for the React app port
CORS(app)

@app.route('/api/hint', methods=['POST'])
def get_hint():
    data = request.json
    
    state = TogyzkumalakState()
    state.board = data.get('board', [9]*18)
    state.kazans = data.get('kazans', [0, 0])
    state.tuzdyks = data.get('tuzdyks', [-1, -1])
    state.currentPlayer = data.get('currentPlayer', 1)
    state.isGameOver = data.get('isGameOver', False)

    if state.isGameOver:
        return jsonify({"hint": "Ойын аяқталды!"})

    # Get a quick best move to feed as context for the LLM
    best_move_index = get_best_move_mcts(state, state.currentPlayer, iterations=2000, max_time_seconds=0.5)
    
    hint_text = generate_hint(state, best_move_index)
    return jsonify({"hint": hint_text})

@app.route('/api/best-move', methods=['POST'])
def best_move():
    data = request.json
    
    # Reconstruct state from JSON payload
    state = TogyzkumalakState()
    state.board = data.get('board', [9]*18)
    state.kazans = data.get('kazans', [0, 0])
    state.tuzdyks = data.get('tuzdyks', [-1, -1])
    state.currentPlayer = data.get('currentPlayer', 1)
    state.isGameOver = data.get('isGameOver', False)
    state.winner = data.get('winner', None)

    algorithm = data.get('algorithm', 'mcts')
    
    if state.isGameOver:
        return jsonify({"move": -1, "error": "Game is over"})

    # Determine move
    best_move_index = -1
    if algorithm == 'mcts':
        iterations = data.get('iterations', 20000)
        max_time = data.get('max_time_seconds', 3.0)
        best_move_index = get_best_move_mcts(state, state.currentPlayer, iterations, max_time)
    else:
        # Fallback to pure random if algorithm is weird
        moves = state.getPossibleMoves(state.currentPlayer)
        import random
        if moves:
            best_move_index = random.choice(moves)

    return jsonify({"move": best_move_index})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
