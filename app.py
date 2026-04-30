from flask import Flask, request, jsonify
from flask_cors import CORS
from game_logic import TogyzkumalakState
from ai import get_best_move_mcts, record_learning

app = Flask(__name__)
# Enable CORS for the React app port
CORS(app)

@app.route('/api/learn', methods=['POST'])
def learn():
    data = request.json or {}
    samples = data.get('samples', [])
    winner = data.get('winner', None)
    ai_player = data.get('aiPlayer', 1)

    if not isinstance(samples, list):
        return jsonify({"error": "samples must be a list"}), 400

    learned = record_learning(samples, winner, ai_player)
    return jsonify({"ok": True, "learned": learned})

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
        # In Python, we can comfortably do 10,000+ iterations because it's much faster
        # 30,000 gives a very strong bot.
        iterations = data.get('iterations', 20000)
        max_time_seconds = data.get('max_time_seconds', 3.0)
        best_move_index = get_best_move_mcts(state, state.currentPlayer, iterations, max_time_seconds=max_time_seconds)
    else:
        # Fallback to pure random if algorithm is weird
        moves = state.getPossibleMoves(state.currentPlayer)
        import random
        if moves:
            best_move_index = random.choice(moves)

    return jsonify({"move": best_move_index})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
