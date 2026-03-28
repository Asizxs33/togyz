from flask import Flask, request, jsonify
from flask_cors import CORS
from game_logic import TogyzkumalakState
from ai import get_best_move_alphabeta

app = Flask(__name__)
# Enable CORS for the React app port
CORS(app)

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
    max_time = data.get('max_time_seconds', 3.0)
    best_move_index = get_best_move_alphabeta(state, state.currentPlayer, max_time)

    return jsonify({"move": best_move_index})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
