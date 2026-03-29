from flask import Flask, request, jsonify
from flask_cors import CORS
from game_logic import TogyzkumalakState
from ai import get_best_move_alphabeta

app = Flask(__name__)
# Enable CORS for the React app port
CORS(app)

@app.route('/api/best-move', methods=['POST'])
def best_move():
    try:
        data = request.json

        state = TogyzkumalakState()
        state.board        = data.get('board',         [9]*18)
        state.kazans       = data.get('kazans',        [0, 0])
        raw_tuz            = data.get('tuzdyks',       [-1, -1])
        state.tuzdyks      = [t if t is not None else -1 for t in raw_tuz]
        state.currentPlayer = data.get('currentPlayer', 1)
        state.isGameOver   = data.get('isGameOver',    False)
        state.winner       = data.get('winner',        None)

        if state.isGameOver:
            return jsonify({"move": -1})

        max_time = float(data.get('max_time_seconds', 3.0))
        move = get_best_move_alphabeta(state, state.currentPlayer, max_time)
        return jsonify({"move": move})

    except Exception as e:
        import traceback
        print(traceback.format_exc(), flush=True)
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
