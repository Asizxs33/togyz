import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from game_logic import TogyzkumalakState
from ai import get_best_move_mcts, record_learning
import ai as ai_module
from hint_generator import generate_hint

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

@app.route('/api/learn-status', methods=['GET'])
def learn_status():
    db_url_set = bool(os.environ.get("DATABASE_URL"))
    psycopg_installed = ai_module.psycopg is not None
    db_active = ai_module._db_enabled()

    json_path = ai_module.LEARNING_FILE
    json_exists = os.path.exists(json_path)
    json_entries = 0
    json_size = 0
    if json_exists:
        try:
            json_size = os.path.getsize(json_path)
            data = ai_module._load_learning()
            json_entries = len(data) if isinstance(data, dict) else 0
        except Exception:
            pass

    db_entries = None
    if db_active:
        try:
            import psycopg
            with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
                ai_module._ensure_learning_table(conn)
                with conn.cursor() as cur:
                    cur.execute("SELECT COUNT(*) FROM ai_learning")
                    row = cur.fetchone()
                    db_entries = int(row[0]) if row else 0
        except Exception as exc:
            db_entries = f"error: {exc}"

    return jsonify({
        "storage": "postgres" if db_active else "json_file",
        "database_url_set": db_url_set,
        "psycopg_installed": psycopg_installed,
        "db_active": db_active,
        "db_entries": db_entries,
        "json_path": json_path,
        "json_exists": json_exists,
        "json_size_bytes": json_size,
        "json_entries": json_entries,
    })

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
