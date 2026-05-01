import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

_client = None

def _get_client():
    global _client
    if _client is None:
        _client = OpenAI(
            api_key=os.environ.get("GROQ_API_KEY"),
            base_url="https://api.groq.com/openai/v1",
        )
    return _client


def generate_hint(board_state, best_move_index):
    try:
        if not os.environ.get("GROQ_API_KEY"):
            return "Кешіріңіз, AI көмекшісі баптаулары толық емес (GROQ_API_KEY жоқ)."

        client = _get_client()

        # Determine if it's white (0) or black (1) moving
        player = "Ақ" if board_state.currentPlayer == 0 else "Қара"
        
        # Prepare board text for the prompt
        board_text = f"Кезек: {player}\nОтаулардағы тастар саны (0-17 индекстері):\n"
        for i, count in enumerate(board_state.board):
            board_text += f"Отау {i}: {count} тас\n"

        prompt = f"""
Сен - Тоғызқұмалақ ойыны бойынша тәжірибелі жаттықтырушысың. Оқушыға көмектесуге арналғансың.
Сенің мақсатыңыз - оқушыға дұрыс жүрісті тікелей айтпай, стратегиялық тұрғыдан бағыт-бағдар беру.

Ережелер:
- Жауабың қазақ немесе орыс тілінде болуы керек.
- Тікелей '{best_move_index} отауымен жүр' деп айтуға ҚАТАҢ ТЫЙЫМ САЛЫНАДЫ.
- Оның орнына оқушыға назар аудару керек аймақты көрсетіп, ойлануға итермеле (мысалы, 'Қарсыластың отауында жұп сан шығаруға тырыс', 'Өз жағыңда тастар көп жиналып қалды' немесе 'Тұздық алу мүмкіндігін қарастыр').
- Жауабың қысқа әрі түсінікті (1-3 сөйлем) болсын.

Қазіргі тақта жағдайы:
{board_text}

Қазандар: Ақ: {board_state.kazans[0]}, Қара: {board_state.kazans[1]}
Тұздықтар: Ақ: {board_state.tuzdyks[0]}, Қара: {board_state.tuzdyks[1]}

(Саған құпия ақпарат: ең үздiк жүріс - {best_move_index} отауы. Осы жүріске апаратын логиканы тұспалдап түсіндір).
"""

        response = client.chat.completions.create(
            model="llama3-8b-8192", # Using LLaMA 3 via Groq for fast inference
            messages=[
                {"role": "system", "content": "You are a helpful Togyzkumalak coach."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=150,
        )

        hint = response.choices[0].message.content.strip()
        return hint

    except Exception as e:
        print(f"Error generating hint with Groq: {e}")
        return "AI көмекшісі қазір қолжетімсіз. Кейінірек қайталап көріңіз."
