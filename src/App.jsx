import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { TogyzkumalakState } from './logic/Togyzkumalak';
import { calculateBestMove } from './logic/AI';
import { Otau } from './components/Otau';
import { Kazan } from './components/Kazan';

const ANIM_DELAY = 180;

const difficultyLabels = {
    2: 'Легкий',
    3: 'Средний',
    5: 'Сложный',
    mcts: 'MCTS AI',
};

function App() {
    const [gameState, setGameState] = useState(new TogyzkumalakState());
    const [difficulty, setDifficulty] = useState(3);
    const [isAiThinking, setIsAiThinking] = useState(false);

    const [isAnimating, setIsAnimating] = useState(false);
    const [animBoard, setAnimBoard] = useState(null);
    const [animKazans, setAnimKazans] = useState(null);
    const [animTuzdyks, setAnimTuzdyks] = useState(null);
    const [highlightIndex, setHighlightIndex] = useState(-1);
    const [animPhase, setAnimPhase] = useState(null);
    const animTimerRef = useRef(null);

    const displayBoard = animBoard || gameState.board;
    const displayKazans = animKazans || gameState.kazans;
    const displayTuzdyks = animTuzdyks || gameState.tuzdyks;

    const playAnimation = useCallback((frames, finalState) => {
        setIsAnimating(true);
        let frameIndex = 0;

        const playNextFrame = () => {
            if (frameIndex < frames.length) {
                const frame = frames[frameIndex];
                setAnimBoard([...frame.board]);
                setAnimKazans([...frame.kazans]);
                setAnimTuzdyks([...frame.tuzdyks]);
                setHighlightIndex(frame.highlightIndex);
                setAnimPhase(frame.phase);
                frameIndex++;
                animTimerRef.current = setTimeout(playNextFrame, ANIM_DELAY);
            } else {
                setAnimBoard(null);
                setAnimKazans(null);
                setAnimTuzdyks(null);
                setHighlightIndex(-1);
                setAnimPhase(null);
                setIsAnimating(false);
                setGameState(finalState);
            }
        };

        playNextFrame();
    }, []);

    useEffect(() => {
        if (gameState.currentPlayer === 1 && !gameState.isGameOver && !isAiThinking && !isAnimating) {
            setIsAiThinking(true);

            const handleBestMove = (bestMove) => {
                if (bestMove !== -1) {
                    const frames = gameState.getMoveSteps(bestMove);
                    const finalState = gameState.clone();
                    finalState.makeMove(bestMove);

                    setIsAiThinking(false);
                    playAnimation(frames, finalState);
                } else {
                    setIsAiThinking(false);
                }
            };

            if (difficulty === 'mcts') {
                const payload = {
                    board: gameState.board,
                    kazans: gameState.kazans,
                    tuzdyks: gameState.tuzdyks,
                    currentPlayer: gameState.currentPlayer,
                    isGameOver: gameState.isGameOver,
                    winner: gameState.winner,
                    algorithm: 'mcts',
                    iterations: 20000,
                    max_time_seconds: 3.0,
                };

                fetch('https://togyz.onrender.com/api/best-move', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload),
                })
                    .then((res) => res.json())
                    .then((data) => {
                        handleBestMove(data.move);
                    })
                    .catch((err) => {
                        console.error('Failed to reach Python backend:', err);
                        alert('Не удалось подключиться к AI-серверу. Включен локальный сложный режим.');
                        const fallbackMove = calculateBestMove(gameState, 5, 1, 'minimax');
                        setTimeout(() => handleBestMove(fallbackMove), 400);
                    });
            } else {
                setTimeout(() => {
                    const depth = parseInt(difficulty);
                    const bestMove = calculateBestMove(gameState, depth, 1, 'minimax');
                    handleBestMove(bestMove);
                }, 400);
            }
        }
    }, [gameState.currentPlayer, gameState.isGameOver, isAiThinking, isAnimating, difficulty, gameState, playAnimation]);

    const handlePocketClick = useCallback((index) => {
        if (gameState.currentPlayer === 0 && !gameState.isGameOver && !isAiThinking && !isAnimating) {
            if (gameState.isValidMove(0, index)) {
                const frames = gameState.getMoveSteps(index);
                const finalState = gameState.clone();
                finalState.makeMove(index);
                playAnimation(frames, finalState);
            }
        }
    }, [gameState, isAiThinking, isAnimating, playAnimation]);

    const handleRestart = () => {
        if (animTimerRef.current) clearTimeout(animTimerRef.current);
        setAnimBoard(null);
        setAnimKazans(null);
        setAnimTuzdyks(null);
        setHighlightIndex(-1);
        setAnimPhase(null);
        setIsAnimating(false);
        setIsAiThinking(false);
        setGameState(new TogyzkumalakState());
    };

    const topRowIndices = [17, 16, 15, 14, 13, 12, 11, 10, 9];
    const bottomRowIndices = [0, 1, 2, 3, 4, 5, 6, 7, 8];

    const isBusy = isAnimating || isAiThinking;
    const winnerText = gameState.winner === 0
        ? 'Вы выиграли'
        : gameState.winner === 1
            ? 'Компьютер выиграл'
            : 'Ничья';

    const statusText = useMemo(() => {
        if (gameState.isGameOver) return 'Партия завершена';
        if (isAiThinking) return 'Компьютер думает';
        if (isAnimating) return 'Ход выполняется';
        return gameState.currentPlayer === 0 ? 'Ваш ход' : 'Ход компьютера';
    }, [gameState.currentPlayer, gameState.isGameOver, isAiThinking, isAnimating]);

    return (
        <main className="app-shell">
            <section className="hero-bar">
                <div className="brand-block">
                    <span className="eyebrow">Классическая казахская игра</span>
                    <h1>Тоғызқұмалақ</h1>
                    <p>Играйте против компьютера, следите за қазанами и создавайте тұздық в нужный момент.</p>
                </div>

                <div className="control-panel" aria-label="Настройки игры">
                    <label className="select-field">
                        <span>Сложность</span>
                        <select value={difficulty} onChange={(e) => setDifficulty(e.target.value)} disabled={isBusy}>
                            <option value="2">Легкий</option>
                            <option value="3">Средний</option>
                            <option value="5">Сложный</option>
                            <option value="mcts">MCTS AI</option>
                        </select>
                    </label>
                    <button className="primary-action" onClick={handleRestart}>Новая игра</button>
                </div>
            </section>

            <section className="match-panel">
                <div className="score-strip">
                    <div className={`player-card computer ${gameState.currentPlayer === 1 && !isBusy ? 'active' : ''}`}>
                        <span className="player-label">Компьютер</span>
                        <strong>{displayKazans[1]}</strong>
                    </div>
                    <div className="status-card">
                        <span>{statusText}</span>
                        <strong>{difficultyLabels[difficulty]}</strong>
                    </div>
                    <div className={`player-card human ${gameState.currentPlayer === 0 && !isBusy ? 'active' : ''}`}>
                        <span className="player-label">Вы</span>
                        <strong>{displayKazans[0]}</strong>
                    </div>
                </div>

                <div className="board-scroll" aria-label="Горизонтальная игровая доска">
                    <div className="board-container">
                        <div className="side-label opponent-label">Қарсылас</div>
                        <div className="otau-row top-row">
                            {topRowIndices.map((index) => (
                                <Otau
                                    key={index}
                                    index={index}
                                    count={displayBoard[index]}
                                    isEnabled={false}
                                    isTuzdyk={displayTuzdyks.includes(index)}
                                    owner={displayTuzdyks.indexOf(index)}
                                    onClick={() => {}}
                                    label={index - 8}
                                    isHighlighted={highlightIndex === index}
                                    animPhase={highlightIndex === index ? animPhase : null}
                                />
                            ))}
                        </div>

                        <div className="kazan-section">
                            <Kazan player={1} count={displayKazans[1]} isActive={gameState.currentPlayer === 1 && !isBusy} />
                            <Kazan player={0} count={displayKazans[0]} isActive={gameState.currentPlayer === 0 && !isBusy} />
                        </div>

                        <div className="otau-row bottom-row">
                            {bottomRowIndices.map((index) => (
                                <Otau
                                    key={index}
                                    index={index}
                                    count={displayBoard[index]}
                                    isEnabled={gameState.currentPlayer === 0 && !isBusy}
                                    isTuzdyk={displayTuzdyks.includes(index)}
                                    owner={displayTuzdyks.indexOf(index)}
                                    onClick={handlePocketClick}
                                    label={index + 1}
                                    isHighlighted={highlightIndex === index}
                                    animPhase={highlightIndex === index ? animPhase : null}
                                />
                            ))}
                        </div>
                        <div className="side-label player-label-board">Ойыншы</div>
                    </div>
                </div>
            </section>

            <div className={`modal-overlay ${gameState.isGameOver ? 'visible' : ''}`}>
                <div className="modal-content">
                    <span className="modal-kicker">Игра окончена</span>
                    <h2>{winnerText}</h2>
                    <p>Счет: {gameState.kazans[0]} - {gameState.kazans[1]}</p>
                    <button className="primary-action" onClick={handleRestart}>Сыграть еще</button>
                </div>
            </div>
        </main>
    );
}

export default App;
