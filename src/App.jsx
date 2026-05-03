import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { TogyzkumalakState } from './logic/Togyzkumalak';
import { calculateBestMove } from './logic/AI';
import { Otau } from './components/Otau';
import { Kazan } from './components/Kazan';

const ANIM_DELAY = 180;
const AI_MODE_LABEL = 'MCTS AI';
const AI_API_BASE = 'https://togyz.onrender.com';

function snapshotState(state) {
    return {
        board: [...state.board],
        kazans: [...state.kazans],
        tuzdyks: [...state.tuzdyks],
        player: state.currentPlayer,
    };
}

function App() {
    const [gameState, setGameState] = useState(new TogyzkumalakState());
    const [isAiThinking, setIsAiThinking] = useState(false);
    const [moveHistory, setMoveHistory] = useState([]);
    const [humanPlayer, setHumanPlayer] = useState(0);
    const aiPlayer = 1 - humanPlayer;

    const [isAnimating, setIsAnimating] = useState(false);
    const [animBoard, setAnimBoard] = useState(null);
    const [animKazans, setAnimKazans] = useState(null);
    const [animTuzdyks, setAnimTuzdyks] = useState(null);
    const [highlightIndex, setHighlightIndex] = useState(-1);
    const [animPhase, setAnimPhase] = useState(null);
    const animTimerRef = useRef(null);
    const aiMoveHistoryRef = useRef([]);
    const learningSentRef = useRef(false);

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
        if (gameState.currentPlayer === aiPlayer && !gameState.isGameOver && !isAiThinking && !isAnimating) {
            setIsAiThinking(true);

            const handleBestMove = (bestMove) => {
                if (bestMove !== -1) {
                    aiMoveHistoryRef.current.push({
                        ...snapshotState(gameState),
                        move: bestMove,
                    });

                    const frames = gameState.getMoveSteps(bestMove);
                    const finalState = gameState.clone();
                    const moveResult = finalState.makeMove(bestMove);
                    if (moveResult && moveResult.notation) {
                        setMoveHistory((prev) => [...prev, { player: aiPlayer, notation: moveResult.notation }]);
                    }

                    setIsAiThinking(false);
                    playAnimation(frames, finalState);
                } else {
                    setIsAiThinking(false);
                }
            };

            const payload = {
                board: gameState.board,
                kazans: gameState.kazans,
                tuzdyks: gameState.tuzdyks,
                currentPlayer: gameState.currentPlayer,
                isGameOver: gameState.isGameOver,
                winner: gameState.winner,
                algorithm: 'mcts',
                iterations: 20000,
                max_time_seconds: 12.0,
            };

            fetch(`${AI_API_BASE}/api/best-move`, {
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
                    const fallbackMove = calculateBestMove(gameState, 5, aiPlayer, 'minimax');
                    setTimeout(() => handleBestMove(fallbackMove), 400);
                });
        }
    }, [gameState.currentPlayer, gameState.isGameOver, isAiThinking, isAnimating, gameState, playAnimation, aiPlayer]);

    useEffect(() => {
        if (!gameState.isGameOver || learningSentRef.current) return;

        learningSentRef.current = true;
        const samples = aiMoveHistoryRef.current;
        if (samples.length === 0) return;

        fetch(`${AI_API_BASE}/api/learn`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                samples,
                winner: gameState.winner,
                aiPlayer,
                finalKazans: gameState.kazans,
            }),
        }).catch((err) => {
            console.error('Failed to submit AI learning data:', err);
        });
    }, [gameState.isGameOver, gameState.winner, gameState.kazans, aiPlayer]);

    const handlePocketClick = useCallback((index) => {
        if (gameState.currentPlayer === humanPlayer && !gameState.isGameOver && !isAiThinking && !isAnimating) {
            if (gameState.isValidMove(humanPlayer, index)) {
                const frames = gameState.getMoveSteps(index);
                const finalState = gameState.clone();
                const moveResult = finalState.makeMove(index);
                if (moveResult && moveResult.notation) {
                    setMoveHistory((prev) => [...prev, { player: humanPlayer, notation: moveResult.notation }]);
                }
                playAnimation(frames, finalState);
            }
        }
    }, [gameState, isAiThinking, isAnimating, playAnimation, humanPlayer]);

    const handleRestart = useCallback((nextHumanPlayer = humanPlayer) => {
        if (animTimerRef.current) clearTimeout(animTimerRef.current);
        setAnimBoard(null);
        setAnimKazans(null);
        setAnimTuzdyks(null);
        setHighlightIndex(-1);
        setAnimPhase(null);
        setIsAnimating(false);
        setIsAiThinking(false);
        aiMoveHistoryRef.current = [];
        learningSentRef.current = false;
        setMoveHistory([]);
        setHumanPlayer(nextHumanPlayer);
        setGameState(new TogyzkumalakState());
    }, [humanPlayer]);

    // Группируем ходы парами (белые / чёрные) — нотация в стиле togyz_js (yernarsha).
    const movePairs = useMemo(() => {
        const pairs = [];
        for (let i = 0; i < moveHistory.length; i += 2) {
            pairs.push({
                index: i / 2 + 1,
                white: moveHistory[i],
                black: moveHistory[i + 1] || null,
            });
        }
        return pairs;
    }, [moveHistory]);

    const topRowIndices = [17, 16, 15, 14, 13, 12, 11, 10, 9];
    const bottomRowIndices = [0, 1, 2, 3, 4, 5, 6, 7, 8];

    const isBusy = isAnimating || isAiThinking;
    const winnerText = gameState.winner === humanPlayer
        ? 'Вы выиграли'
        : gameState.winner === aiPlayer
            ? 'Компьютер выиграл'
            : 'Ничья';

    const statusText = useMemo(() => {
        if (gameState.isGameOver) return 'Партия завершена';
        if (isAiThinking) return 'Компьютер думает';
        if (isAnimating) return 'Ход выполняется';
        return gameState.currentPlayer === humanPlayer ? 'Ваш ход' : 'Ход компьютера';
    }, [gameState.currentPlayer, gameState.isGameOver, isAiThinking, isAnimating, humanPlayer]);

    return (
        <main className="app-shell">
            <section className="hero-bar">
                <div className="brand-block">
                    <span className="eyebrow">Классическая казахская игра</span>
                    <h1>Тоғызқұмалақ</h1>
                    <p>Играйте против компьютера, следите за қазанами и создавайте тұздық в нужный момент.</p>
                </div>

                <div className="control-panel" aria-label="Настройки игры">
                    <div className="ai-mode-badge">
                        <span>AI режим</span>
                        <strong>{AI_MODE_LABEL}</strong>
                    </div>
                    <div className="start-toggle" role="group" aria-label="Кто ходит первым">
                        <button
                            type="button"
                            className={`toggle-btn ${humanPlayer === 0 ? 'active' : ''}`}
                            onClick={() => handleRestart(0)}
                            aria-pressed={humanPlayer === 0}
                        >
                            Я первым
                        </button>
                        <button
                            type="button"
                            className={`toggle-btn ${humanPlayer === 1 ? 'active' : ''}`}
                            onClick={() => handleRestart(1)}
                            aria-pressed={humanPlayer === 1}
                        >
                            ИИ первым
                        </button>
                    </div>
                    <button className="primary-action" onClick={() => handleRestart()}>Новая игра</button>
                </div>
            </section>

            <section className="match-panel">
                <div className="score-strip">
                    <div className={`player-card computer ${gameState.currentPlayer === aiPlayer && !isBusy ? 'active' : ''}`}>
                        <span className="player-label">Компьютер</span>
                        <strong>{displayKazans[aiPlayer]}</strong>
                    </div>
                    <div className="status-card">
                        <span>{statusText}</span>
                        <strong>{AI_MODE_LABEL}</strong>
                    </div>
                    <div className={`player-card human ${gameState.currentPlayer === humanPlayer && !isBusy ? 'active' : ''}`}>
                        <span className="player-label">Вы</span>
                        <strong>{displayKazans[humanPlayer]}</strong>
                    </div>
                </div>

                <div className="board-scroll" aria-label="Горизонтальная игровая доска">
                    <div className="board-container">
                        <div className="side-label opponent-label">{humanPlayer === 0 ? 'Қарсылас' : 'Ойыншы'}</div>
                        <div className="otau-row top-row">
                            {topRowIndices.map((index) => (
                                <Otau
                                    key={index}
                                    index={index}
                                    count={displayBoard[index]}
                                    isEnabled={humanPlayer === 1 && gameState.currentPlayer === 1 && !isBusy}
                                    isTuzdyk={displayTuzdyks.includes(index)}
                                    owner={displayTuzdyks.indexOf(index)}
                                    onClick={humanPlayer === 1 ? handlePocketClick : () => {}}
                                    label={index - 8}
                                    isHighlighted={highlightIndex === index}
                                    animPhase={highlightIndex === index ? animPhase : null}
                                />
                            ))}
                        </div>

                        <div className="kazan-section">
                            <Kazan player={0} count={displayKazans[0]} isActive={gameState.currentPlayer === 0 && !isBusy} />
                            <Kazan player={1} count={displayKazans[1]} isActive={gameState.currentPlayer === 1 && !isBusy} />
                        </div>

                        <div className="otau-row bottom-row">
                            {bottomRowIndices.map((index) => (
                                <Otau
                                    key={index}
                                    index={index}
                                    count={displayBoard[index]}
                                    isEnabled={humanPlayer === 0 && gameState.currentPlayer === 0 && !isBusy}
                                    isTuzdyk={displayTuzdyks.includes(index)}
                                    owner={displayTuzdyks.indexOf(index)}
                                    onClick={humanPlayer === 0 ? handlePocketClick : () => {}}
                                    label={index + 1}
                                    isHighlighted={highlightIndex === index}
                                    animPhase={highlightIndex === index ? animPhase : null}
                                />
                            ))}
                        </div>
                        <div className="side-label player-label-board">{humanPlayer === 0 ? 'Ойыншы' : 'Қарсылас'}</div>
                    </div>
                </div>
            </section>

            <section className="moves-panel" aria-label="История ходов">
                <header className="moves-header">
                    <span className="moves-title">История ходов</span>
                    <span className="moves-counter">{moveHistory.length}</span>
                </header>
                {movePairs.length === 0 ? (
                    <p className="moves-empty">Партия ещё не начата.</p>
                ) : (
                    <ol className="moves-list">
                        {movePairs.map((pair) => (
                            <li key={pair.index} className="moves-row">
                                <span className="moves-index">{pair.index}.</span>
                                <span className="moves-cell white">{pair.white?.notation || ''}</span>
                                <span className="moves-cell black">{pair.black?.notation || ''}</span>
                            </li>
                        ))}
                    </ol>
                )}
            </section>

            <div className={`modal-overlay ${gameState.isGameOver ? 'visible' : ''}`}>
                <div className="modal-content">
                    <span className="modal-kicker">Игра окончена</span>
                    <h2>{winnerText}</h2>
                    <p>Счет: вы {gameState.kazans[humanPlayer]} – {gameState.kazans[aiPlayer]} компьютер</p>
                    <button className="primary-action" onClick={() => handleRestart()}>Сыграть еще</button>
                </div>
            </div>
        </main>
    );
}

export default App;
