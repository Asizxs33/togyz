import React, { useState, useEffect, useCallback, useRef } from 'react';
import { TogyzkumalakState } from './logic/Togyzkumalak';
import { calculateBestMove } from './logic/AI';
import { Otau } from './components/Otau';
import { Kazan } from './components/Kazan';

const ANIM_DELAY = 180; // ms between each stone placement frame

function App() {
    const [gameState, setGameState] = useState(new TogyzkumalakState());
    const [difficulty, setDifficulty] = useState(3);
    const [isAiThinking, setIsAiThinking] = useState(false);
    
    // Animation state
    const [isAnimating, setIsAnimating] = useState(false);
    const [animBoard, setAnimBoard] = useState(null);        // Overrides gameState.board during animation
    const [animKazans, setAnimKazans] = useState(null);      // Overrides gameState.kazans during animation
    const [animTuzdyks, setAnimTuzdyks] = useState(null);    // Overrides gameState.tuzdyks during animation
    const [highlightIndex, setHighlightIndex] = useState(-1); // Which pocket is currently highlighted
    const [animPhase, setAnimPhase] = useState(null);         // 'pickup' | 'sow' | 'capture' | 'tuzdyk'
    const animTimerRef = useRef(null);

    // Derived display state (animation overrides game state visually)
    const displayBoard = animBoard || gameState.board;
    const displayKazans = animKazans || gameState.kazans;
    const displayTuzdyks = animTuzdyks || gameState.tuzdyks;

    // Play animation frames sequentially
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
                // Animation done — apply the real final game state
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

    // AI turn
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
                // Fetch from Python Backend MCTS
                const payload = {
                    board: gameState.board,
                    kazans: gameState.kazans,
                    tuzdyks: gameState.tuzdyks,
                    currentPlayer: gameState.currentPlayer,
                    isGameOver: gameState.isGameOver,
                    winner: gameState.winner,
                    algorithm: 'mcts',
                    iterations: 20000 // Send to python to do Heavy lifting
                };
                
                fetch('http://localhost:5000/api/best-move', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                })
                .then(res => res.json())
                .then(data => {
                    handleBestMove(data.move);
                })
                .catch(err => {
                    console.error("Failed to reach Python backend:", err);
                    alert("Серверге (Python Backend) қосылу мүмкін емес! Стандартты Minimax (Қиын) режимі орындалады.");
                    // Fallback to local minimax
                    const fallbackMove = calculateBestMove(gameState, 5, 1, 'minimax');
                    setTimeout(() => handleBestMove(fallbackMove), 400);
                });
            } else {
                // Local Standard Minimax
                setTimeout(() => {
                    const depth = parseInt(difficulty);
                    const bestMove = calculateBestMove(gameState, depth, 1, 'minimax');
                    handleBestMove(bestMove);
                }, 400);
            }
        }
    }, [gameState.currentPlayer, gameState.isGameOver, isAiThinking, isAnimating, difficulty, gameState, playAnimation]);

    // Player move
    const handlePocketClick = useCallback((index) => {
        if (gameState.currentPlayer === 0 && !gameState.isGameOver && !isAiThinking && !isAnimating) {
            if (gameState.isValidMove(0, index)) {
                // Get animation frames
                const frames = gameState.getMoveSteps(index);
                
                // Get the real final state
                const finalState = gameState.clone();
                finalState.makeMove(index);
                
                playAnimation(frames, finalState);
            }
        }
    }, [gameState, isAiThinking, isAnimating, playAnimation]);

    const handleRestart = () => {
        // Clear any running animation
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

    const isWinner = gameState.winner === 0 ? "Сен жеңдің! 🎉" : (gameState.winner === 1 ? "Компьютер жеңді 🤖" : "Тең ойын 🤝");

    const isBusy = isAnimating || isAiThinking;

    return (
        <div className="app-container">
            <header>
                <h1>Тоғызқұмалақ</h1>
                <div className="controls">
                    <select value={difficulty} onChange={(e) => setDifficulty(e.target.value)} disabled={isBusy}>
                        <option value="2">Оңай (Easy)</option>
                        <option value="3">Орташа (Medium)</option>
                        <option value="5">Қиын (Hard)</option>
                        <option value="mcts">Экстремалды (MCTS - AI)</option>
                    </select>
                    <button onClick={handleRestart}>Жаңа ойын</button>
                    {isAiThinking && <span className="thinking-indicator">🤖 Компьютер ойлануда...</span>}
                    {isAnimating && <span className="thinking-indicator">⏳</span>}
                </div>
            </header>

            <div className="board-container classic-wood">
                {/* TOP ROW (AI) */}
                <div className="otau-row">
                    {topRowIndices.map(index => (
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

                {/* MIDDLE SECTION (KAZANS) */}
                <div className="kazan-section">
                    <Kazan player={1} count={displayKazans[1]} isActive={gameState.currentPlayer === 1 && !isBusy} />
                    <Kazan player={0} count={displayKazans[0]} isActive={gameState.currentPlayer === 0 && !isBusy} />
                </div>

                {/* BOTTOM ROW (PLAYER) */}
                <div className="otau-row">
                    {bottomRowIndices.map(index => (
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
            </div>

            <div className={`modal-overlay ${gameState.isGameOver ? 'visible' : ''}`}>
                <div className="modal-content glass">
                    <h2>Ойын аяқталды!</h2>
                    <p style={{fontSize: '2rem', color: '#38bdf8', fontWeight: 'bold'}}>{isWinner}</p>
                    <p>Есеп: {gameState.kazans[0]} - {gameState.kazans[1]}</p>
                    <button onClick={handleRestart} style={{
                        background: '#38bdf8', color: '#0f172a', padding: '1rem 2rem', 
                        fontSize: '1.2rem', fontWeight: 'bold', border: 'none', borderRadius: '12px',
                        cursor: 'pointer', marginTop: '1rem'
                    }}>
                        Жаңа басынан
                    </button>
                </div>
            </div>
        </div>
    );
}

export default App;
