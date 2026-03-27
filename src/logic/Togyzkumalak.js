export class TogyzkumalakState {
    constructor() {
        this.board = Array(18).fill(9); // 18 pockets
        this.kazans = [0, 0]; // 2 kazans (Player 1, Player 2)
        this.tuzdyks = [-1, -1]; // Stores the index of the tuzdyk pocket for Player 1 and Player 2. -1 means no tuzdyk.
        this.currentPlayer = 0; // 0 for Player 1, 1 for Player 2
        this.isGameOver = false;
        this.winner = null;
    }

    clone() {
        const newState = new TogyzkumalakState();
        newState.board = [...this.board];
        newState.kazans = [...this.kazans];
        newState.tuzdyks = [...this.tuzdyks];
        newState.currentPlayer = this.currentPlayer;
        newState.isGameOver = this.isGameOver;
        newState.winner = this.winner;
        return newState;
    }

    isValidMove(player, index) {
        if (this.isGameOver || player !== this.currentPlayer) return false;
        if (player === 0 && (index < 0 || index > 8)) return false;
        if (player === 1 && (index < 9 || index > 17)) return false;
        return this.board[index] > 0;
    }

    makeMove(index) {
        if (!this.isValidMove(this.currentPlayer, index)) return false;

        let stones = this.board[index];
        this.board[index] = 0;
        
        let currentIndex = index;

        if (stones === 1) {
            currentIndex = (currentIndex + 1) % 18;
            this.addStoneToPocket(currentIndex);
        } else {
            this.board[index] = 1;
            stones--;
            while (stones > 0) {
                currentIndex = (currentIndex + 1) % 18;
                this.addStoneToPocket(currentIndex);
                stones--;
            }
        }

        this.checkCaptureAndTuzdyk(currentIndex, this.currentPlayer);
        this.checkGameState();
        
        if (!this.isGameOver) {
            this.currentPlayer = 1 - this.currentPlayer;
        }

        return true;
    }

    addStoneToPocket(index) {
        // If the stone drops into a tuzdyk, send to owner's kazan
        if (this.tuzdyks[0] === index) {
            this.kazans[0]++;
        } else if (this.tuzdyks[1] === index) {
            this.kazans[1]++;
        } else {
            this.board[index]++;
        }
    }

    checkCaptureAndTuzdyk(lastIndex, player) {
        const oppStart = player === 0 ? 9 : 0;
        const oppEnd = player === 0 ? 17 : 8;

        // Is the last stone on the opponent's side?
        if (lastIndex >= oppStart && lastIndex <= oppEnd) {
            if (this.board[lastIndex] === 3) {
                // Tuzdyk Rule
                const isNinthPocket = (lastIndex === 8 || lastIndex === 17);
                const hasNoTuzdyk = this.tuzdyks[player] === -1;
                
                const symIndex = lastIndex % 9;
                const oppTuzdykSym = this.tuzdyks[1 - player] !== -1 ? this.tuzdyks[1 - player] % 9 : -1;
                const isNotSymmetrical = symIndex !== oppTuzdykSym;

                if (!isNinthPocket && hasNoTuzdyk && isNotSymmetrical) {
                    this.tuzdyks[player] = lastIndex;
                    this.kazans[player] += 3;
                    this.board[lastIndex] = 0;
                }
            } else if (this.board[lastIndex] > 0 && this.board[lastIndex] % 2 === 0) {
                // Capture Rule
                this.kazans[player] += this.board[lastIndex];
                this.board[lastIndex] = 0;
            }
        }
    }

    checkGameState() {
        // Condition 1: Someone reached >= 82
        if (this.kazans[0] >= 82) {
            this.isGameOver = true;
            this.winner = 0;
            return;
        }
        if (this.kazans[1] >= 82) {
            this.isGameOver = true;
            this.winner = 1;
            return;
        }

        // Condition 2: Atsyrau (Out of stones to move)
        const nextPlayer = 1 - this.currentPlayer;
        const start = nextPlayer === 0 ? 0 : 9;
        const end = nextPlayer === 0 ? 8 : 17;
        
        let hasStones = false;
        for (let i = start; i <= end; i++) {
            if (this.board[i] > 0) {
                hasStones = true;
                break;
            }
        }

        if (!hasStones) {
            // Atsyrau: opponent claims all remaining stones
            const oppWinner = this.currentPlayer; 
            let remainingStones = 0;
            const oppStart = oppWinner === 0 ? 0 : 9;
            const oppEnd = oppWinner === 0 ? 8 : 17;

            for (let i = oppStart; i <= oppEnd; i++) {
                remainingStones += this.board[i];
                this.board[i] = 0;
            }
            this.kazans[oppWinner] += remainingStones;
            
            this.isGameOver = true;
            if (this.kazans[0] > this.kazans[1]) this.winner = 0;
            else if (this.kazans[1] > this.kazans[0]) this.winner = 1;
            else this.winner = null; // Draw
        }
    }

    getPossibleMoves(player) {
        const moves = [];
        const start = player === 0 ? 0 : 9;
        const end = player === 0 ? 8 : 17;
        for (let i = start; i <= end; i++) {
            if (this.board[i] > 0) {
                moves.push(i);
            }
        }
        return moves;
    }

    /**
     * Returns an array of animation frames for the given move.
     * Each frame is { board, kazans, tuzdyks, highlightIndex, phase }.
     * phase: 'pickup' | 'sow' | 'capture' | 'tuzdyk' | 'done'
     */
    getMoveSteps(index) {
        if (!this.isValidMove(this.currentPlayer, index)) return [];

        const frames = [];
        const player = this.currentPlayer;

        // Work on a clone for frame generation
        const board = [...this.board];
        const kazans = [...this.kazans];
        const tuzdyks = [...this.tuzdyks];

        let stones = board[index];
        board[index] = 0;

        // Frame 0: Pickup — show the pocket emptied
        frames.push({
            board: [...board],
            kazans: [...kazans],
            tuzdyks: [...tuzdyks],
            highlightIndex: index,
            phase: 'pickup'
        });

        let currentIndex = index;

        if (stones === 1) {
            currentIndex = (currentIndex + 1) % 18;
            // Place stone (respecting tuzdyk)
            if (tuzdyks[0] === currentIndex) kazans[0]++;
            else if (tuzdyks[1] === currentIndex) kazans[1]++;
            else board[currentIndex]++;

            frames.push({
                board: [...board],
                kazans: [...kazans],
                tuzdyks: [...tuzdyks],
                highlightIndex: currentIndex,
                phase: 'sow'
            });
        } else {
            // Leave 1 stone in the original pocket
            board[index] = 1;
            stones--;

            frames.push({
                board: [...board],
                kazans: [...kazans],
                tuzdyks: [...tuzdyks],
                highlightIndex: index,
                phase: 'sow'
            });

            while (stones > 0) {
                currentIndex = (currentIndex + 1) % 18;

                if (tuzdyks[0] === currentIndex) kazans[0]++;
                else if (tuzdyks[1] === currentIndex) kazans[1]++;
                else board[currentIndex]++;

                stones--;

                frames.push({
                    board: [...board],
                    kazans: [...kazans],
                    tuzdyks: [...tuzdyks],
                    highlightIndex: currentIndex,
                    phase: 'sow'
                });
            }
        }

        // Now check capture / tuzdyk on the last index
        const oppStart = player === 0 ? 9 : 0;
        const oppEnd = player === 0 ? 17 : 8;

        if (currentIndex >= oppStart && currentIndex <= oppEnd) {
            if (board[currentIndex] === 3) {
                const isNinthPocket = (currentIndex === 8 || currentIndex === 17);
                const hasNoTuzdyk = tuzdyks[player] === -1;
                const symIndex = currentIndex % 9;
                const oppTuzdykSym = tuzdyks[1 - player] !== -1 ? tuzdyks[1 - player] % 9 : -1;
                const isNotSymmetrical = symIndex !== oppTuzdykSym;

                if (!isNinthPocket && hasNoTuzdyk && isNotSymmetrical) {
                    tuzdyks[player] = currentIndex;
                    kazans[player] += 3;
                    board[currentIndex] = 0;
                    frames.push({
                        board: [...board],
                        kazans: [...kazans],
                        tuzdyks: [...tuzdyks],
                        highlightIndex: currentIndex,
                        phase: 'tuzdyk'
                    });
                }
            } else if (board[currentIndex] > 0 && board[currentIndex] % 2 === 0) {
                kazans[player] += board[currentIndex];
                board[currentIndex] = 0;
                frames.push({
                    board: [...board],
                    kazans: [...kazans],
                    tuzdyks: [...tuzdyks],
                    highlightIndex: currentIndex,
                    phase: 'capture'
                });
            }
        }

        return frames;
    }
}
