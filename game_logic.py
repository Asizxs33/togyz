import copy
import random

class TogyzkumalakState:
    def __init__(self):
        # 18 pockets, 9 per player.
        # Indices 0-8 belong to Player 0 (bottom row, left to right)
        # Indices 9-17 belong to Player 1 (top row, right to left)
        self.board = [9] * 18
        self.kazans = [0, 0] # player 0, player 1
        self.tuzdyks = [-1, -1] # The index of the pocket that is a tuzdyk
        self.currentPlayer = 0 # 0 or 1
        self.isGameOver = False
        self.winner = None

    def clone(self):
        new_state = TogyzkumalakState()
        new_state.board = self.board[:]
        new_state.kazans = self.kazans[:]
        new_state.tuzdyks = self.tuzdyks[:]
        new_state.currentPlayer = self.currentPlayer
        new_state.isGameOver = self.isGameOver
        new_state.winner = self.winner
        return new_state

    def isValidMove(self, player, index):
        if player != self.currentPlayer:
            return False
        # Player 0 uses pockets 0-8, Player 1 uses 9-17
        if player == 0 and not (0 <= index <= 8):
            return False
        if player == 1 and not (9 <= index <= 17):
            return False
        
        # Can't move from an empty pocket
        if self.board[index] == 0:
            return False
            
        return True

    def getPossibleMoves(self, player):
        moves = []
        start = 0 if player == 0 else 9
        end = 8 if player == 0 else 17
        for i in range(start, end + 1):
            if self.board[i] > 0:
                moves.append(i)
        return moves

    def makeMove(self, index):
        if not self.isValidMove(self.currentPlayer, index):
            return False
            
        stones = self.board[index]
        self.board[index] = 0
        
        current_idx = index
        
        # Special case: if only 1 stone, we drop it in the next pocket
        if stones == 1:
            current_idx = (current_idx + 1) % 18
            # Skip pockets that have become Tuzdyks
            while current_idx in self.tuzdyks:
                self._addToTuzdyk(current_idx, 1) # This stone falls into tuzdyk
                # But physically we need to move the pointer
                # Actually, in standard Togyzkumalak, stones fall into the Tuzdyk owner's kazan.
                # So the stone is dropped there and sowing ends.
                return self._finalizeMove()
            
            self.board[current_idx] += 1
        else:
            # Rule: drop first stone in starting pocket
            self.board[current_idx] += 1
            stones -= 1
            
            while stones > 0:
                current_idx = (current_idx + 1) % 18
                
                # Check if it's a Tuzdyk
                if current_idx in self.tuzdyks:
                    self._addToTuzdyk(current_idx, 1)
                else:
                    self.board[current_idx] += 1
                stones -= 1

        # Harvest rule (Tuzdyk or Capture)
        # Capture only happens if the last stone landed on the OPPONENT's side
        opponent = 1 - self.currentPlayer
        opp_start = 0 if opponent == 0 else 9
        opp_end = 8 if opponent == 0 else 17
        
        if opp_start <= current_idx <= opp_end:
            # Check for Capture (Even number of stones)
            if self.board[current_idx] % 2 == 0:
                self.kazans[self.currentPlayer] += self.board[current_idx]
                self.board[current_idx] = 0
            # Check for Tuzdyk (Exactly 3 stones)
            elif self.board[current_idx] == 3 and self.tuzdyks[self.currentPlayer] == -1:    
                # Rule: Cannot make Tuzdyk on the 9th pocket (index 8 or 17)
                is_ninth = (current_idx == 8) or (current_idx == 17)
                # Rule: Cannot make Tuzdyk symmetrically if opponent has one attached to the same pocket index relative to player length
                my_relative = current_idx % 9
                opp_tuz = self.tuzdyks[opponent]
                is_symmetric = False
                if opp_tuz != -1:
                    opp_relative = opp_tuz % 9
                    if my_relative == opp_relative:
                        is_symmetric = True
                
                if not is_ninth and not is_symmetric:
                    self.tuzdyks[self.currentPlayer] = current_idx
                    self.kazans[self.currentPlayer] += 3 # existing stones go to kazan
                    self.board[current_idx] = 0
                    
        return self._finalizeMove()

    def _addToTuzdyk(self, index, amount):
        if self.tuzdyks[0] == index:
            self.kazans[0] += amount
        elif self.tuzdyks[1] == index:
            self.kazans[1] += amount

    def _finalizeMove(self):
        self.currentPlayer = 1 - self.currentPlayer
        self._checkGameOver()
        return True
        
    def _checkGameOver(self):
        # A player wins if they collect more than 81 stones
        if self.kazans[0] > 81:
            self.isGameOver = True
            self.winner = 0
            return
        if self.kazans[1] > 81:
            self.isGameOver = True
            self.winner = 1
            return
            
        # Draw condition
        if self.kazans[0] == 81 and self.kazans[1] == 81:
            self.isGameOver = True
            self.winner = -1 # draw
            return
            
        # Atsyrau Check (No stones left to make a move)
        p0_moves = self.getPossibleMoves(0)
        p1_moves = self.getPossibleMoves(1)
        
        if len(p0_moves) == 0 and self.currentPlayer == 0:
            # P1 gets all remaining stones
            remaining = sum(self.board)
            self.kazans[1] += remaining
            for i in range(18):
                self.board[i] = 0
            self.isGameOver = True
            self._determineWinnerByKazan()
        elif len(p1_moves) == 0 and self.currentPlayer == 1:
            # P0 gets all remaining
            remaining = sum(self.board)
            self.kazans[0] += remaining
            for i in range(18):
                self.board[i] = 0
            self.isGameOver = True
            self._determineWinnerByKazan()

    def _determineWinnerByKazan(self):
        if self.kazans[0] > self.kazans[1]:
            self.winner = 0
        elif self.kazans[1] > self.kazans[0]:
            self.winner = 1
        else:
            self.winner = -1
