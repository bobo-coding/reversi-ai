import numpy as np
from typing import List, Tuple, Dict, Any, Optional

class ReversiGame:
    """Core Reversi game logic"""
    
    # Board representation: 0 = empty, 1 = black, 2 = white
    # Black moves first
    
    def __init__(self, board_size=8):
        self.board_size = board_size
        self.reset()
        
    def reset(self):
        """Reset the game to the starting position"""
        self.board = np.zeros((self.board_size, self.board_size), dtype=np.int8)
        # Set up the initial four pieces in the center
        center = self.board_size // 2
        self.board[center-1][center-1] = 2  # White
        self.board[center][center] = 2      # White
        self.board[center-1][center] = 1    # Black
        self.board[center][center-1] = 1    # Black
        
        self.current_player = 1  # Black starts
        self.game_over = False
        self.winner = None
        self.last_move = None
        return self.get_state()
    
    def get_state(self) -> Dict[str, Any]:
        """Return the current game state"""
        return {
            'board': self.board.copy(),
            'current_player': self.current_player,
            'game_over': self.game_over,
            'winner': self.winner,
            'valid_moves': self.get_valid_moves(),
            'last_move': self.last_move
        }
    
    def get_valid_moves(self) -> List[Tuple[int, int]]:
        """Return a list of valid moves for the current player"""
        valid_moves = []
        for i in range(self.board_size):
            for j in range(self.board_size):
                if self._is_valid_move(i, j):
                    valid_moves.append((i, j))
        return valid_moves
    
    def make_move(self, row, col):
        """Make a move at the specified position and return new state"""
        if self.game_over:
            return self.get_state()
            
        if not self._is_valid_move(row, col):
            return self.get_state()
            
        # Place the piece and flip opponent's pieces
        self.board[row][col] = self.current_player
        self._flip_pieces(row, col)
        self.last_move = (row, col)
        
        # Switch player
        self.current_player = 3 - self.current_player  # 1->2, 2->1
        
        # Check if the next player has valid moves
        if not self.get_valid_moves():
            # If the current player has no valid moves, switch back and check again
            self.current_player = 3 - self.current_player
            if not self.get_valid_moves():
                # If both players have no valid moves, the game is over
                self.game_over = True
                self._determine_winner()
                
        return self.get_state()
    
    def _is_valid_move(self, row, col):
        """Check if a move is valid"""
        # Check if position is on the board and empty
        if not (0 <= row < self.board_size and 0 <= col < self.board_size) or self.board[row][col] != 0:
            return False
            
        # Directions: horizontal, vertical, and diagonal
        directions = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1)
        ]
        
        opponent = 3 - self.current_player  # 1->2, 2->1
        
        # Check each direction
        for dr, dc in directions:
            r, c = row + dr, col + dc
            # First adjacent piece must be opponent's
            if not (0 <= r < self.board_size and 0 <= c < self.board_size) or self.board[r][c] != opponent:
                continue
                
            # Continue in this direction
            r += dr
            c += dc
            found_player_piece = False
            
            while 0 <= r < self.board_size and 0 <= c < self.board_size:
                if self.board[r][c] == 0:
                    break
                if self.board[r][c] == self.current_player:
                    found_player_piece = True
                    break
                r += dr
                c += dc
                
            if found_player_piece:
                return True
                
        return False
    
    def _flip_pieces(self, row, col):
        """Flip opponent pieces as a result of a move"""
        directions = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1)
        ]
        
        opponent = 3 - self.current_player
        
        for dr, dc in directions:
            pieces_to_flip = []
            r, c = row + dr, col + dc
            
            # Check for opponent pieces in this direction
            while 0 <= r < self.board_size and 0 <= c < self.board_size and self.board[r][c] == opponent:
                pieces_to_flip.append((r, c))
                r += dr
                c += dc
                
            # If we found the current player's piece at the end, flip all pieces in between
            if 0 <= r < self.board_size and 0 <= c < self.board_size and self.board[r][c] == self.current_player:
                for flip_r, flip_c in pieces_to_flip:
                    self.board[flip_r][flip_c] = self.current_player
    
    def _determine_winner(self):
        """Determine the winner based on the number of pieces"""
        black_count = np.count_nonzero(self.board == 1)
        white_count = np.count_nonzero(self.board == 2)
        
        if black_count > white_count:
            self.winner = 1  # Black wins
        elif white_count > black_count:
            self.winner = 2  # White wins
        else:
            self.winner = 0  # Draw
    
    def get_score(self):
        """Return the current score (black pieces, white pieces)"""
        black_count = np.count_nonzero(self.board == 1)
        white_count = np.count_nonzero(self.board == 2)
        return black_count, white_count
    
    def get_board_for_player(self, player_id):
        """Get the board representation from a player's perspective
        This is useful for AI training to maintain perspective consistency
        """
        if player_id == 1:  # Black
            return self.board.copy()
        else:  # White
            # Swap 1s and 2s for the white player's perspective
            board_copy = self.board.copy()
            board_copy[board_copy == 1] = 3  # Temporary value
            board_copy[board_copy == 2] = 1
            board_copy[board_copy == 3] = 2
            return board_copy 