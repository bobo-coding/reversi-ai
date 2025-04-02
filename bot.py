import random
import numpy as np
import abc

class Bot(abc.ABC):
    """Abstract base class for Reversi bots"""
    
    def __init__(self, player_id):
        """Initialize bot
        
        Args:
            player_id: The player ID this bot controls (1 for black, 2 for white)
        """
        self.player_id = player_id
    
    @abc.abstractmethod
    def get_move(self, state):
        """Get the next move given the current game state
        
        Args:
            state: Game state dictionary from ReversiGame.get_state()
            
        Returns:
            Tuple of (row, col) for the move, or None if no move is possible
        """
        pass

class RandomBot(Bot):
    """Bot that makes random valid moves"""
    
    def get_move(self, state):
        """Return a random valid move"""
        valid_moves = state['valid_moves']
        if not valid_moves:
            return None
        return random.choice(valid_moves)

class MinimaxBot(Bot):
    """Simple Minimax bot with a limited depth"""
    
    def __init__(self, player_id, depth=3):
        super().__init__(player_id)
        self.depth = depth
    
    def get_move(self, state):
        """Return the best move according to minimax evaluation"""
        valid_moves = state['valid_moves']
        if not valid_moves:
            return None
            
        best_score = float('-inf')
        best_move = None
        
        for move in valid_moves:
            # Create a copy of the game to simulate moves
            from game import ReversiGame
            game_copy = ReversiGame(state['board'].shape[0])
            game_copy.board = state['board'].copy()
            game_copy.current_player = state['current_player']
            
            # Make the move
            row, col = move
            game_copy.make_move(row, col)
            
            # Evaluate the move
            score = self._minimax(game_copy, self.depth - 1, False)
            
            if score > best_score:
                best_score = score
                best_move = move
        
        return best_move
    
    def _minimax(self, game, depth, is_maximizing):
        """Minimax algorithm implementation"""
        state = game.get_state()
        
        # Terminal conditions
        if depth == 0 or state['game_over']:
            return self._evaluate(state)
        
        valid_moves = state['valid_moves']
        
        if is_maximizing:
            best_score = float('-inf')
            for move in valid_moves:
                row, col = move
                
                # Create a new game state with this move
                from game import ReversiGame
                game_copy = ReversiGame(state['board'].shape[0])
                game_copy.board = state['board'].copy()
                game_copy.current_player = state['current_player']
                
                game_copy.make_move(row, col)
                score = self._minimax(game_copy, depth - 1, False)
                best_score = max(score, best_score)
            
            return best_score
        else:
            best_score = float('inf')
            for move in valid_moves:
                row, col = move
                
                # Create a new game state with this move
                from game import ReversiGame
                game_copy = ReversiGame(state['board'].shape[0])
                game_copy.board = state['board'].copy()
                game_copy.current_player = state['current_player']
                
                game_copy.make_move(row, col)
                score = self._minimax(game_copy, depth - 1, True)
                best_score = min(score, best_score)
            
            return best_score
    
    def _evaluate(self, state):
        """Simple evaluation function that counts pieces with positional weighting"""
        board = state['board']
        
        # Position weights - corners and edges are more valuable
        weights = np.ones_like(board, dtype=float)
        size = len(board)
        
        # Corners are most valuable
        corners = [(0,0), (0,size-1), (size-1,0), (size-1,size-1)]
        for r, c in corners:
            weights[r][c] = 10.0
            
        # Edges are valuable
        for i in range(1, size-1):
            weights[0][i] = 3.0  # Top edge
            weights[size-1][i] = 3.0  # Bottom edge
            weights[i][0] = 3.0  # Left edge
            weights[i][size-1] = 3.0  # Right edge
        
        # Bad positions (adjacent to corners)
        for r, c in corners:
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < size and 0 <= nc < size and (nr, nc) not in corners:
                        weights[nr][nc] = -2.0
        
        my_pieces = board == self.player_id
        opponent_pieces = board == (3 - self.player_id)
        
        my_score = np.sum(weights * my_pieces)
        opponent_score = np.sum(weights * opponent_pieces)
        
        return my_score - opponent_score 