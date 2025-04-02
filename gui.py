import pygame
import sys
from game import ReversiGame
from typing import List, Tuple, Dict, Any, Optional

class ReversiGUI:
    """Pygame GUI for Reversi game"""
    
    # Colors
    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)
    GREEN = (0, 128, 0)
    DARK_GREEN = (0, 100, 0)
    BLUE = (100, 100, 255)
    GRAY = (128, 128, 128)
    
    def __init__(self, game, screen_size=600, human_players=[1, 2]):
        """Initialize the GUI
        
        Args:
            game: ReversiGame instance
            screen_size: Size of the game window in pixels
            human_players: List of player IDs that are human-controlled (1 for black, 2 for white)
        """
        # Initialize pygame if not already initialized
        if not pygame.get_init():
            pygame.init()
        
        pygame.display.set_caption("Reversi Game")
        
        self.game = game
        self.screen_size = screen_size
        self.cell_size = screen_size // game.board_size
        self.screen = pygame.display.set_mode((screen_size, screen_size + 50))  # Extra space for status bar
        self.font = pygame.font.SysFont(None, 24)
        self.clock = pygame.time.Clock()
        self.human_players = human_players.copy()  # Create a copy to avoid modifying the original
        self.bots = {}  # Maps player ID to bot instance
        
    def register_bot(self, player_id, bot):
        """Register a bot for a player"""
        self.bots[player_id] = bot
        if player_id in self.human_players:
            self.human_players.remove(player_id)
    
    def draw_board(self, state):
        """Draw the game board"""
        board = state['board']
        valid_moves = state['valid_moves']
        current_player = state['current_player']
        last_move = state['last_move']
        
        # Draw background
        self.screen.fill(self.GREEN)
        
        # Draw grid lines
        for i in range(self.game.board_size + 1):
            pygame.draw.line(self.screen, self.BLACK, 
                            (0, i * self.cell_size), 
                            (self.screen_size, i * self.cell_size), 2)
            pygame.draw.line(self.screen, self.BLACK, 
                            (i * self.cell_size, 0), 
                            (i * self.cell_size, self.screen_size), 2)
        
        # Draw pieces
        for i in range(self.game.board_size):
            for j in range(self.game.board_size):
                center_x = j * self.cell_size + self.cell_size // 2
                center_y = i * self.cell_size + self.cell_size // 2
                radius = self.cell_size // 2 - 4
                
                if board[i][j] == 1:  # Black
                    pygame.draw.circle(self.screen, self.BLACK, (center_x, center_y), radius)
                elif board[i][j] == 2:  # White
                    pygame.draw.circle(self.screen, self.WHITE, (center_x, center_y), radius)
                
                # Highlight the last move
                if last_move and last_move[0] == i and last_move[1] == j:
                    pygame.draw.circle(self.screen, self.BLUE, (center_x, center_y), radius // 3)
        
        # Draw valid moves for current player (if human)
        if current_player in self.human_players:
            for row, col in valid_moves:
                center_x = col * self.cell_size + self.cell_size // 2
                center_y = row * self.cell_size + self.cell_size // 2
                pygame.draw.circle(self.screen, self.GRAY, (center_x, center_y), self.cell_size // 6)
        
        # Draw status bar
        black_score, white_score = self.game.get_score()
        status_y = self.screen_size + 10
        
        # Current player indicator
        if current_player == 1:
            pygame.draw.circle(self.screen, self.BLACK, (20, status_y + 15), 10)
        else:
            pygame.draw.circle(self.screen, self.WHITE, (20, status_y + 15), 10)
        
        # Scores
        black_text = self.font.render(f"Black: {black_score}", True, self.BLACK)
        white_text = self.font.render(f"White: {white_score}", True, self.BLACK)
        self.screen.blit(black_text, (50, status_y))
        self.screen.blit(white_text, (150, status_y))
        
        # Game over message
        if state['game_over']:
            if state['winner'] == 1:
                msg = "Black wins!"
            elif state['winner'] == 2:
                msg = "White wins!"
            else:
                msg = "Draw!"
            game_over_text = self.font.render(msg, True, self.BLACK)
            self.screen.blit(game_over_text, (300, status_y))
            
        pygame.display.flip()
    
    def handle_events(self):
        """Handle pygame events, return move coordinates if a move is made"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            
            if event.type == pygame.MOUSEBUTTONDOWN:
                # Check if click is on the board
                if event.pos[1] < self.screen_size:
                    col = event.pos[0] // self.cell_size
                    row = event.pos[1] // self.cell_size
                    return row, col
        
        return None
    
    def run_game_loop(self, fps=30):
        """Main game loop"""
        running = True
        state = self.game.get_state()
        
        while running:
            move = None
            current_player = state['current_player']
            
            # If game is not over and current player is a bot, get move from bot
            if not state['game_over'] and current_player not in self.human_players:
                if current_player in self.bots:
                    bot_move = self.bots[current_player].get_move(state)
                    if bot_move:
                        move = bot_move
            
            # For human players, get move from GUI
            elif not state['game_over'] and current_player in self.human_players:
                move = self.handle_events()
            else:
                # Just handle quit events if game is over
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
            
            # Make move if one is available
            if move:
                row, col = move
                new_state = self.game.make_move(row, col)
                
                # Check if the move was valid (by checking if something changed)
                if (new_state['current_player'] != state['current_player'] or 
                    new_state['game_over'] != state['game_over'] or
                    new_state['last_move'] != state['last_move']):
                    state = new_state
            
            # Draw the board
            self.draw_board(state)
            self.clock.tick(fps)
        
        pygame.quit()
    
    def get_clicked_position(self):
        """Wait for a mouse click and return the board position"""
        waiting = True
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                
                if event.type == pygame.MOUSEBUTTONDOWN:
                    # Check if click is on the board
                    if event.pos[1] < self.screen_size:
                        col = event.pos[0] // self.cell_size
                        row = event.pos[1] // self.cell_size
                        return row, col
            
            self.clock.tick(30) 