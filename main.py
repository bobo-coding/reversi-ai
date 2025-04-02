import argparse
import os
import matplotlib.pyplot as plt
import numpy as np

from game import ReversiGame
from gui import ReversiGUI
from bot import RandomBot, MinimaxBot
from dqn_bot import DQNBot, DQNTrainer

def play_game(player1=None, player2=None, board_size=8, gui=True):
    """Play a game of Reversi with the specified players
    
    Args:
        player1: Bot for player 1 (black), or None for human
        player2: Bot for player 2 (white), or None for human
        board_size: Size of the board
        gui: Whether to show GUI
        
    Returns:
        The winner of the game (1=black, 2=white, 0=draw)
    """
    game = ReversiGame(board_size)
    
    # Determine which players are human
    human_players = []
    if player1 is None:
        human_players.append(1)
    if player2 is None:
        human_players.append(2)
    
    if gui:
        # Create GUI
        gui = ReversiGUI(game, human_players=human_players)
        
        # Register bots
        if player1:
            gui.register_bot(1, player1)
        if player2:
            gui.register_bot(2, player2)
        
        # Run game loop
        gui.run_game_loop()
    else:
        # Headless mode for faster training
        state = game.get_state()
        
        while not state['game_over']:
            player = player1 if state['current_player'] == 1 else player2
            
            # Skip turn if no player for current player
            if (state['current_player'] == 1 and player1 is None) or \
               (state['current_player'] == 2 and player2 is None):
                break
            
            # Get move from bot
            move = player.get_move(state)
            
            # Make move
            if move:
                row, col = move
                state = game.make_move(row, col)
        
    return game.winner

def train_dqn(num_episodes=1000, board_size=8, model_dir='models'):
    """Train a DQN bot through self-play"""
    # Create model directory if it doesn't exist
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    model_path = os.path.join(model_dir, 'dqn_reversi')
    
    # Create trainer
    trainer = DQNTrainer(
        board_size=board_size,
        epsilon_start=1.0,
        epsilon_end=0.1,
        epsilon_decay=0.999,
        model_path=model_path
    )
    
    # Train the bot
    trained_bot = trainer.train(num_episodes=num_episodes, save_interval=100)
    
    # Plot training progress if there's enough data
    if trainer.rewards_history:
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 3, 1)
        plt.plot(trainer.rewards_history)
        plt.title('Rewards per Episode')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        
        # Plot win rate if there's enough data
        if len(trainer.win_history) > 100:
            plt.subplot(1, 3, 2)
            # Calculate win rate over a window
            window = min(100, len(trainer.win_history) // 2)  # Use smaller window if needed
            win_rates = []
            for i in range(len(trainer.win_history) - window + 1):
                wins = sum(1 for w in trainer.win_history[i:i+window] if w == 1)
                win_rates.append(wins / window)
            plt.plot(range(window-1, window-1+len(win_rates)), win_rates)
            plt.title(f'Win Rate ({window}-episode window)')
            plt.xlabel('Episode')
            plt.ylabel('Win Rate')
        
        # Plot loss if there's enough data
        if trainer.loss_history and len(trainer.loss_history) > 50:
            plt.subplot(1, 3, 3)
            # Smooth the loss curve
            window_size = min(100, len(trainer.loss_history) // 4)  # Use smaller window if needed
            if window_size > 0:
                smoothed_losses = np.convolve(trainer.loss_history, 
                                            np.ones(window_size)/window_size, 
                                            mode='valid')
                plt.plot(smoothed_losses)
                plt.title('Smoothed Loss')
                plt.xlabel('Training Step')
                plt.ylabel('Loss')
        
        plt.tight_layout()
        plt.savefig(os.path.join(model_dir, 'training_progress.png'))
    
    return trained_bot

def main():
    parser = argparse.ArgumentParser(description='Reversi Game and AI Training')
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    subparsers.required = False  # Make command optional for help message
    
    # Play game command
    play_parser = subparsers.add_parser('play', help='Play Reversi')
    play_parser.add_argument('--black', choices=['human', 'random', 'minimax', 'dqn'], 
                             default='human', help='Black player type')
    play_parser.add_argument('--white', choices=['human', 'random', 'minimax', 'dqn'], 
                             default='human', help='White player type')
    play_parser.add_argument('--board-size', type=int, default=8, help='Board size')
    play_parser.add_argument('--model-path', type=str, help='Path to DQN model file')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train DQN bot')
    train_parser.add_argument('--episodes', type=int, default=1000, help='Number of episodes to train')
    train_parser.add_argument('--board-size', type=int, default=8, help='Board size')
    train_parser.add_argument('--model-dir', type=str, default='models', help='Directory to save models')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    if args.command == 'play':
        # Create players
        player1 = None  # Black
        player2 = None  # White
        
        if args.black == 'random':
            player1 = RandomBot(1)
        elif args.black == 'minimax':
            player1 = MinimaxBot(1)
        elif args.black == 'dqn':
            if not args.model_path:
                print("Error: Must specify --model-path when using DQN player")
                return
            player1 = DQNBot(1, board_size=args.board_size, model_path=args.model_path)
            player1.set_training_mode(False)
        
        if args.white == 'random':
            player2 = RandomBot(2)
        elif args.white == 'minimax':
            player2 = MinimaxBot(2)
        elif args.white == 'dqn':
            if not args.model_path:
                print("Error: Must specify --model-path when using DQN player")
                return
            player2 = DQNBot(2, board_size=args.board_size, model_path=args.model_path)
            player2.set_training_mode(False)
        
        play_game(player1, player2, args.board_size)
        
    elif args.command == 'train':
        train_dqn(num_episodes=args.episodes, 
                  board_size=args.board_size, 
                  model_dir=args.model_dir)
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main() 