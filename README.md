# Reversi-AI-Training

A Reversi (Othello) game implementation with Pygame and a Deep Q-Network (DQN) AI for training and gameplay.

## Features

- Complete Reversi game implementation with Pygame GUI
- Bot interfaces that can be used for AI development
- Sample bot implementations:
  - Random Bot (makes random valid moves)
  - Minimax Bot (uses a minimax algorithm with a positional evaluation function)
  - DQN Bot (uses a Deep Q-Network for reinforcement learning)
- Training framework for the DQN AI with self-play
- Command-line interface for playing games and training AI

## Requirements

```
pygame==2.5.2
numpy==1.24.3
torch==2.0.1
matplotlib==3.7.2
```

## Installation

1. Clone the repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

### Playing a Game

To play a game of Reversi, use the `play` command:

```
python main.py play [options]
```

Options:
- `--black`: Type of player for black (human, random, minimax, dqn) [default: human]
- `--white`: Type of player for white (human, random, minimax, dqn) [default: human]
- `--board-size`: Size of the board [default: 8]
- `--model-path`: Path to a trained DQN model (required if using DQN player)

Examples:
```
# Play as black against a random bot
python main.py play --white random

# Play as white against a minimax bot
python main.py play --black minimax

# Watch a game between a DQN bot and a minimax bot
python main.py play --black dqn --white minimax --model-path models/dqn_reversi_final.pt

# Play on a 6x6 board
python main.py play --board-size 6
```

### Training the DQN AI

To train the DQN AI through self-play:

```
python main.py train [options]
```

Options:
- `--episodes`: Number of episodes to train [default: 1000]
- `--board-size`: Size of the board [default: 8]
- `--model-dir`: Directory to save models and training progress [default: models]

Examples:
```
# Train for 1000 episodes
python main.py train

# Train for 5000 episodes
python main.py train --episodes 5000

# Train on a 6x6 board
python main.py train --board-size 6 --model-dir models_6x6
```

## Project Structure

- `game.py`: Core Reversi game logic
- `gui.py`: Pygame GUI for the game
- `bot.py`: Bot interface and simple bot implementations
- `dqn_bot.py`: DQN implementation and training framework
- `main.py`: Command-line interface and main entry point

## Customizing Bots

To create your own bot, simply inherit from the `Bot` class in `bot.py` and implement the `get_move` method:

```python
from bot import Bot

class MyCustomBot(Bot):
    def __init__(self, player_id):
        super().__init__(player_id)
        # Your initialization code here
    
    def get_move(self, state):
        # Your move selection logic here
        # Return a tuple (row, col) for the move
        pass
```

## DQN Architecture

The DQN uses a convolutional neural network architecture:
- Input: 3-channel representation of the board (empty spaces, player pieces, opponent pieces)
- 3 convolutional layers with ReLU activations
- 3 fully connected layers
- Output: Q-values for each possible position on the board

The DQN is trained using:
- Experience replay for more stable learning
- Target network to reduce correlation between updates
- Rewards for winning/losing and capturing pieces

## License

This project is open source and available under the MIT License. 