import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from collections.abc import Sequence
from collections import deque
import os
from typing import NamedTuple, Optional, Tuple, List, Dict, Any

from bot import Bot
from game import ReversiGame

# Experience replay memory
class Experience(NamedTuple):
    state: Dict[str, Any]
    action: Tuple[int, int]
    reward: float
    next_state: Dict[str, Any]
    done: bool

class ReplayMemory:
    """Experience replay memory buffer"""
    
    def __init__(self, capacity=10000):
        self.memory = deque(maxlen=capacity)
    
    def add(self, state, action, reward, next_state, done):
        """Add experience to memory"""
        self.memory.append(Experience(state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """Sample random batch from memory"""
        return random.sample(self.memory, min(batch_size, len(self.memory)))
    
    def __len__(self):
        return len(self.memory)


class DQNNetwork(nn.Module):
    """Deep Q-Network for Reversi"""
    
    def __init__(self, board_size=8):
        super(DQNNetwork, self).__init__()
        
        # Input channels: 3 (one-hot encoding of empty/black/white)
        # Board representations as input tensors:
        # channel 0: empty spaces (1 where empty, 0 elsewhere)
        # channel 1: player's pieces (1 where present, 0 elsewhere)
        # channel 2: opponent's pieces (1 where present, 0 elsewhere)
        
        self.board_size = board_size
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 * board_size * board_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, board_size * board_size)  # Output for each possible position
        
    def forward(self, x):
        # Convolutional layers with ReLU activations
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # Flatten the output
        x = x.view(-1, 128 * self.board_size * self.board_size)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x


class DQNBot(Bot):
    """Bot that uses Deep Q-Network for decision making"""
    
    def __init__(self, player_id, board_size=8, epsilon=0.1, gamma=0.99, 
                 learning_rate=0.001, batch_size=64, target_update=10,
                 model_path=None):
        """Initialize DQN bot
        
        Args:
            player_id: The player ID (1=black, 2=white)
            board_size: Size of the board
            epsilon: Exploration rate (probability of random move)
            gamma: Discount factor for future rewards
            learning_rate: Learning rate for optimizer
            batch_size: Batch size for training
            target_update: How often to update target network
            model_path: Path to load the model from (if None, create new model)
        """
        super().__init__(player_id)
        
        self.board_size = board_size
        self.epsilon = epsilon
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update = target_update
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize Q-networks (policy and target)
        self.policy_net = DQNNetwork(board_size).to(self.device)
        self.target_net = DQNNetwork(board_size).to(self.device)
        
        # If model path is provided, load weights
        if model_path and os.path.exists(model_path):
            try:
                self.policy_net.load_state_dict(torch.load(
                    model_path, 
                    map_location=self.device
                ))
            except Exception as e:
                print(f"Error loading model: {e}")
                print("Creating a new model instead")
        
        # Copy policy net weights to target net
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Target net is only used for inference
        
        # Setup optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        
        # Experience replay memory
        self.memory = ReplayMemory()
        
        # Training tracking
        self.step_count = 0
        self.training_mode = False
    
    def set_training_mode(self, mode=True):
        """Set whether the bot is training or just playing"""
        self.training_mode = mode
        if mode:
            self.policy_net.train()
        else:
            self.policy_net.eval()
    
    def save_model(self, path):
        """Save model weights to file"""
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(path), exist_ok=True)
            torch.save(self.policy_net.state_dict(), path)
            return True
        except Exception as e:
            print(f"Error saving model: {e}")
            return False
    
    def get_move(self, state):
        """Get move using epsilon-greedy policy"""
        valid_moves = state['valid_moves']
        if not valid_moves:
            return None
            
        # Convert state to tensor input for the network
        state_tensor = self._state_to_tensor(state)
        
        # With probability epsilon, choose a random move (exploration)
        if random.random() < self.epsilon and self.training_mode:
            return random.choice(valid_moves)
            
        # Otherwise choose the best move according to the policy network (exploitation)
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)
        
        # Reshape Q-values to board dimensions
        q_values = q_values.reshape(self.board_size, self.board_size)
        
        # Find the move with the highest Q-value among valid moves
        best_value = float('-inf')
        best_move = None
        
        for row, col in valid_moves:
            value = q_values[row, col].item()
            if value > best_value:
                best_value = value
                best_move = (row, col)
        
        return best_move
    
    def store_experience(self, state, action, reward, next_state, done):
        """Store experience in replay memory"""
        self.memory.add(state, action, reward, next_state, done)
    
    def train(self):
        """Train the network on a batch from replay memory"""
        if len(self.memory) < self.batch_size:
            return
            
        # Sample experiences
        experiences = self.memory.sample(self.batch_size)
        
        # Convert experiences to tensors
        states = torch.cat([self._state_to_tensor(e.state) for e in experiences])
        actions = torch.tensor([(e.action[0] * self.board_size + e.action[1]) for e in experiences], 
                               dtype=torch.long, device=self.device).unsqueeze(1)
        rewards = torch.tensor([e.reward for e in experiences], 
                               dtype=torch.float, device=self.device)
        next_states = torch.cat([self._state_to_tensor(e.next_state) for e in experiences])
        dones = torch.tensor([e.done for e in experiences], 
                             dtype=torch.bool, device=self.device)
        
        # Compute current Q-values: Q(s, a)
        current_q = self.policy_net(states).gather(1, actions)
        
        # Compute expected Q-values
        next_q = torch.zeros(self.batch_size, device=self.device)
        with torch.no_grad():
            # Use target network for next state values to stabilize training
            next_q_values = self.target_net(next_states).reshape(self.batch_size, -1)
            
            # For each next_state, get the valid moves and find max Q-value
            for i, exp in enumerate(experiences):
                if not exp.done:
                    valid_moves = exp.next_state['valid_moves']
                    if valid_moves:
                        max_q = float('-inf')
                        for row, col in valid_moves:
                            q_val = next_q_values[i, row * self.board_size + col].item()
                            max_q = max(max_q, q_val)
                        next_q[i] = max_q
        
        # Compute target Q-values: r + γ * max Q(s', a')
        target_q = rewards + self.gamma * next_q * (~dones)
        
        # Compute loss
        loss = F.smooth_l1_loss(current_q.squeeze(1), target_q)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        # Clip gradients to stabilize training
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        # Update target network periodically
        self.step_count += 1
        if self.step_count % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
            
        return loss.item()
    
    def _state_to_tensor(self, state):
        """Convert game state to tensor for neural network input"""
        board = state['board']
        
        # Create 3-channel representation
        tensor = torch.zeros(3, self.board_size, self.board_size, device=self.device)
        
        # Channel 0: Empty spaces
        tensor[0] = torch.tensor((board == 0), dtype=torch.float, device=self.device)
        
        # Channel 1: Player's pieces
        tensor[1] = torch.tensor((board == self.player_id), dtype=torch.float, device=self.device)
        
        # Channel 2: Opponent's pieces
        tensor[2] = torch.tensor((board == 3 - self.player_id), dtype=torch.float, device=self.device)
        
        return tensor.unsqueeze(0)  # Add batch dimension


class DQNTrainer:
    """Trainer for DQN Reversi bot using self-play"""
    
    def __init__(self, board_size=8, epsilon_start=1.0, epsilon_end=0.1, 
                 epsilon_decay=0.999, model_path=None):
        self.board_size = board_size
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.model_path = model_path
        
        # Create bot that will be trained
        self.dqn_bot = DQNBot(player_id=1, board_size=board_size, epsilon=epsilon_start,
                             model_path=model_path)
        self.dqn_bot.set_training_mode(True)
        
        # Game environment
        self.game = ReversiGame(board_size)
        
        # Training stats
        self.rewards_history = []
        self.win_history = []
        self.loss_history = []
    
    def self_play_episode(self, opponent=None):
        """Play one episode against itself or provided opponent and train"""
        if opponent is None:
            # Play against a copy of itself (with lower epsilon for better moves)
            opponent = DQNBot(player_id=2, board_size=self.board_size, epsilon=self.epsilon_end)
            opponent.policy_net.load_state_dict(self.dqn_bot.policy_net.state_dict())
            opponent.set_training_mode(False)
        else:
            # Make sure opponent has the right player_id
            opponent.player_id = 2
        
        # Reset game
        state = self.game.reset()
        done = False
        total_reward = 0
        
        # Play until game is over
        while not done:
            # Get player
            player = self.dqn_bot if state['current_player'] == 1 else opponent
            
            # Get move
            action = player.get_move(state)
            if action is None:
                # No valid moves, skip turn
                # In a properly implemented ReversiGame, this shouldn't happen as the game 
                # should automatically switch players or end when no moves are available
                next_state = self.game.get_state()
                if next_state['current_player'] == state['current_player']:
                    # If player didn't change, force switch
                    self.game.current_player = 3 - self.game.current_player
                    next_state = self.game.get_state()
                state = next_state
                continue
                
            # Make move
            row, col = action
            next_state = self.game.make_move(row, col)
            
            # Check if the game is over
            done = next_state['game_over']
            
            # Calculate reward for the player who just moved
            if done:
                if next_state['winner'] == player.player_id:
                    reward = 1.0  # Win
                elif next_state['winner'] == 0:
                    reward = 0.2  # Draw
                else:
                    reward = -1.0  # Loss
            else:
                # Small incentive for capturing more pieces
                p_score, o_score = self.game.get_score()
                if player.player_id == 1:
                    reward = 0.01 * (p_score - o_score) / (self.board_size * self.board_size)
                else:
                    reward = 0.01 * (o_score - p_score) / (self.board_size * self.board_size)
            
            # Store experience for player 1 (the one being trained)
            if player.player_id == 1:
                self.dqn_bot.store_experience(state, action, reward, next_state, done)
                total_reward += reward
                
                # Train the network
                if len(self.dqn_bot.memory) >= self.dqn_bot.batch_size:
                    loss = self.dqn_bot.train()
                    if loss is not None:
                        self.loss_history.append(loss)
            
            # Update state
            state = next_state
        
        # Record results
        self.rewards_history.append(total_reward)
        if state['winner'] == 1:
            self.win_history.append(1)  # Win
        elif state['winner'] == 0:
            self.win_history.append(0)  # Draw
        else:
            self.win_history.append(-1)  # Loss
            
        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        self.dqn_bot.epsilon = self.epsilon
        
        return state['winner'], total_reward
    
    def train(self, num_episodes=1000, save_interval=100, opponent=None):
        """Train the DQN bot for the specified number of episodes"""
        for episode in range(1, num_episodes + 1):
            winner, episode_reward = self.self_play_episode(opponent)
            
            # Print progress
            if episode % 10 == 0:
                avg_reward = sum(self.rewards_history[-10:]) / 10
                avg_wins = sum(1 for w in self.win_history[-10:] if w == 1)
                avg_draws = sum(1 for w in self.win_history[-10:] if w == 0)
                avg_losses = sum(1 for w in self.win_history[-10:] if w == -1)
                
                print(f"Episode {episode}/{num_episodes} - "
                      f"Reward: {episode_reward:.2f} - "
                      f"Avg Reward: {avg_reward:.2f} - "
                      f"Wins/Draws/Losses: {avg_wins}/{avg_draws}/{avg_losses} - "
                      f"Epsilon: {self.epsilon:.3f}")
            
            # Save model periodically
            if episode % save_interval == 0 and self.model_path:
                self.dqn_bot.save_model(f"{self.model_path}_episode_{episode}.pt")
        
        # Save final model
        if self.model_path:
            self.dqn_bot.save_model(f"{self.model_path}_final.pt")
        
        return self.dqn_bot 