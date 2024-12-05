import numpy as np
import torch
from torch import nn
import torch_dwn as dwn
from collections import deque
import random
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os
from datetime import datetime
from typing import List
from abc import ABC, abstractmethod

# Define the abstract base class that was missing
class GameEnvironment(ABC):
    @abstractmethod
    def get_state(self):
        pass
    
    @abstractmethod
    def get_valid_actions(self):
        pass
    
    @abstractmethod
    def make_move(self, action):
        pass
    
    @abstractmethod
    def is_game_over(self):
        pass
    
    @abstractmethod
    def get_reward(self):
        pass

class Game2048(GameEnvironment):
    def __init__(self):
        self.board = np.zeros((4, 4), dtype=int)
        self.score = 0
        self.max_tile = 0
        self.merge_count = 0
        self.moves_since_merge = 0
        self.previous_max_tile = 0
        self._add_new_tile()
        self._add_new_tile()
    
    def get_state(self):
        return self.board.flatten()
    
    def get_state_2d(self):
        return np.copy(self.board)
    
    def get_monotonicity(self):
        mono_left = mono_right = mono_up = mono_down = 0
        
        for i in range(4):
            for j in range(3):
                if self.board[i][j] != 0 and self.board[i][j+1] != 0:
                    if self.board[i][j] >= self.board[i][j+1]:
                        mono_left += np.log2(self.board[i][j]) - np.log2(self.board[i][j+1])
                    if self.board[i][j] <= self.board[i][j+1]:
                        mono_right += np.log2(self.board[i][j+1]) - np.log2(self.board[i][j])
        
        for j in range(4):
            for i in range(3):
                if self.board[i][j] != 0 and self.board[i+1][j] != 0:
                    if self.board[i][j] >= self.board[i+1][j]:
                        mono_up += np.log2(self.board[i][j]) - np.log2(self.board[i+1][j])
                    if self.board[i][j] <= self.board[i+1][j]:
                        mono_down += np.log2(self.board[i+1][j]) - np.log2(self.board[i][j])
        
        return max(mono_left, mono_right) + max(mono_up, mono_down)
    
    def get_smoothness(self):
        smoothness = 0
        for i in range(4):
            for j in range(4):
                if self.board[i][j] != 0:
                    value = np.log2(self.board[i][j])
                    if j < 3 and self.board[i][j+1] != 0:
                        smoothness -= abs(value - np.log2(self.board[i][j+1]))
                    if i < 3 and self.board[i+1][j] != 0:
                        smoothness -= abs(value - np.log2(self.board[i+1][j]))
        return smoothness
    
    def get_empty_tiles(self):
        return len(np.where(self.board == 0)[0])
    
    def get_valid_actions(self):
        valid_actions = []
        for action in range(4):  # 0: up, 1: right, 2: down, 3: left
            if self._is_valid_move(action):
                valid_actions.append(action)
        return valid_actions
    
    def _is_valid_move(self, action):
        temp_board = np.copy(self.board)
        return self._move(temp_board, action, test_only=True)
    
    def make_move(self, action):
        self.previous_max_tile = self.max_tile
        result = self._move(self.board, action)
        if not result:
            self.moves_since_merge += 1
        return result
    
    def _move(self, board, action, test_only=False):
        original_board = np.copy(board)
        
        if action == 0:  # up
            board = np.rot90(board)
        elif action == 1:  # right
            board = np.rot90(board, 2)
        elif action == 2:  # down
            board = np.rot90(board, 3)
        
        moved = False
        merges_this_move = 0
        
        for i in range(4):
            row = board[i]
            row = row[row != 0]
            
            j = 0
            while j < len(row) - 1:
                if row[j] == row[j + 1]:
                    row[j] *= 2
                    if not test_only:
                        self.score += row[j]
                        self.max_tile = max(self.max_tile, row[j])
                        merges_this_move += 1
                    row = np.delete(row, j + 1)
                j += 1
            
            new_row = np.zeros(4, dtype=int)
            new_row[:len(row)] = row
            
            if not np.array_equal(board[i], new_row):
                moved = True
            board[i] = new_row
        
        if action == 0:
            board = np.rot90(board, 3)
        elif action == 1:
            board = np.rot90(board, 2)
        elif action == 2:
            board = np.rot90(board)
        
        if moved and not test_only:
            self._add_new_tile()
            self.merge_count += merges_this_move
            self.moves_since_merge = 0
        
        return moved
    
    def _add_new_tile(self):
        empty_cells = list(zip(*np.where(self.board == 0)))
        if empty_cells:
            x, y = random.choice(empty_cells)
            self.board[x][y] = np.random.choice([2, 4], p=[0.9, 0.1])
    
    def is_game_over(self):
        return len(self.get_valid_actions()) == 0
    
    def get_reward(self):
        """
        Improved reward function that better guides learning toward winning strategies
        """
        reward = 0
        
        # Major reward for achieving higher tiles (exponential scaling)
        if self.max_tile > self.previous_max_tile:
            reward += 2.0 ** (np.log2(self.max_tile) - 7)  # Normalized around 128
        
        # Reward for maintaining empty tiles (quadratic scaling)
        empty_tiles = self.get_empty_tiles()
        reward += 0.1 * (empty_tiles ** 2)
        
        # Penalty for moves without merges (exponential penalty)
        if self.moves_since_merge > 3:
            reward -= 2.0 ** (self.moves_since_merge - 3)
        
        # Reward for maintaining a good board structure
        mono_score = self.get_monotonicity()
        smooth_score = self.get_smoothness()
        reward += 0.5 * mono_score + 0.3 * smooth_score
        
        # Extra reward for reaching milestone tiles
        milestone_tiles = [128, 256, 512, 1024, 2048]
        if self.max_tile in milestone_tiles and self.max_tile > self.previous_max_tile:
            reward += 10.0 * np.log2(self.max_tile)
        
        return float(reward)
    
    def get_max_tile(self):
        return np.max(self.board)
    
    def is_won(self):
        return self.get_max_tile() >= 2048
    
    def __str__(self):
        cell_width = max(len(str(cell)) for cell in self.board.flatten())
        rows = []
        for row in self.board:
            row_str = ' '.join(str(cell).rjust(cell_width) for cell in row)
            rows.append(row_str)
        return '\n'.join(rows)

class MetricsTracker:
    def __init__(self):
        self.episode_scores: List[int] = []
        self.episode_max_tiles: List[int] = []
        self.episode_steps: List[int] = []
        self.episode_times: List[float] = []
        self.wins: int = 0
        
        self.graphics_dir = 'graphics'
        if not os.path.exists(self.graphics_dir):
            os.makedirs(self.graphics_dir)
    
    def add_episode(self, score: int, max_tile: int, steps: int, duration: float):
        self.episode_scores.append(score)
        self.episode_max_tiles.append(max_tile)
        self.episode_steps.append(steps)
        self.episode_times.append(duration)
        if max_tile >= 2048:
            self.wins += 1
    
    def plot_metrics(self):
        plt.style.use('classic')
        sns.set_palette("husl")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 15))
        fig.suptitle('2048 DWN Agent Learning Progress', fontsize=16, y=0.95)
        
        episodes = range(len(self.episode_scores))
        
        # Plot 1: Scores
        ax1.plot(episodes, self.episode_scores, label='Score', linewidth=1)
        z = np.polyfit(episodes, self.episode_scores, 1)
        p = np.poly1d(z)
        ax1.plot(episodes, p(episodes), "r--", alpha=0.8, label=f'Trend (slope: {z[0]:.2f})')
        ax1.set_title('Game Score Evolution', fontsize=14, pad=10)
        ax1.set_xlabel('Episode Number', fontsize=12)
        ax1.set_ylabel('Score', fontsize=12)
        ax1.grid(True, linestyle='--', alpha=0.7)
        ax1.legend(fontsize=10)
        
        # Plot 2: Maximum Tiles
        ax2.plot(episodes, self.episode_max_tiles, label='Max Tile', linewidth=1)
        ax2.set_title('Maximum Tile Achieved per Episode', fontsize=14, pad=10)
        ax2.set_xlabel('Episode Number', fontsize=12)
        ax2.set_ylabel('Maximum Tile Value', fontsize=12)
        ax2.grid(True, linestyle='--', alpha=0.7)
        ax2.legend(fontsize=10)
        
        # Plot 3: Steps
        ax3.plot(episodes, self.episode_steps, label='Steps', linewidth=1)
        ax3.set_title('Steps per Episode', fontsize=14, pad=10)
        ax3.set_xlabel('Episode Number', fontsize=12)
        ax3.set_ylabel('Number of Steps', fontsize=12)
        ax3.grid(True, linestyle='--', alpha=0.7)
        ax3.legend(fontsize=10)
        
        # Plot 4: Moving Averages
        window = 20
        scores_ma = np.convolve(self.episode_scores, np.ones(window)/window, mode='valid')
        tiles_ma = np.convolve(self.episode_max_tiles, np.ones(window)/window, mode='valid')
        
        ax4.plot(range(window-1, len(episodes)), scores_ma, label=f'Score (MA-{window})', linewidth=2)
        ax4.plot(range(window-1, len(episodes)), tiles_ma, label=f'Max Tile (MA-{window})', linewidth=2)
        ax4.set_title('Moving Averages of Score and Max Tile', fontsize=14, pad=10)
        ax4.set_xlabel('Episode Number', fontsize=12)
        ax4.set_ylabel('Value', fontsize=12)
        ax4.grid(True, linestyle='--', alpha=0.7)
        ax4.legend(fontsize=10)
        
        plt.tight_layout()
        
        filename = f'learning_metrics_{timestamp}.png'
        filepath = os.path.join(self.graphics_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print("\nTraining Summary:")
        print(f"Total Episodes: {len(self.episode_scores)}")
        print(f"Total Wins (2048 achieved): {self.wins}")
        print(f"Best Score: {max(self.episode_scores)}")
        print(f"Highest Tile: {max(self.episode_max_tiles)}")
        print(f"Average Score: {np.mean(self.episode_scores):.2f}")
        print(f"Average Steps: {np.mean(self.episode_steps):.2f}")
        print(f"Average Time per Episode: {np.mean(self.episode_times):.2f}s")
        print(f"\nMetrics plot saved as: {filepath}")
        
        data_filename = f'learning_data_{timestamp}.txt'
        data_filepath = os.path.join(self.graphics_dir, data_filename)
        with open(data_filepath, 'w') as f:
            f.write("Episode,Score,MaxTile,Steps,Time\n")
            for i in range(len(self.episode_scores)):
                f.write(f"{i},{self.episode_scores[i]},{self.episode_max_tiles[i]}," +
                       f"{self.episode_steps[i]},{self.episode_times[i]:.2f}\n")
        print(f"Numerical data saved as: {data_filepath}")

class QLearningAgent:
    def __init__(self, state_size, action_size):
        # Check GPU availability and properties
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. Please check your GPU installation.")
        
        self.device = torch.device("cuda")
        
        # Print GPU info
        print("\n=== GPU Information ===")
        print(f"GPU Device: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory Allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
        print(f"GPU Memory Cached: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")
        print("=====================\n")
        
        # Enable cuDNN auto-tuner for best performance
        torch.backends.cudnn.benchmark = True
        
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=50000)
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.998
        self.batch_size = 256
        self.learning_rate = 0.0005
        
        # Create main network with layer norm instead of batch norm
        self.model = self._build_model().to(self.device)
        
        # Create target network
        self.target_model = self._build_model().to(self.device)
        self.update_target_model()
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), 
                                        lr=self.learning_rate,
                                        weight_decay=1e-4)
        self.criterion = nn.MSELoss()
        
        # Tracking metrics
        self.losses = []
        self.avg_q_values = []
        self.action_frequencies = {i: 0 for i in range(action_size)}
        self.track_gpu_memory = True
        self.target_update_counter = 0
        self.target_update_frequency = 10
        
    def _build_model(self):
        """Builds neural network model using LayerNorm instead of BatchNorm"""
        return nn.Sequential(
            nn.Linear(self.state_size, 256),
            nn.LayerNorm(256),  # LayerNorm instead of BatchNorm
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(256, 512),
            nn.LayerNorm(512),  # LayerNorm instead of BatchNorm
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(512, 256),
            nn.LayerNorm(256),  # LayerNorm instead of BatchNorm
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(256, self.action_size)
        )
    
    def update_target_model(self):
        """Copy weights from model to target_model"""
        self.target_model.load_state_dict(self.model.state_dict())
    
    def preprocess_state(self, state):
        """Preprocess state with improved normalization"""
        state_array = np.array(state, dtype=np.float32)
        non_zero_mask = state_array > 0
        processed_state = np.zeros_like(state_array)
        processed_state[non_zero_mask] = np.log2(state_array[non_zero_mask])
        if processed_state.max() > 0:
            processed_state = processed_state / 11.0
        
        return torch.FloatTensor(processed_state).view(1, -1).to(self.device, non_blocking=True)
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience with priority"""
        # Use max priority for new experiences
        max_priority = max([x[2] for x in self.memory]) if self.memory else 1.0
        self.memory.append((state, action, max_priority, reward, next_state, done))
    
    def act(self, state, valid_actions):
        if not valid_actions:
            return None
        
        if random.random() <= self.epsilon:
            action = random.choice(valid_actions)
            self.action_frequencies[action] += 1
            return action
        
        with torch.no_grad():
            self.model.eval()  # Set to evaluation mode
            state = self.preprocess_state(state)
            q_values = self.model(state).cpu()
            self.model.train()  # Set back to training mode
            
            self.avg_q_values.append(q_values.mean().item())
            
            # Mask invalid actions
            action_mask = torch.full((self.action_size,), float('-inf'))
            action_mask[valid_actions] = 0
            masked_q_values = q_values + action_mask
            
            action = masked_q_values.argmax().item()
            self.action_frequencies[action] += 1
            return action

    def replay(self):
        if len(self.memory) < self.batch_size:
            return None
        
        if self.track_gpu_memory:
            before_memory = torch.cuda.memory_allocated() / 1024**2
        
        # Sample batch with priorities
        priorities = np.array([x[2] for x in self.memory])
        probs = priorities ** 0.6  # priority_alpha
        probs /= probs.sum()
        
        indices = np.random.choice(len(self.memory), 
                                 self.batch_size, 
                                 p=probs)
        
        batch = [self.memory[idx] for idx in indices]
        
        # Calculate importance sampling weights
        weights = (len(self.memory) * probs[indices]) ** (-0.4)  # priority_beta
        weights /= weights.max()
        weights = torch.FloatTensor(weights).to(self.device)
        
        # Prepare batch tensors
        states = torch.cat([self.preprocess_state(state[0]) for state in batch])
        actions = torch.tensor([state[1] for state in batch], 
                             device=self.device, 
                             dtype=torch.long).view(-1, 1)
        rewards = torch.tensor([state[3] for state in batch], 
                             device=self.device, 
                             dtype=torch.float32)
        next_states = torch.cat([self.preprocess_state(state[4]) for state in batch])
        dones = torch.tensor([float(state[5]) for state in batch], 
                           device=self.device, 
                           dtype=torch.float32)
        
        # Compute Q-values
        self.optimizer.zero_grad()
        
        with torch.amp.autocast(device_type='cuda'):
            current_q_values = self.model(states).gather(1, actions).squeeze()
            
            with torch.no_grad():
                next_q_values = self.target_model(next_states)
                next_q_max = next_q_values.max(1)[0]
                target_q_values = rewards + (1 - dones) * self.gamma * next_q_max
            
            # Compute weighted loss
            losses = self.criterion(current_q_values, target_q_values)
            loss = (losses * weights).mean()
        
        # Backward pass and optimization
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        # Update priorities
        with torch.no_grad():
            td_errors = abs(target_q_values - current_q_values).cpu().numpy()
        
        for idx, error in zip(indices, td_errors):
            self.memory[idx] = (*self.memory[idx][:2], 
                              error + 1e-6,  # priority_epsilon
                              *self.memory[idx][3:])
        
        # Track GPU memory
        if self.track_gpu_memory:
            after_memory = torch.cuda.memory_allocated() / 1024**2
            if after_memory - before_memory > 100:
                print(f"\nWarning: Large GPU memory increase: {after_memory - before_memory:.2f}MB")
        
        # Clear GPU cache periodically
        if len(self.losses) % 1000 == 0:
            torch.cuda.empty_cache()
        
        # Update target network if needed
        self.target_update_counter += 1
        if self.target_update_counter >= self.target_update_frequency:
            self.update_target_model()
            self.target_update_counter = 0
        
        # Update epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        loss_value = loss.item()
        self.losses.append(loss_value)
        return loss_value
    
    def get_learning_metrics(self):
        """Return current learning metrics with GPU info"""
        metrics = {
            'avg_loss': np.mean(self.losses[-100:]) if self.losses else 0,
            'avg_q_value': np.mean(self.avg_q_values[-100:]) if self.avg_q_values else 0,
            'action_distribution': {k: v/sum(self.action_frequencies.values()) 
                                  for k, v in self.action_frequencies.items()},
            'epsilon': self.epsilon,
            'gpu_memory_used': f"{torch.cuda.memory_allocated() / 1024**2:.2f}MB",
            'gpu_memory_cached': f"{torch.cuda.memory_reserved() / 1024**2:.2f}MB"
        }
        return metrics

def train(episodes=100, log_interval=1):
    print("\n=== Training Configuration ===")
    print(f"Episodes: {episodes}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("============================\n")
    
    env = Game2048()
    agent = QLearningAgent(16, 4)
    metrics = MetricsTracker()
    
    best_score = 0
    best_tile = 0
    
    for episode in range(episodes):
        env = Game2048()
        state = env.get_state()
        total_reward = 0
        steps = 0
        start_time = time.time()
        
        while not env.is_game_over() and not env.is_won():
            valid_actions = env.get_valid_actions()
            action = agent.act(state, valid_actions)
            
            if action is None:
                break
            
            env.make_move(action)
            next_state = env.get_state()
            reward = env.get_reward()
            done = env.is_game_over() or env.is_won()
            
            agent.remember(state, action, reward, next_state, done)
            loss = agent.replay()
            
            state = next_state
            total_reward += reward
            steps += 1
            
            if steps % 10 == 0:
                learning_metrics = agent.get_learning_metrics()
                print(f"\rEpisode {episode+1}/{episodes} - Steps: {steps} - Score: {env.score} "
                      f"- Max Tile: {env.get_max_tile()} - Loss: {learning_metrics['avg_loss']:.4f} "
                      f"- Avg Q: {learning_metrics['avg_q_value']:.4f} - ε: {learning_metrics['epsilon']:.3f}", 
                      end="")
        
        episode_time = time.time() - start_time
        metrics.add_episode(env.score, env.get_max_tile(), steps, episode_time)
        
        if env.score > best_score:
            best_score = env.score
        if env.get_max_tile() > best_tile:
            best_tile = env.get_max_tile()
        
        if episode % log_interval == 0:
            learning_metrics = agent.get_learning_metrics()
            print(f"\n\nEpisode {episode+1} Summary:")
            print(f"Score: {env.score} (Best: {best_score})")
            print(f"Max Tile: {env.get_max_tile()} (Best: {best_tile})")
            print(f"Steps: {steps}")
            print(f"Average Loss: {learning_metrics['avg_loss']:.4f}")
            print(f"Average Q-Value: {learning_metrics['avg_q_value']:.4f}")
            print("Action Distribution:", 
                  {k: f"{v:.2%}" for k, v in learning_metrics['action_distribution'].items()})
            print("\nCurrent Board:")
            print(env)
            print("-" * 40)
    
    metrics.plot_metrics()
    return agent, metrics

if __name__ == "__main__":
    try:
        agent, metrics = train(episodes=1000, log_interval=20)
    except Exception as e:
        print(f"\nError: {str(e)}")
        raise