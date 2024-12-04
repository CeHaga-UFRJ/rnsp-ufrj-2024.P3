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
from typing import List, Tuple, Dict
from abc import ABC, abstractmethod
import heapq

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
        reward = 0
        
        if self.max_tile > self.previous_max_tile:
            if self.previous_max_tile == 0:
                reward += 2.0
            else:
                reward += 2.0 * (np.log2(self.max_tile) - np.log2(self.previous_max_tile))
        
        if self.moves_since_merge > 5:
            reward -= 0.5 * (self.moves_since_merge - 5)
        
        empty_tiles = self.get_empty_tiles()
        reward += 0.1 * empty_tiles
        
        reward += 0.2 * self.get_monotonicity()
        reward += 0.1 * self.get_smoothness()
        
        return float(reward)  # Ensure we return a number
    
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

class PrioritizedReplayBuffer:
    def __init__(self, maxlen=20000, alpha=0.6, beta=0.4):
        self.maxlen = maxlen
        self.memory = deque(maxlen=maxlen)
        self.priorities = deque(maxlen=maxlen)
        self.alpha = alpha      # Priority exponent
        self.beta = beta        # Importance sampling weight
        self.epsilon = 1e-6     # Small constant to avoid zero probabilities
        
    def add(self, state, action, reward, next_state, done):
        # New experiences get max priority
        max_priority = max(self.priorities) if self.priorities else 1.0
        self.memory.append((state, action, reward, next_state, done))
        self.priorities.append(max_priority)
        
        # Maintain maxlen
        if len(self.memory) > self.maxlen:
            self.memory.popleft()
            self.priorities.popleft()
    
    def sample(self, batch_size):
        total = len(self.memory)
        
        # Calculate sampling probabilities
        probs = np.array(self.priorities) ** self.alpha
        probs /= probs.sum()
        
        # Sample indices and calculate importance weights
        indices = np.random.choice(total, batch_size, p=probs)
        samples = [self.memory[idx] for idx in indices]
        
        # Calculate importance sampling weights
        weights = (total * probs[indices]) ** (-self.beta)
        weights /= weights.max()  # Normalize weights
        
        return samples, indices, weights
    
    def update_priorities(self, indices, errors):
        for idx, error in zip(indices, errors):
            self.priorities[idx] = error.item() + self.epsilon
    
    def __len__(self):
        return len(self.memory)

class DWNQLearningAgent:
    def __init__(self, state_size, action_size):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")
        
        self.device = torch.device("cuda")
        print(f"\nGPU: {torch.cuda.get_device_name(0)}")
        torch.backends.cudnn.benchmark = True
        
        # Core parameters
        self.state_size = state_size
        self.action_size = action_size
        self.memory = PrioritizedReplayBuffer(maxlen=50000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.997
        self.batch_size = 128
        self.learning_rate = 0.001
        self.tau = 0.005  # Soft update parameter
        
        # Curriculum learning
        self.curriculum_stage = 0
        self.stage_thresholds = {
            0: 0,     # Initial stage
            1: 256,   # Basic merging
            2: 512,   # Advanced merging
            3: 1024,  # Expert merging
            4: 2048   # Master level
        }
        
        # Build networks
        self.policy_net = self._build_model()
        self.target_net = self._build_model()
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss(reduction='none')  # For prioritized replay
        
        # Metrics
        self.losses = []
        self.avg_q_values = []
        self.action_frequencies = {i: 0 for i in range(action_size)}
        self.stage_history = []
        
    def _build_model(self):
        model = nn.Sequential(
            # Input preprocessing
            nn.Linear(self.state_size, self.state_size * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            # First DWN block
            dwn.LUTLayer(self.state_size * 2, 256, n=4, mapping='learnable'),
            nn.LayerNorm(256),  # Replace BatchNorm with LayerNorm
            nn.ReLU(),
            nn.Dropout(0.1),
            
            # Second DWN block
            dwn.LUTLayer(256, 128, n=4, mapping='learnable'),
            nn.LayerNorm(128),  # Replace BatchNorm with LayerNorm
            nn.ReLU(),
            nn.Dropout(0.1),
            
            # Third DWN block
            dwn.LUTLayer(128, 64, n=4, mapping='learnable'),
            nn.LayerNorm(64),   # Replace BatchNorm with LayerNorm
            nn.ReLU(),
            
            # Output layer
            nn.Linear(64, self.action_size)
        ).to(self.device)
        
        # Initialize weights
        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
        
        model.apply(init_weights)
        return model
    
    def update_curriculum(self, max_tile):
        """Update curriculum stage based on maximum tile achieved"""
        for stage, threshold in sorted(self.stage_thresholds.items(), reverse=True):
            if max_tile >= threshold and stage > self.curriculum_stage:
                self.curriculum_stage = stage
                break
    
    def get_exploration_rate(self):
        """Get curriculum-adjusted exploration rate"""
        base_epsilon = max(self.epsilon, self.epsilon_min)
        stage_bonus = 0.05 * (4 - self.curriculum_stage)  # More exploration in early stages
        return min(base_epsilon + stage_bonus, 1.0)
    
    def preprocess_state(self, state):
        """Enhanced state preprocessing with curriculum-based normalization and proper type handling"""
        try:
            # Ensure state is a numpy array
            if isinstance(state, (float, np.float64, np.float32)):
                state = np.array([state], dtype=np.float32)
            else:
                state = np.array(state, dtype=np.float32)
            
            # Reshape if needed
            if state.ndim == 1:
                state = state.reshape(-1)
            
            non_zero_mask = state > 0
            processed_state = np.zeros_like(state, dtype=np.float32)
            processed_state[non_zero_mask] = np.log2(state[non_zero_mask])
            
            # Curriculum-based normalization
            norm_factor = 11.0  # log2(2048)
            if self.curriculum_stage < 2:
                norm_factor = 8.0  # log2(256)
            elif self.curriculum_stage < 3:
                norm_factor = 9.0  # log2(512)
            
            if processed_state.max() > 0:
                processed_state = processed_state / norm_factor
            
            # Convert to tensor with proper shape
            tensor_state = torch.FloatTensor(processed_state).view(1, -1)
            return tensor_state.to(self.device)
        
        except Exception as e:
            print(f"State preprocessing error: {str(e)}")
            print(f"State type: {type(state)}")
            print(f"State value: {state}")
            raise
    
    def act(self, state, valid_actions):
        if not valid_actions:
            return None
        
        # Use curriculum-adjusted exploration rate
        if random.random() <= self.get_exploration_rate():
            action = random.choice(valid_actions)
            self.action_frequencies[action] += 1
            return action
        
        with torch.no_grad():
            state = self.preprocess_state(state)
            q_values = self.policy_net(state).cpu()
            self.avg_q_values.append(q_values.mean().item())
            
            # Mask invalid actions
            action_mask = torch.full((self.action_size,), float('-inf'))
            action_mask[valid_actions] = 0
            masked_q_values = q_values + action_mask
            
            action = masked_q_values.argmax().item()
            self.action_frequencies[action] += 1
            return action
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)
    
    def replay(self):
        if len(self.memory) < self.batch_size:
            return None
        
        try:
            # Sample with priorities
            samples, indices, weights = self.memory.sample(self.batch_size)
            weights = torch.FloatTensor(weights).to(self.device)
            
            # Prepare batch data with proper type handling
            states = torch.cat([self.preprocess_state(np.array(state)) 
                            for state, *_ in samples])
            actions = torch.tensor([action for _, action, *_ in samples], 
                                device=self.device, dtype=torch.long).view(-1, 1)
            rewards = torch.tensor([reward for _, _, reward, *_ in samples], 
                                device=self.device, dtype=torch.float32)
            next_states = torch.cat([self.preprocess_state(np.array(next_state)) 
                                for _, _, _, next_state, _ in samples])
            dones = torch.tensor([float(done) for _, _, _, _, done in samples], 
                            device=self.device, dtype=torch.float32)
            
            # Compute Q values without autocast
            self.optimizer.zero_grad()
            
            # Forward pass with policy network
            current_q_values = self.policy_net(states).gather(1, actions).squeeze()
            
            with torch.no_grad():
                next_q_values = self.target_net(next_states).max(1)[0]
                target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
            
            # Compute weighted loss for prioritized replay
            elementwise_loss = self.criterion(current_q_values, target_q_values)
            weighted_loss = (elementwise_loss * weights).mean()
            
            weighted_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Update target network with soft update
            with torch.no_grad():
                for target_param, policy_param in zip(self.target_net.parameters(), 
                                                    self.policy_net.parameters()):
                    target_param.data.copy_(
                        self.tau * policy_param.data + (1 - self.tau) * target_param.data
                    )
            
            # Update priorities
            self.memory.update_priorities(indices, elementwise_loss)
            
            # Update exploration rate
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
            
            loss_value = weighted_loss.item()
            self.losses.append(loss_value)
            
            return loss_value
            
        except Exception as e:
            print(f"Replay error: {str(e)}")
            print(f"Sample types:")
            for i, (state, action, reward, next_state, done) in enumerate(samples[:5]):
                print(f"Sample {i}:")
                print(f"  State type: {type(state)}, shape: {np.array(state).shape}")
                print(f"  Action type: {type(action)}")
                print(f"  Reward type: {type(reward)}")
                print(f"  Next state type: {type(next_state)}, shape: {np.array(next_state).shape}")
                print(f"  Done type: {type(done)}")
            raise
    
    def get_learning_metrics(self):
        return {
            'avg_loss': np.mean(self.losses[-100:]) if self.losses else 0,
            'avg_q_value': np.mean(self.avg_q_values[-100:]) if self.avg_q_values else 0,
            'action_distribution': {k: v/sum(self.action_frequencies.values()) 
                                  for k, v in self.action_frequencies.items()},
            'epsilon': self.epsilon,
            'curriculum_stage': self.curriculum_stage,
            'gpu_memory_used': f"{torch.cuda.memory_allocated() / 1024**2:.2f}MB",
            'gpu_memory_cached': f"{torch.cuda.memory_reserved() / 1024**2:.2f}MB"
        }

def enhanced_reward_function(env, agent):
    """Enhanced reward function that scales with curriculum stage"""
    base_reward = env.get_reward()  # Get basic reward from environment
    
    # Curriculum stage bonuses
    stage_multiplier = 1.0 + (0.2 * agent.curriculum_stage)
    
    # Additional rewards based on game state
    monotonicity_reward = env.get_monotonicity() * (0.2 + 0.1 * agent.curriculum_stage)
    smoothness_reward = env.get_smoothness() * (0.1 + 0.05 * agent.curriculum_stage)
    empty_tiles_reward = env.get_empty_tiles() * (0.3 - 0.05 * agent.curriculum_stage)
    
    # Merge rewards
    merge_reward = 0
    if env.merge_count > 0:
        merge_reward = 0.5 * env.merge_count * stage_multiplier
    
    # Penalty for moves without merges
    stagnation_penalty = -0.2 * env.moves_since_merge if env.moves_since_merge > 3 else 0
    
    total_reward = (
        base_reward * stage_multiplier +
        monotonicity_reward +
        smoothness_reward +
        empty_tiles_reward +
        merge_reward +
        stagnation_penalty
    )
    
    return float(total_reward)

def train(episodes=1000, log_interval=25, save_interval=100):
    print("\n=== Training Configuration ===")
    print(f"Episodes: {episodes}")
    print(f"Log Interval: {log_interval}")
    print(f"Save Interval: {save_interval}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("============================\n")
    
    env = Game2048()
    agent = DWNQLearningAgent(16, 4)
    metrics = MetricsTracker()
    
    best_score = 0
    best_tile = 0
    running_scores = deque(maxlen=100)
    
    checkpoint_dir = 'checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    try:
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
                reward = enhanced_reward_function(env, agent)
                done = env.is_game_over() or env.is_won()
                
                agent.remember(state, action, reward, next_state, done)
                loss = agent.replay()
                
                state = next_state
                total_reward += reward
                steps += 1
                
                # Update curriculum based on current max tile
                agent.update_curriculum(env.get_max_tile())
                
                if steps % 10 == 0:
                    learning_metrics = agent.get_learning_metrics()
                    print(f"\rEpisode {episode+1}/{episodes} - Steps: {steps} "
                          f"- Score: {env.score} - Max Tile: {env.get_max_tile()} "
                          f"- Loss: {learning_metrics['avg_loss']:.4f} "
                          f"- Avg Q: {learning_metrics['avg_q_value']:.4f} "
                          f"- ε: {learning_metrics['epsilon']:.3f} "
                          f"- Stage: {learning_metrics['curriculum_stage']}", end="")
            
            episode_time = time.time() - start_time
            metrics.add_episode(env.score, env.get_max_tile(), steps, episode_time)
            running_scores.append(env.score)
            
            if env.score > best_score:
                best_score = env.score
            if env.get_max_tile() > best_tile:
                best_tile = env.get_max_tile()
                
                # Save checkpoint on new best tile
                checkpoint_path = os.path.join(checkpoint_dir, 
                                             f'best_tile_{best_tile}_episode_{episode}.pt')
                torch.save({
                    'episode': episode,
                    'model_state_dict': agent.policy_net.state_dict(),
                    'optimizer_state_dict': agent.optimizer.state_dict(),
                    'best_score': best_score,
                    'best_tile': best_tile,
                    'curriculum_stage': agent.curriculum_stage
                }, checkpoint_path)
            
            if episode % log_interval == 0:
                learning_metrics = agent.get_learning_metrics()
                print(f"\n\nEpisode {episode+1} Summary:")
                print(f"Score: {env.score} (Best: {best_score})")
                print(f"Max Tile: {env.get_max_tile()} (Best: {best_tile})")
                print(f"Steps: {steps}")
                print(f"Average Loss: {learning_metrics['avg_loss']:.4f}")
                print(f"Average Q-Value: {learning_metrics['avg_q_value']:.4f}")
                print(f"Curriculum Stage: {learning_metrics['curriculum_stage']}")
                print(f"Average Score (last 100): {np.mean(running_scores):.1f}")
                print("Action Distribution:", 
                      {k: f"{v:.2%}" for k, v in learning_metrics['action_distribution'].items()})
                print("\nCurrent Board:")
                print(env)
                print("-" * 40)
            
            if episode % save_interval == 0:
                checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_episode_{episode}.pt')
                torch.save({
                    'episode': episode,
                    'model_state_dict': agent.policy_net.state_dict(),
                    'optimizer_state_dict': agent.optimizer.state_dict(),
                    'best_score': best_score,
                    'best_tile': best_tile,
                    'curriculum_stage': agent.curriculum_stage
                }, checkpoint_path)
        
        metrics.plot_metrics()
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"\nError during training: {str(e)}")
        raise
    finally:
        # Save final model
        torch.save({
            'episode': episodes,
            'model_state_dict': agent.policy_net.state_dict(),
            'optimizer_state_dict': agent.optimizer.state_dict(),
            'best_score': best_score,
            'best_tile': best_tile,
            'curriculum_stage': agent.curriculum_stage
        }, os.path.join(checkpoint_dir, 'final_model.pt'))
    
    return agent, metrics

if __name__ == "__main__":
    try:
        agent, metrics = train(episodes=1000, log_interval=25, save_interval=100)
    except Exception as e:
        print(f"\nError: {str(e)}")
        raise