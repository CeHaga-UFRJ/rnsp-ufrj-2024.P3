import numpy as np
import wisardpkg as wp
from abc import ABC, abstractmethod
from collections import deque
import random
from enum import Enum

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

class Direction(Enum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3

    def rotate(direction, rotation, symmetry):
        rotate = Direction((direction.value - rotation) % 4)
        if symmetry:
            if rotate == Direction.LEFT:
                rotate = Direction.RIGHT
            elif rotate == Direction.RIGHT:
                rotate = Direction.LEFT
        return rotate

class Game2048(GameEnvironment):
    def __init__(self, size=4, initial_pieces=2, base=2):
        self.size = size
        self.board = [[0 for _ in range(size)] for _ in range(size)]
        self.game_ended = False
        self.rounds = 0
        self.score = 0
        for _ in range(initial_pieces):
            self.random_piece()

    def get_state(self):
        return np.array(self.board).flatten()
    
    def get_valid_actions(self):
        valid_actions = set()
        for direction in Direction:
            for i in range(self.size):
                for j in range(self.size):
                    if i < self.size - 1:
                        if direction == Direction.UP and self.board[i][j] == self.board[i + 1][j]:
                            valid_actions.add(direction)
                        elif direction == Direction.DOWN and self.board[i][j] == self.board[i + 1][j]:
                            valid_actions.add(direction)
                        
                        if direction == Direction.UP and self.board[i][j] == 0 and self.board[i + 1][j] != 0:
                            valid_actions.add(direction)
                        elif direction == Direction.DOWN and self.board[i][j] == 0 and self.board[i + 1][j] != 0:
                            valid_actions.add(direction)
                    if j < self.size - 1:
                        if direction == Direction.LEFT and self.board[i][j] == self.board[i][j + 1]:
                            valid_actions.add(direction)
                        elif direction == Direction.RIGHT and self.board[i][j] == self.board[i][j + 1]:
                            valid_actions.add(direction)

                        if direction == Direction.LEFT and self.board[i][j] == 0 and self.board[i][j + 1] != 0:
                            valid_actions.add(direction)
                        elif direction == Direction.RIGHT and self.board[i][j] == 0 and self.board[i][j + 1] != 0:
                            valid_actions.add(direction)
        return list(valid_actions)

    def random_piece(self):
        piece = 1 if random.randint(0, 9) < 9 else 2
        empty_cells = [(i, j) for i in range(self.size) for j in range(self.size) if self.board[i][j] == 0]
        if not empty_cells: return None
        i, j = empty_cells[random.randint(0, len(empty_cells) - 1)]
        self.board[i][j] = piece
        return {
            "piece": piece,
            "position": (i, j)
        }

    def make_move(self, direction):
        self.rounds += 1
        new_score = 0
        new_board = None

        if direction == Direction.UP:
            new_board, new_score = self.move_up()
        elif direction == Direction.DOWN:
            new_board, new_score = self.move_down()
        elif direction == Direction.LEFT:
            new_board, new_score = self.move_left()
        elif direction == Direction.RIGHT:
            new_board, new_score = self.move_right()

        self.board = new_board
        self.score += new_score
        new_piece = self.random_piece()

        return self.board

    def move_up(self):
        new_board = [[0 for _ in range(self.size)] for _ in range(self.size)]
        new_score = 0
        for j in range(self.size):
            pieces = [piece[j] for piece in self.board if piece[j] != 0]
            merged_pieces, current_score = self.merge_pieces(pieces)
            new_score += current_score
            for i in range(self.size):
                new_board[i][j] = merged_pieces[i]
        return new_board, new_score

    def move_down(self):
        new_board = [[0 for _ in range(self.size)] for _ in range(self.size)]
        new_score = 0
        for j in range(self.size):
            pieces = [piece[j] for piece in self.board if piece[j] != 0]
            merged_pieces, current_score = self.merge_pieces(pieces, inverse=True)
            new_score += current_score
            for i in range(self.size):
                new_board[i][j] = merged_pieces[i]
        return new_board, new_score

    def move_right(self):
        new_board = [[0 for _ in range(self.size)] for _ in range(self.size)]
        new_score = 0
        for i in range(self.size):
            pieces = [piece for piece in self.board[i] if piece != 0]
            merged_pieces, current_score = self.merge_pieces(pieces, inverse=True)
            new_board[i] = merged_pieces
            new_score += current_score
        return new_board, new_score

    def move_left(self):
        new_board = [[0 for _ in range(self.size)] for _ in range(self.size)]
        new_score = 0
        for i in range(self.size):
            pieces = [piece for piece in self.board[i] if piece != 0]
            merged_pieces, current_score = self.merge_pieces(pieces)
            new_board[i] = merged_pieces
            new_score += current_score
        return new_board, new_score

    def merge_pieces(self, pieces, inverse=False):
        merged_pieces = []
        new_score = 0
        i = 0

        if inverse:
            pieces = pieces[::-1]

        while i < len(pieces):
            if i < len(pieces) - 1 and pieces[i] == pieces[i + 1]:
                new_piece = pieces[i] + 1
                merged_pieces.append(new_piece)
                new_score += 2**new_piece
                i += 2
            else:
                merged_pieces.append(pieces[i])
                i += 1

        remaining = self.size - len(merged_pieces)
        merged_pieces += [0] * remaining

        if inverse:
            merged_pieces = merged_pieces[::-1]

        return merged_pieces, new_score

    def get_reward(self):
        return self.score
    
    def is_game_over(self):
        return not self.get_valid_actions()

    def __str__(self):
        board_str = ""
        for row in self.board:
            board_str += " ".join(map(str, row)) + "\n"
        return board_str

class TicTacToe(GameEnvironment):
    def __init__(self):
        self.board = np.zeros((3, 3))
        self.current_player = 1
        
    def get_state(self):
        return self.board.flatten()
    
    def get_valid_actions(self):
        return [(i, j) for i in range(3) for j in range(3) if self.board[i][j] == 0]
    
    def make_move(self, action):
        i, j = action
        if self.board[i][j] == 0:
            self.board[i][j] = self.current_player
            self.current_player *= -1
            return True
        return False
    
    def is_game_over(self):
        # Check rows, columns and diagonals
        for i in range(3):
            if abs(sum(self.board[i])) == 3 or abs(sum(self.board[:, i])) == 3:
                return True
        if abs(np.trace(self.board)) == 3 or abs(np.trace(np.fliplr(self.board))) == 3:
            return True
        return len(self.get_valid_actions()) == 0
    
    def get_reward(self):
        if self.is_game_over():
            if self.check_winner() == 1:
                return 1
            elif self.check_winner() == -1:
                return -1
        return 0
    
    def check_winner(self):
        # Check rows, columns and diagonals
        for i in range(3):
            if sum(self.board[i]) == 3 or sum(self.board[:, i]) == 3:
                return 1
            elif sum(self.board[i]) == -3 or sum(self.board[:, i]) == -3:
                return -1
        if np.trace(self.board) == 3 or np.trace(np.fliplr(self.board)) == 3:
                return 1
        elif np.trace(self.board) == -3 or np.trace(np.fliplr(self.board)) == -3:
                return -1
        return 0
        
    def __str__(self):
        board_str = ""
        for i in range(3):
            for j in range(3):
                if self.board[i][j] == 1:
                    board_str += "X"
                elif self.board[i][j] == -1:
                    board_str += "O"
                else:
                    board_str += " "
                if j < 2: board_str += "|"
            board_str += "\n"
            if i < 2: board_str += "-----\n"
        return board_str
    

class WisardDQN:
    def __init__(self, state_size, action_size, addressing_size=3):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.model = wp.Wisard(addressing_size, verbose=True, returnConfidence=True)

    def preprocess_board(self, board):
        binary_board = []
        for cell in board[0]:
            if cell == 0:
                binary_board.extend([0, 0])
            elif cell == 1:
                binary_board.extend([0, 1])
            elif cell == -1:
                binary_board.extend([1, 0])
        return [binary_board]
        
    def _preprocess_state(self, state):
        # Convert state to binary representation
        state_normalized = (state - np.min(state)) / (np.max(state) - np.min(state) + 1e-10)
        return (state_normalized > 0.5).astype(int)
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def classify_confidence(self, state_bin):
        # This method should return a dictionary with action: confidence pairs
        classification = self.model.classify(self.preprocess_board(state_bin))
        print('oi')
        print(state_bin)
        print(self.preprocess_board(state_bin))
        print(classification)
        return {int(result['class']): result['confidence'] for result in classification if result['class']}
        
    
    def act(self, state, valid_actions):
        if random.random() <= self.epsilon:
            return random.choice(valid_actions)
        
        state_bin = self._preprocess_state(state)
        action_values = self.classify_confidence(state_bin.reshape(1, -1))
        
        # Filter only valid actions
        valid_action_values = {action: action_values[action] for action in valid_actions}
        return max(valid_action_values.items(), key=lambda x: x[1])[0]
    
    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        
        minibatch = random.sample(self.memory, batch_size)
        
        for state, action, reward, next_state, done in minibatch:
            state_bin = self._preprocess_state(state)
            next_state_bin = self._preprocess_state(next_state)
            
            target = reward
            if not done:
                next_values = self.classify_confidence(next_state_bin.reshape(1, -1))
                target = reward + self.gamma * np.max(list(next_values.values()))
            
            # Train the model
            print('a')
            print(state_bin.reshape(1, -1))
            print(self.preprocess_board(state_bin.reshape(1, -1)))
            print(target)
            print('c')
            self.model.train(self.preprocess_board(state_bin.reshape(1, -1)), [str(target)])
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

def train_agent():
    state_size = 9  # 3x3 board
    action_size = 9  # 9 possible positions
    agent = WisardDQN(state_size, action_size)
    
    episodes = 1000
    batch_size = 32
    
    for episode in range(episodes):
        env = TicTacToe()
        state = env.get_state()
        total_reward = 0
        print_moves = episode % 100 == 0
        if print_moves:
            print(env)
        
        while not env.is_game_over():
            valid_actions = env.get_valid_actions()
            action = agent.act(state, valid_actions)
            
            env.make_move(action)
            next_state = env.get_state()

            if print_moves:
                print(f"Player {env.current_player}: {action}")
                print(env)

            reward = env.get_reward()
            done = env.is_game_over()
            
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            
            agent.replay(batch_size)
        
        if episode % 100 == 0:
            print(f"Episode: {episode}, Total Reward: {total_reward}, Epsilon: {agent.epsilon:.2f}")

if __name__ == "__main__":
    train_agent()
