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

# Define uma classe base abstrata para o ambiente do jogo, que é usada como blueprint para implementar diferentes jogos
class GameEnvironment(ABC):
    @abstractmethod
    def get_state(self):
        # Retorna o estado atual do jogo (representação do tabuleiro)
        pass
    
    @abstractmethod
    def get_valid_actions(self):
        # Retorna as ações válidas possíveis no estado atual
        pass
    
    @abstractmethod
    def make_move(self, action):
        # Executa a ação fornecida e altera o estado do jogo
        pass
    
    @abstractmethod
    def is_game_over(self):
        # Retorna True se o jogo acabou, caso contrário, False
        pass
    
    @abstractmethod
    def get_reward(self):
        # Retorna a recompensa do estado atual do jogo
        pass

# Implementação do ambiente do jogo 2048, derivado de GameEnvironment
class Game2048(GameEnvironment):
    def __init__(self):
        # Inicializa o tabuleiro com zeros, a pontuação e outras variáveis auxiliares
        self.board = np.zeros((4, 4), dtype=int)
        self.score = 0
        self.max_tile = 0
        self.merge_count = 0
        self.moves_since_merge = 0
        self.previous_max_tile = 0
        self._add_new_tile()  # Adiciona o primeiro tile
        self._add_new_tile()  # Adiciona o segundo tile

    def get_state(self):
        # Retorna o estado do jogo como uma matriz achatada (1D)
        return self.board.flatten()
    
    def get_state_2d(self):
        # Retorna uma cópia do tabuleiro como matriz 2D
        return np.copy(self.board)
    
    def get_monotonicity(self):
        # Calcula uma métrica para medir a monotonicidade do tabuleiro (valores crescentes ou decrescentes)
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
        # Mede a suavidade do tabuleiro, penalizando diferenças grandes entre valores adjacentes
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
        # Retorna o número de tiles vazios no tabuleiro
        return len(np.where(self.board == 0)[0])
    
    def get_valid_actions(self):
        # Retorna uma lista de ações válidas (0: cima, 1: direita, 2: baixo, 3: esquerda)
        valid_actions = []
        for action in range(4):  # Itera sobre todas as ações possíveis
            if self._is_valid_move(action):  # Verifica se a ação é válida
                valid_actions.append(action)
        return valid_actions
    
    def _is_valid_move(self, action):
        # Determina se uma ação é válida sem alterar o estado do tabuleiro real
        temp_board = np.copy(self.board)
        return self._move(temp_board, action, test_only=True)
    
    def make_move(self, action):
        # Executa a ação no tabuleiro real e retorna True se o movimento for válido
        self.previous_max_tile = self.max_tile
        result = self._move(self.board, action)
        if not result:
            self.moves_since_merge += 1
        return result
    
    def _move(self, board, action, test_only=False):
        # Realiza o movimento no tabuleiro, com suporte para apenas teste (sem modificar o estado real)
        original_board = np.copy(board)
        
        # Rotaciona o tabuleiro dependendo da direção do movimento
        if action == 0:  # cima
            board = np.rot90(board)
        elif action == 1:  # direita
            board = np.rot90(board, 2)
        elif action == 2:  # baixo
            board = np.rot90(board, 3)
        
        moved = False  # Marca se o movimento resultou em alterações no tabuleiro
        merges_this_move = 0  # Contador de fusões
        
        for i in range(4):
            row = board[i]
            row = row[row != 0]  # Remove zeros
            
            # Fusão de tiles adjacentes
            j = 0
            while j < len(row) - 1:
                if row[j] == row[j + 1]:
                    row[j] *= 2  # Dobra o valor do tile
                    if not test_only:  # Apenas aplica alterações reais no tabuleiro real
                        self.score += row[j]  # Atualiza a pontuação
                        self.max_tile = max(self.max_tile, row[j])  # Atualiza o maior tile
                        merges_this_move += 1
                    row = np.delete(row, j + 1)  # Remove o segundo tile da fusão
                j += 1
            
            # Reconstrói a linha com zeros à direita
            new_row = np.zeros(4, dtype=int)
            new_row[:len(row)] = row
            
            # Verifica se a linha foi alterada
            if not np.array_equal(board[i], new_row):
                moved = True
            board[i] = new_row
        
        # Reverte a rotação
        if action == 0:
            board = np.rot90(board, 3)
        elif action == 1:
            board = np.rot90(board, 2)
        elif action == 2:
            board = np.rot90(board)
        
        # Adiciona um novo tile se o movimento foi válido
        if moved and not test_only:
            self._add_new_tile()
            self.merge_count += merges_this_move
            self.moves_since_merge = 0
        
        return moved
    
    def _add_new_tile(self):
        # Adiciona um novo tile (2 ou 4) em uma posição vazia aleatória no tabuleiro
        empty_cells = list(zip(*np.where(self.board == 0)))
        if empty_cells:
            x, y = random.choice(empty_cells)
            self.board[x][y] = np.random.choice([2, 4], p=[0.9, 0.1])
    
    def is_game_over(self):
        # Verifica se o jogo acabou (nenhuma ação válida restante)
        return len(self.get_valid_actions()) == 0
    
    def get_reward(self):
        # Função de recompensa que incentiva estratégias vencedoras
        reward = 0
        
        # Recompensa maior para alcançar tiles mais altos
        if self.max_tile > self.previous_max_tile:
            reward += 2.0 ** (np.log2(self.max_tile) - 7)
        
        # Recompensa por manter tiles vazios
        empty_tiles = self.get_empty_tiles()
        reward += 0.1 * (empty_tiles ** 2)
        
        # Penalidade para movimentos sem fusões
        if self.moves_since_merge > 3:
            reward -= 2.0 ** (self.moves_since_merge - 3)
        
        # Recompensa pela estrutura do tabuleiro
        mono_score = self.get_monotonicity()
        smooth_score = self.get_smoothness()
        reward += 0.5 * mono_score + 0.3 * smooth_score
        
        # Recompensa extra para alcançar milestones (128, 256, etc.)
        milestone_tiles = [128, 256, 512, 1024, 2048]
        if self.max_tile in milestone_tiles and self.max_tile > self.previous_max_tile:
            reward += 10.0 * np.log2(self.max_tile)
        
        return float(reward)
    
    def get_max_tile(self):
        # Retorna o maior valor presente no tabuleiro
        return np.max(self.board)
    
    def is_won(self):
        # Verifica se o jogador alcançou o objetivo (2048)
        return self.get_max_tile() >= 2048
    
    def __str__(self):
        # Representação do tabuleiro como string para fácil visualização
        cell_width = max(len(str(cell)) for cell in self.board.flatten())
        rows = []
        for row in self.board:
            row_str = ' '.join(str(cell).rjust(cell_width) for cell in row)
            rows.append(row_str)
        return '\n'.join(rows)

class MetricsTracker:
    def __init__(self):
        # Inicializa as listas para rastrear métricas durante o treinamento
        self.episode_scores: List[int] = []  # Lista para armazenar as pontuações de cada episódio
        self.episode_max_tiles: List[int] = []  # Lista para armazenar o maior tile alcançado por episódio
        self.episode_steps: List[int] = []  # Lista para armazenar o número de passos por episódio
        self.episode_times: List[float] = []  # Lista para armazenar o tempo gasto em cada episódio
        self.wins: int = 0  # Contador de vitórias (quando o tile 2048 é alcançado)
        
        # Diretório onde os gráficos serão salvos
        self.graphics_dir = 'graphics'
        if not os.path.exists(self.graphics_dir):  # Verifica se o diretório existe
            os.makedirs(self.graphics_dir)  # Cria o diretório se ele não existir
    
    def add_episode(self, score: int, max_tile: int, steps: int, duration: float):
        # Adiciona os dados de um episódio ao rastreador de métricas
        self.episode_scores.append(score)  # Adiciona a pontuação do episódio
        self.episode_max_tiles.append(max_tile)  # Adiciona o maior tile alcançado
        self.episode_steps.append(steps)  # Adiciona o número de passos
        self.episode_times.append(duration)  # Adiciona a duração do episódio
        if max_tile >= 2048:  # Incrementa o contador de vitórias se o tile 2048 foi alcançado
            self.wins += 1
    
    def plot_metrics(self):
        # Plota gráficos para análise das métricas de treinamento
        plt.style.use('classic')  # Define o estilo clássico do matplotlib
        sns.set_palette("husl")  # Define uma paleta de cores para os gráficos
        
        # Cria um timestamp único para nomear os arquivos
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Cria uma figura com 4 subgráficos
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 15))
        fig.suptitle('2048 DWN Agent Learning Progress', fontsize=16, y=0.95)  # Título geral dos gráficos
        
        episodes = range(len(self.episode_scores))  # Lista dos episódios
        
        # Subgráfico 1: Evolução das pontuações ao longo dos episódios
        ax1.plot(episodes, self.episode_scores, label='Score', linewidth=1)  # Pontuações por episódio
        z = np.polyfit(episodes, self.episode_scores, 1)  # Ajusta uma linha de tendência linear
        p = np.poly1d(z)  # Cria a função da linha de tendência
        ax1.plot(episodes, p(episodes), "r--", alpha=0.8, label=f'Trend (slope: {z[0]:.2f})')  # Adiciona a linha de tendência
        ax1.set_title('Game Score Evolution', fontsize=14, pad=10)  # Título do subgráfico
        ax1.set_xlabel('Episode Number', fontsize=12)  # Rótulo do eixo x
        ax1.set_ylabel('Score', fontsize=12)  # Rótulo do eixo y
        ax1.grid(True, linestyle='--', alpha=0.7)  # Grade no subgráfico
        ax1.legend(fontsize=10)  # Legenda
        
        # Subgráfico 2: Maior tile alcançado em cada episódio
        ax2.plot(episodes, self.episode_max_tiles, label='Max Tile', linewidth=1)
        ax2.set_title('Maximum Tile Achieved per Episode', fontsize=14, pad=10)
        ax2.set_xlabel('Episode Number', fontsize=12)
        ax2.set_ylabel('Maximum Tile Value', fontsize=12)
        ax2.grid(True, linestyle='--', alpha=0.7)
        ax2.legend(fontsize=10)
        
        # Subgráfico 3: Número de passos por episódio
        ax3.plot(episodes, self.episode_steps, label='Steps', linewidth=1)
        ax3.set_title('Steps per Episode', fontsize=14, pad=10)
        ax3.set_xlabel('Episode Number', fontsize=12)
        ax3.set_ylabel('Number of Steps', fontsize=12)
        ax3.grid(True, linestyle='--', alpha=0.7)
        ax3.legend(fontsize=10)
        
        # Subgráfico 4: Médias móveis das pontuações e tiles máximos
        window = 20  # Janela de suavização
        scores_ma = np.convolve(self.episode_scores, np.ones(window)/window, mode='valid')  # Média móvel das pontuações
        tiles_ma = np.convolve(self.episode_max_tiles, np.ones(window)/window, mode='valid')  # Média móvel dos tiles
        
        ax4.plot(range(window-1, len(episodes)), scores_ma, label=f'Score (MA-{window})', linewidth=2)
        ax4.plot(range(window-1, len(episodes)), tiles_ma, label=f'Max Tile (MA-{window})', linewidth=2)
        ax4.set_title('Moving Averages of Score and Max Tile', fontsize=14, pad=10)
        ax4.set_xlabel('Episode Number', fontsize=12)
        ax4.set_ylabel('Value', fontsize=12)
        ax4.grid(True, linestyle='--', alpha=0.7)
        ax4.legend(fontsize=10)
        
        plt.tight_layout()  # Ajusta os espaçamentos dos subgráficos
        
        # Salva o gráfico em um arquivo
        filename = f'learning_metrics_{timestamp}.png'
        filepath = os.path.join(self.graphics_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')  # Salva o arquivo com alta qualidade
        plt.close()  # Fecha o gráfico
        
        print("\nTraining Summary:")  # Resumo do treinamento
        print(f"Total Episodes: {len(self.episode_scores)}")
        print(f"Total Wins (2048 achieved): {self.wins}")
        print(f"Best Score: {max(self.episode_scores)}")
        print(f"Highest Tile: {max(self.episode_max_tiles)}")
        print(f"Average Score: {np.mean(self.episode_scores):.2f}")
        print(f"Average Steps: {np.mean(self.episode_steps):.2f}")
        print(f"Average Time per Episode: {np.mean(self.episode_times):.2f}s")
        print(f"\nMetrics plot saved as: {filepath}")
        
        # Salva os dados numéricos em um arquivo .txt
        data_filename = f'learning_data_{timestamp}.txt'
        data_filepath = os.path.join(self.graphics_dir, data_filename)
        with open(data_filepath, 'w') as f:
            f.write("Episode,Score,MaxTile,Steps,Time\n")  # Cabeçalho do arquivo
            for i in range(len(self.episode_scores)):
                f.write(f"{i},{self.episode_scores[i]},{self.episode_max_tiles[i]}," +
                        f"{self.episode_steps[i]},{self.episode_times[i]:.2f}\n")  # Escreve os dados de cada episódio
        print(f"Numerical data saved as: {data_filepath}")


class QLearningAgent:
    def __init__(self, state_size, action_size):
        # Verifica a disponibilidade de GPU
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. Please check your GPU installation.")
        
        # Define o dispositivo como GPU
        self.device = torch.device("cuda")
        
        # Imprime informações sobre a GPU
        print("\n=== GPU Information ===")
        print(f"GPU Device: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory Allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
        print(f"GPU Memory Cached: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")
        print("=====================\n")
        
        # Habilita o autotuner do cuDNN para melhor desempenho
        torch.backends.cudnn.benchmark = True
        
        # Parâmetros do agente
        self.state_size = state_size  # Dimensão do estado
        self.action_size = action_size  # Número de ações possíveis
        self.memory = deque(maxlen=50000)  # Memória para armazenar experiências
        self.gamma = 0.99  # Fator de desconto
        self.epsilon = 1.0  # Taxa inicial de exploração
        self.epsilon_min = 0.01  # Taxa mínima de exploração
        self.epsilon_decay = 0.998  # Taxa de decaimento da exploração
        self.batch_size = 256  # Tamanho do lote para aprendizado
        self.learning_rate = 0.0005  # Taxa de aprendizado
        
        # Cria a rede principal com normalização por camada (LayerNorm)
        self.model = self._build_model().to(self.device)
        
        # Cria a rede-alvo (target network)
        self.target_model = self._build_model().to(self.device)
        self.update_target_model()  # Inicializa a rede-alvo com os pesos da rede principal
        
        # Define o otimizador e a função de perda
        self.optimizer = torch.optim.Adam(self.model.parameters(), 
                                          lr=self.learning_rate,
                                          weight_decay=1e-4)
        self.criterion = nn.MSELoss()  # Função de perda quadrática média (MSE)
        
        # Variáveis para rastrear métricas de aprendizado
        self.losses = []  # Lista para armazenar perdas
        self.avg_q_values = []  # Lista para armazenar os valores médios de Q
        self.action_frequencies = {i: 0 for i in range(action_size)}  # Frequência de ações tomadas
        self.track_gpu_memory = True  # Indica se o uso de memória GPU será rastreado
        self.target_update_counter = 0  # Contador para atualizar a rede-alvo
        self.target_update_frequency = 10  # Frequência de atualização da rede-alvo

    def _build_model(self):
        """Constrói o modelo de rede neural com camadas normalizadas"""
        return nn.Sequential(
            nn.Linear(self.state_size, 256),  # Primeira camada totalmente conectada
            nn.LayerNorm(256),  # Normalização por camada
            nn.ReLU(),  # Função de ativação ReLU
            nn.Dropout(0.2),  # Dropout para regularização
            
            nn.Linear(256, 512),  # Segunda camada totalmente conectada
            nn.LayerNorm(512),  # Normalização por camada
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(512, 256),  # Terceira camada totalmente conectada
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(256, self.action_size)  # Camada de saída com ações possíveis
        )
    
    def update_target_model(self):
        """Copia os pesos da rede principal para a rede-alvo"""
        self.target_model.load_state_dict(self.model.state_dict())

    def preprocess_state(self, state):
        """Pré-processa o estado normalizando os valores"""
        state_array = np.array(state, dtype=np.float32)  # Converte o estado para um array de float32
        non_zero_mask = state_array > 0  # Máscara para valores diferentes de zero
        processed_state = np.zeros_like(state_array)  # Inicializa o estado processado
        processed_state[non_zero_mask] = np.log2(state_array[non_zero_mask])  # Aplica log base 2 aos valores
        if processed_state.max() > 0:
            processed_state = processed_state / 11.0  # Normaliza os valores pelo log base 2 de 2048
        
        return torch.FloatTensor(processed_state).view(1, -1).to(self.device, non_blocking=True)

    def remember(self, state, action, reward, next_state, done):
        """Armazena a experiência na memória com prioridade"""
        max_priority = max([x[2] for x in self.memory]) if self.memory else 1.0  # Define a prioridade máxima
        self.memory.append((state, action, max_priority, reward, next_state, done))  # Adiciona à memória

    def act(self, state, valid_actions):
        """Escolhe uma ação com base no estado atual e na política ε-greedy"""
        if not valid_actions:  # Se não houver ações válidas, retorna None
            return None
        
        # Exploração: seleciona uma ação aleatória com probabilidade ε
        if random.random() <= self.epsilon:
            action = random.choice(valid_actions)
            self.action_frequencies[action] += 1
            return action
        
        # Exploração direcionada: usa o modelo para prever os valores Q
        with torch.no_grad():
            self.model.eval()  # Define o modelo para modo de avaliação
            state = self.preprocess_state(state)  # Pré-processa o estado
            q_values = self.model(state).cpu()  # Obtém os valores Q
            self.model.train()  # Retorna ao modo de treinamento
            
            self.avg_q_values.append(q_values.mean().item())  # Salva o valor médio de Q
            
            # Máscara para invalidar ações não permitidas
            action_mask = torch.full((self.action_size,), float('-inf'))  # Inicializa com valores negativos infinitos
            action_mask[valid_actions] = 0  # Permite ações válidas
            masked_q_values = q_values + action_mask  # Aplica a máscara
            
            action = masked_q_values.argmax().item()  # Escolhe a ação com o maior valor Q
            self.action_frequencies[action] += 1  # Atualiza a frequência da ação
            return action

    def replay(self):
        """Realiza o aprendizado por meio de uma amostra de experiências"""
        if len(self.memory) < self.batch_size:  # Verifica se há experiências suficientes na memória
            return None
        
        if self.track_gpu_memory:  # Rastreia o uso de memória GPU antes do replay
            before_memory = torch.cuda.memory_allocated() / 1024**2
        
        # Seleciona uma amostra de experiências com prioridades
        priorities = np.array([x[2] for x in self.memory])  # Obtém as prioridades
        probs = priorities ** 0.6  # Ajusta as probabilidades com um fator de priorização
        probs /= probs.sum()  # Normaliza as probabilidades
        
        indices = np.random.choice(len(self.memory), self.batch_size, p=probs)  # Amostra índices com base nas probabilidades
        batch = [self.memory[idx] for idx in indices]  # Obtém as experiências correspondentes
        
        # Calcula os pesos de amostragem com base na importância
        weights = (len(self.memory) * probs[indices]) ** (-0.4)  # Fator de importância
        weights /= weights.max()  # Normaliza os pesos
        weights = torch.FloatTensor(weights).to(self.device)
        
        # Prepara os tensores de entrada para o aprendizado
        states = torch.cat([self.preprocess_state(state[0]) for state in batch])
        actions = torch.tensor([state[1] for state in batch], device=self.device, dtype=torch.long).view(-1, 1)
        rewards = torch.tensor([state[3] for state in batch], device=self.device, dtype=torch.float32)
        next_states = torch.cat([self.preprocess_state(state[4]) for state in batch])
        dones = torch.tensor([float(state[5]) for state in batch], device=self.device, dtype=torch.float32)
        
        # Calcula os valores Q e realiza a retropropagação
        self.optimizer.zero_grad()
        
        with torch.amp.autocast(device_type='cuda'):  # Usa precisão mista para acelerar o treinamento
            current_q_values = self.model(states).gather(1, actions).squeeze()  # Obtém os valores Q atuais
            
            with torch.no_grad():  # Calcula os valores Q-alvo sem rastrear gradientes
                next_q_values = self.target_model(next_states)
                next_q_max = next_q_values.max(1)[0]
                target_q_values = rewards + (1 - dones) * self.gamma * next_q_max
            
            losses = self.criterion(current_q_values, target_q_values)  # Calcula a perda
            loss = (losses * weights).mean()  # Ajusta a perda com os pesos de amostragem
        
        loss.backward()  # Calcula os gradientes
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)  # Clipa os gradientes
        self.optimizer.step()  # Atualiza os pesos do modelo
        
        # Atualiza as prioridades na memória com base nos erros TD
        with torch.no_grad():
            td_errors = abs(target_q_values - current_q_values).cpu().numpy()
        
        for idx, error in zip(indices, td_errors):
            self.memory[idx] = (*self.memory[idx][:2], error + 1e-6, *self.memory[idx][3:])  # Atualiza prioridades
        
        # Rastreia o uso de memória GPU após o replay
        if self.track_gpu_memory:
            after_memory = torch.cuda.memory_allocated() / 1024**2
            if after_memory - before_memory > 100:  # Verifica aumentos significativos no uso de memória
                print(f"\nWarning: Large GPU memory increase: {after_memory - before_memory:.2f}MB")
        
        # Atualiza a rede-alvo conforme necessário
        self.target_update_counter += 1
        if self.target_update_counter >= self.target_update_frequency:
            self.update_target_model()  # Atualiza a rede-alvo
            self.target_update_counter = 0
        
        # Decai o ε conforme o treinamento avança
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        loss_value = loss.item()  # Armazena o valor da perda atual
        self.losses.append(loss_value)
        return loss_value

    def get_learning_metrics(self):
        """Retorna métricas atuais de aprendizado, incluindo informações da GPU"""
        metrics = {
            'avg_loss': np.mean(self.losses[-100:]) if self.losses else 0,  # Média das perdas recentes
            'avg_q_value': np.mean(self.avg_q_values[-100:]) if self.avg_q_values else 0,  # Média dos valores Q recentes
            'action_distribution': {k: v/sum(self.action_frequencies.values()) for k, v in self.action_frequencies.items()},  # Distribuição de ações
            'epsilon': self.epsilon,  # Taxa de exploração atual
            'gpu_memory_used': f"{torch.cuda.memory_allocated() / 1024**2:.2f}MB",  # Memória GPU alocada
            'gpu_memory_cached': f"{torch.cuda.memory_reserved() / 1024**2:.2f}MB"  # Memória GPU reservada
        }
        return metrics

def train(episodes=100, log_interval=1):
    """
    Função principal de treinamento do agente Q-Learning no ambiente 2048.
    
    Args:
        episodes (int): Número total de episódios de treinamento.
        log_interval (int): Intervalo de episódios para exibir o progresso no terminal.
        
    Returns:
        agent: O agente treinado.
        metrics: Objeto de rastreamento de métricas para análise.
    """
    print("\n=== Training Configuration ===")
    print(f"Episodes: {episodes}")  # Exibe o número de episódios configurados
    print(f"GPU: {torch.cuda.get_device_name(0)}")  # Exibe o nome da GPU em uso
    print("============================\n")
    
    # Inicializa o ambiente de jogo 2048
    env = Game2048()
    # Cria o agente Q-Learning com tamanho do estado (16 tiles) e 4 ações possíveis
    agent = QLearningAgent(16, 4)
    # Inicializa o rastreador de métricas
    metrics = MetricsTracker()
    
    # Variáveis para armazenar a melhor pontuação e o maior tile alcançado
    best_score = 0
    best_tile = 0
    
    # Loop principal de treinamento por episódios
    for episode in range(episodes):
        # Reinicializa o ambiente para um novo episódio
        env = Game2048()
        state = env.get_state()  # Obtém o estado inicial
        total_reward = 0  # Acumula a recompensa total no episódio
        steps = 0  # Contador de passos no episódio
        start_time = time.time()  # Marca o tempo de início do episódio
        
        # Loop do episódio até o jogo terminar ou ser vencido
        while not env.is_game_over() and not env.is_won():
            valid_actions = env.get_valid_actions()  # Obtém as ações válidas no estado atual
            action = agent.act(state, valid_actions)  # O agente decide a ação
            
            if action is None:  # Se nenhuma ação for válida, encerra o loop
                break
            
            env.make_move(action)  # Executa a ação no ambiente
            next_state = env.get_state()  # Obtém o novo estado após a ação
            reward = env.get_reward()  # Calcula a recompensa do estado atual
            done = env.is_game_over() or env.is_won()  # Verifica se o jogo terminou
            
            # Armazena a experiência na memória do agente
            agent.remember(state, action, reward, next_state, done)
            # Realiza o aprendizado por replay (atualiza a rede neural)
            loss = agent.replay()
            
            # Atualiza o estado atual e incrementa os contadores
            state = next_state
            total_reward += reward
            steps += 1
            
            # Exibe informações do progresso a cada 10 passos
            if steps % 10 == 0:
                learning_metrics = agent.get_learning_metrics()
                print(f"\rEpisode {episode+1}/{episodes} - Steps: {steps} - Score: {env.score} "
                      f"- Max Tile: {env.get_max_tile()} - Loss: {learning_metrics['avg_loss']:.4f} "
                      f"- Avg Q: {learning_metrics['avg_q_value']:.4f} - ε: {learning_metrics['epsilon']:.3f}", 
                      end="")
        
        # Calcula a duração do episódio e adiciona as métricas do episódio
        episode_time = time.time() - start_time
        metrics.add_episode(env.score, env.get_max_tile(), steps, episode_time)
        
        # Atualiza as melhores pontuações, se necessário
        if env.score > best_score:
            best_score = env.score
        if env.get_max_tile() > best_tile:
            best_tile = env.get_max_tile()
        
        # Exibe o resumo do episódio em intervalos configurados
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
            print(env)  # Exibe o tabuleiro atual do jogo
            print("-" * 40)  # Separador visual
    
    # Plota gráficos com as métricas de aprendizado após o treinamento
    metrics.plot_metrics()
    return agent, metrics  # Retorna o agente treinado e as métricas

if __name__ == "__main__":
    # Ponto de entrada do script
    try:
        # Inicia o treinamento com 1000 episódios e logs a cada 20 episódios
        agent, metrics = train(episodes=1000, log_interval=20)
    except Exception as e:
        # Captura e exibe qualquer erro durante o treinamento
        print(f"\nError: {str(e)}")
        raise