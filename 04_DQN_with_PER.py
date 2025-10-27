#!/usr/bin/env python3
"""
Experimento con Prioritized Experience Replay (PER)
Implementación del paper "Prioritized Experience Replay" (Schaul et al., 2016)
VizDoom Implementation - Reinforcement Learning
Universidad Nacional del Altiplano - Puno
Autor: Edson Denis Zanabria Ticona.
Doctorado en ciencias de la computación
Inteligencia artificial
"""

import vizdoom as vzd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import time
import os

# ============================================
# Configuración del dispositivo
# ============================================
device = torch.device("cpu")
print(f"Usando dispositivo: {device}")

# ============================================
# Red Neuronal Q-Network (DQN)
# ============================================
class DQN(nn.Module):
    """Red neuronal convolucional para aproximar la función Q"""
    def __init__(self, input_shape, n_actions):
        super(DQN, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        conv_out_size = self._get_conv_out(input_shape)
        
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )
    
    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))
    
    def forward(self, x):
        conv_out = self.conv(x).view(x.size()[0], -1)
        return self.fc(conv_out)

# ============================================
# SumTree para Prioritized Experience Replay
# ============================================
class SumTree:
    """
    Estructura de datos SumTree para muestreo eficiente basado en prioridades.
    Árbol binario donde cada nodo padre contiene la suma de sus hijos.
    """
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.write = 0
        self.n_entries = 0
    
    def _propagate(self, idx, change):
        """Propagar cambio de prioridad hacia arriba del árbol"""
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)
    
    def _retrieve(self, idx, s):
        """Recuperar índice de hoja basado en valor acumulado"""
        left = 2 * idx + 1
        right = left + 1
        
        if left >= len(self.tree):
            return idx
        
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])
    
    def total(self):
        """Retornar prioridad total (raíz del árbol)"""
        return self.tree[0]
    
    def add(self, priority, data):
        """Añadir nueva experiencia con prioridad"""
        idx = self.write + self.capacity - 1
        
        self.data[self.write] = data
        self.update(idx, priority)
        
        self.write += 1
        if self.write >= self.capacity:
            self.write = 0
        
        if self.n_entries < self.capacity:
            self.n_entries += 1
    
    def update(self, idx, priority):
        """Actualizar prioridad de una experiencia"""
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)
    
    def get(self, s):
        """Obtener experiencia basada en valor acumulado"""
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1
        return (idx, self.tree[idx], self.data[dataIdx])

# ============================================
# Prioritized Replay Buffer
# ============================================
class PrioritizedReplayBuffer:
    """
    Buffer de replay con priorización según TD-error.
    Implementa el algoritmo del paper "Prioritized Experience Replay" (Schaul et al., 2016)
    """
    def __init__(self, capacity, alpha=0.6, beta=0.4, beta_increment=0.001):
        """
        Args:
            capacity: Tamaño máximo del buffer
            alpha: Controla cuánta priorización usar (0 = uniforme, 1 = total)
            beta: Controla corrección de importance sampling (0 = sin corrección, 1 = total)
            beta_increment: Incremento de beta por muestreo
        """
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.alpha = alpha  # [0, 1] controla priorización
        self.beta = beta    # [0, 1] controla importance sampling
        self.beta_increment = beta_increment
        self.epsilon = 0.01  # Pequeña constante para evitar prioridad cero
        self.max_priority = 1.0
    
    def push(self, state, action, reward, next_state, done):
        """Añadir experiencia con prioridad máxima inicial"""
        experience = (state, action, reward, next_state, done)
        priority = self.max_priority ** self.alpha
        self.tree.add(priority, experience)
    
    def sample(self, batch_size):
        """
        Muestrear batch con priorización y calcular importance sampling weights
        
        Returns:
            batch: tupla (states, actions, rewards, next_states, dones)
            indices: índices en el SumTree
            weights: pesos de importance sampling
        """
        batch = []
        indices = []
        priorities = []
        segment = self.tree.total() / batch_size
        
        # Incrementar beta
        self.beta = np.min([1.0, self.beta + self.beta_increment])
        
        # Muestrear experiencias proporcionalmente a su prioridad
        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            idx, priority, data = self.tree.get(s)
            
            batch.append(data)
            indices.append(idx)
            priorities.append(priority)
        
        # Calcular importance sampling weights
        sampling_probabilities = np.array(priorities) / self.tree.total()
        weights = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
        weights /= weights.max()  # Normalizar
        
        # Desempaquetar batch
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (states, actions, rewards, next_states, dones), indices, weights
    
    def update_priorities(self, indices, td_errors):
        """Actualizar prioridades basado en TD-errors"""
        for idx, td_error in zip(indices, td_errors):
            priority = (abs(td_error) + self.epsilon) ** self.alpha
            self.tree.update(idx, priority)
            self.max_priority = max(self.max_priority, priority)
    
    def __len__(self):
        return self.tree.n_entries

# ============================================
# Agente DQN con Prioritized Experience Replay
# ============================================
class DQNAgentPER:
    """Agente que implementa Deep Q-Learning con PER"""
    def __init__(self, input_shape, n_actions, learning_rate=0.00025, 
                 gamma=0.99, epsilon_start=1.0, epsilon_end=0.1, 
                 epsilon_decay=10000, buffer_size=10000,
                 alpha=0.6, beta=0.4):
        
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.steps = 0
        
        # Red Q principal y red Q objetivo
        self.policy_net = DQN(input_shape, n_actions).to(device)
        self.target_net = DQN(input_shape, n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # Optimizador y buffer de replay PRIORIZADO
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.replay_buffer = PrioritizedReplayBuffer(buffer_size, alpha=alpha, beta=beta)
        
        print(f"\n=== Configuración del Agente con PER ===")
        print(f"Acciones disponibles: {n_actions}")
        print(f"Learning Rate: {learning_rate}")
        print(f"Gamma: {gamma}")
        print(f"Alpha (priorización): {alpha}")
        print(f"Beta inicial (IS): {beta}")
        print(f"Buffer size: {buffer_size}")
    
    def select_action(self, state, training=True):
        """Seleccionar acción usando política epsilon-greedy"""
        if training:
            self.epsilon = self.epsilon_end + (1.0 - self.epsilon_end) * \
                          np.exp(-1. * self.steps / self.epsilon_decay)
            self.steps += 1
        
        if training and random.random() < self.epsilon:
            return random.randrange(self.n_actions)
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                q_values = self.policy_net(state_tensor)
                return q_values.max(1)[1].item()
    
    def update(self, batch_size):
        """Actualizar la red Q usando mini-batch PRIORIZADO"""
        if len(self.replay_buffer) < batch_size:
            return None
        
        # Muestrear batch priorizado
        (states, actions, rewards, next_states, dones), indices, weights = \
            self.replay_buffer.sample(batch_size)
        
        # Convertir a tensores
        states = torch.FloatTensor(np.array(states)).to(device)
        actions = torch.LongTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).to(device)
        next_states = torch.FloatTensor(np.array(next_states)).to(device)
        dones = torch.FloatTensor(dones).to(device)
        weights = torch.FloatTensor(weights).to(device)
        
        # Q valores actuales
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze()
        
        # Q valores objetivo (usando target network)
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Calcular TD-errors
        td_errors = target_q_values - current_q_values
        
        # Actualizar prioridades en el buffer
        self.replay_buffer.update_priorities(indices, td_errors.detach().cpu().numpy())
        
        # Calcular pérdida ponderada por importance sampling
        loss = (weights * td_errors.pow(2)).mean()
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def update_target_network(self):
        """Copiar pesos de policy network a target network"""
        self.target_net.load_state_dict(self.policy_net.state_dict())

# ============================================
# Configuración del entorno
# ============================================
def create_game(scenario="basic"):
    """Crear y configurar el juego"""
    game = vzd.DoomGame()
    
    if scenario == "basic":
        game.load_config("scenarios/basic.cfg")
    else:
        game.load_config("scenarios/health_gathering.cfg")
    
    game.set_screen_resolution(vzd.ScreenResolution.RES_160X120)
    game.set_screen_format(vzd.ScreenFormat.GRAY8)
    game.set_window_visible(False)
    game.set_sound_enabled(False)
    game.set_mode(vzd.Mode.PLAYER)
    
    return game

def preprocess_frame(frame):
    """Preprocesar frame"""
    frame = frame.astype(np.float32) / 255.0
    frame = np.expand_dims(frame, axis=0)
    return frame

# ============================================
# Función de entrenamiento con PER
# ============================================
def train_with_per(scenario="basic", episodes=1000, batch_size=32, 
                    target_update=10, save_interval=100):
    """Entrenar agente DQN con Prioritized Experience Replay"""
    
    print("\n" + "="*70)
    print(f"EXPERIMENTO CON PRIORITIZED EXPERIENCE REPLAY - {scenario.upper()}")
    print("="*70)
    
    game = create_game(scenario)
    
    try:
        game.init()
        print("✓ Juego inicializado correctamente")
    except Exception as e:
        print(f"✗ Error: {e}")
        return
    
    n_actions = game.get_available_buttons_size()
    input_shape = (1, 120, 160)
    
    print(f"\n=== Configuración del Entorno ===")
    print(f"Escenario: {scenario}")
    print(f"Acciones disponibles: {n_actions}")
    
    # Crear agente CON PER
    agent = DQNAgentPER(input_shape, n_actions, alpha=0.6, beta=0.4)
    
    episode_rewards = []
    episode_lengths = []
    losses = []
    start_time = time.time()
    
    print(f"\n{'='*70}")
    print("INICIANDO ENTRENAMIENTO CON PER")
    print(f"{'='*70}\n")
    
    for episode in range(episodes):
        game.new_episode()
        state = preprocess_frame(game.get_state().screen_buffer)
        
        total_reward = 0
        steps_in_episode = 0
        episode_loss = []
        
        while not game.is_episode_finished():
            action = agent.select_action(state)
            reward = game.make_action([action == i for i in range(n_actions)])
            
            if not game.is_episode_finished():
                next_state = preprocess_frame(game.get_state().screen_buffer)
                done = False
            else:
                next_state = np.zeros_like(state)
                done = True
            
            agent.replay_buffer.push(state, action, reward, next_state, done)
            
            loss = agent.update(batch_size)
            if loss is not None:
                episode_loss.append(loss)
            
            state = next_state
            total_reward += reward
            steps_in_episode += 1
        
        if episode % target_update == 0:
            agent.update_target_network()
        
        episode_rewards.append(total_reward)
        episode_lengths.append(steps_in_episode)
        if episode_loss:
            losses.append(np.mean(episode_loss))
        
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            avg_loss = np.mean(losses[-10:]) if losses else 0
            elapsed = time.time() - start_time
            beta_current = agent.replay_buffer.beta
            
            print(f"Episodio {episode + 1}/{episodes} | "
                  f"Reward: {total_reward:.1f} | "
                  f"Avg (10): {avg_reward:.1f} | "
                  f"Loss: {avg_loss:.4f} | "
                  f"β: {beta_current:.3f} | "
                  f"ε: {agent.epsilon:.3f} | "
                  f"Tiempo: {elapsed:.1f}s")
        
        if (episode + 1) % save_interval == 0:
            os.makedirs('models', exist_ok=True)
            torch.save({
                'episode': episode,
                'model_state_dict': agent.policy_net.state_dict(),
                'rewards': episode_rewards,
            }, f'models/{scenario}_PER_ep{episode+1}.pth')
            print(f"✓ Modelo PER guardado")
    
    game.close()
    
    print(f"\n{'='*70}")
    print("ENTRENAMIENTO CON PER COMPLETADO")
    print(f"{'='*70}")
    print(f"Tiempo total: {time.time() - start_time:.1f}s")
    print(f"Reward promedio (últimos 100): {np.mean(episode_rewards[-100:]):.2f}")
    
    os.makedirs('results', exist_ok=True)
    np.savez(f'results/{scenario}_PER_metrics.npz',
             rewards=episode_rewards,
             lengths=episode_lengths,
             losses=losses)
    
    return agent, episode_rewards, losses

# ============================================
# Función principal
# ============================================
if __name__ == "__main__":
    print("\n" + "="*70)
    print("Prioritized Experience Replay (PER) - Schaul et al., 2016")
    print("="*70)
    
    # Entrenar en ambos escenarios
    print("\n[1/2] Entrenando en escenario BASIC...")
    train_with_per(scenario="basic", episodes=1000)
    
    print("\n[2/2] Entrenando en escenario HEALTH GATHERING...")
    train_with_per(scenario="health_gathering", episodes=1000)
    
    print("\n✓ Todos los experimentos con PER completados")
