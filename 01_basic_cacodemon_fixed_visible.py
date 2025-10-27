#!/usr/bin/env python3
"""
Experimento 1: Basic - Agente vs Cacodemon
VizDoom Implementation - Reinforcement Learning
Universidad Nacional del Altiplano - Puno
Autor: Edson Denis Zanabria Ticona.
VERSIÓN CORREGIDA - Encuentra automáticamente los escenarios
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
from datetime import datetime

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
        
        # Capas convolucionales para procesar imágenes
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        # Calcular tamaño después de convoluciones
        conv_out_size = self._get_conv_out(input_shape)
        
        # Capas fully connected
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )
    
    def _get_conv_out(self, shape):
        """Calcular dimensión de salida de las capas convolucionales"""
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))
    
    def forward(self, x):
        conv_out = self.conv(x).view(x.size()[0], -1)
        return self.fc(conv_out)

# ============================================
# Replay Buffer (Experience Replay)
# ============================================
class ReplayBuffer:
    """Buffer para almacenar experiencias de entrenamiento"""
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return state, action, reward, next_state, done
    
    def __len__(self):
        return len(self.buffer)

# ============================================
# Agente DQN
# ============================================
class DQNAgent:
    """Agente que implementa Deep Q-Learning"""
    def __init__(self, input_shape, n_actions, learning_rate=0.00025, 
                 gamma=0.99, epsilon_start=1.0, epsilon_end=0.1, 
                 epsilon_decay=10000, buffer_size=10000):
        
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
        
        # Optimizador y buffer de replay
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.replay_buffer = ReplayBuffer(buffer_size)
        
        print(f"\n=== Configuración del Agente ===")
        print(f"Acciones disponibles: {n_actions}")
        print(f"Learning Rate: {learning_rate}")
        print(f"Gamma (descuento): {gamma}")
        print(f"Epsilon inicial: {epsilon_start}")
        print(f"Epsilon final: {epsilon_end}")
        print(f"Buffer size: {buffer_size}")
    
    def select_action(self, state, training=True):
        """Seleccionar acción usando política epsilon-greedy"""
        if training:
            # Decaimiento de epsilon
            self.epsilon = self.epsilon_end + (1.0 - self.epsilon_end) * \
                          np.exp(-1. * self.steps / self.epsilon_decay)
            self.steps += 1
        
        if training and random.random() < self.epsilon:
            # Exploración: acción aleatoria
            return random.randrange(self.n_actions)
        else:
            # Explotación: mejor acción según Q
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                q_values = self.policy_net(state_tensor)
                return q_values.max(1)[1].item()
    
    def update(self, batch_size):
        """Actualizar la red Q usando mini-batch"""
        if len(self.replay_buffer) < batch_size:
            return None
        
        # Muestrear batch del buffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        
        # Convertir a tensores
        states = torch.FloatTensor(np.array(states)).to(device)
        actions = torch.LongTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).to(device)
        next_states = torch.FloatTensor(np.array(next_states)).to(device)
        dones = torch.FloatTensor(dones).to(device)
        
        # Q valores actuales
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))
        
        # Q valores objetivo (usando target network)
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Calcular pérdida y actualizar
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def update_target_network(self):
        """Copiar pesos de policy network a target network"""
        self.target_net.load_state_dict(self.policy_net.state_dict())

# ============================================
# Configuración del entorno VizDoom
# ============================================
def create_basic_game():
    """Crear y configurar el juego básico con Cacodemon"""
    game = vzd.DoomGame()
    
    # SOLUCIÓN: Usar la ruta de escenarios de VizDoom
    try:
        # Intentar cargar desde la instalación de VizDoom
        scenario_path = os.path.join(vzd.scenarios_path, "basic.cfg")
        print(f"Intentando cargar escenario desde: {scenario_path}")
        game.load_config(scenario_path)
        print("✓ Escenario cargado desde instalación de VizDoom")
    except:
        print(" No se pudo cargar .cfg, configurando manualmente...")
        # Configuración manual si falla
        game.set_doom_scenario_path(os.path.join(vzd.scenarios_path, "basic.wad"))
        game.set_doom_map("map01")
        
        # Configuración de la ventana
        game.set_screen_resolution(vzd.ScreenResolution.RES_160X120)
        game.set_screen_format(vzd.ScreenFormat.GRAY8)
        game.set_render_hud(False)
        game.set_render_crosshair(False)
        game.set_render_weapon(True)
        game.set_render_decals(False)
        game.set_render_particles(False)
        
        # Configuración de los botones disponibles
        game.set_available_buttons([
            vzd.Button.MOVE_LEFT,
            vzd.Button.MOVE_RIGHT,
            vzd.Button.ATTACK
        ])
        
        # Variables de juego
        game.set_available_game_variables([vzd.GameVariable.HEALTH])
        
        # Configuración del modo y timing
        game.set_episode_timeout(300)
        game.set_episode_start_time(10)
        game.set_window_visible(False)
        game.set_sound_enabled(False)
        game.set_living_reward(-1)
        game.set_mode(vzd.Mode.PLAYER)
        
        print("✓ Configuración manual aplicada")
    
    # CRÍTICO: Forzar configuración correcta después de cargar
    # Esto sobrescribe cualquier configuración del .cfg
    game.set_screen_resolution(vzd.ScreenResolution.RES_160X120)
    game.set_screen_format(vzd.ScreenFormat.GRAY8)
    game.set_window_visible(False)
    print(f"✓ Formato forzado: 160x120 GRAY8")
    
    return game

# ============================================
# Preprocesamiento de imágenes
# ============================================
def preprocess_frame(frame):
    """Preprocesar frame para la red neuronal"""
    # Normalizar a [0, 1]
    frame = frame.astype(np.float32) / 255.0
    # Añadir dimensión de canal
    frame = np.expand_dims(frame, axis=0)
    return frame

# ============================================
# Función de entrenamiento
# ============================================
def train(episodes=1000, batch_size=32, target_update=10, save_interval=100):
    """Entrenar el agente DQN"""
    
    print("\n" + "="*60)
    print("EXPERIMENTO 1: BASIC - CACODEMON")
    print("="*60)
    
    # Crear el juego
    game = create_basic_game()
    
    try:
        game.init()
        print("✓ Juego inicializado correctamente")
    except Exception as e:
        print(f"✗ Error al inicializar: {e}")
        print("\nVerificando instalación de VizDoom...")
        print(f"Ruta de escenarios: {vzd.scenarios_path}")
        return None, [], [], []
    
    # Obtener información del juego
    n_actions = game.get_available_buttons_size()
    input_shape = (1, 120, 160)  # Grayscale, 120x160
    
    print(f"\n=== Configuración del Entorno ===")
    print(f"Acciones disponibles: {n_actions}")
    print(f"Forma de entrada: {input_shape}")
    print(f"Botones: {[str(b) for b in game.get_available_buttons()]}")
    
    # Crear agente
    agent = DQNAgent(input_shape, n_actions)
    
    # Métricas de entrenamiento
    episode_rewards = []
    episode_lengths = []
    losses = []
    start_time = time.time()
    
    print(f"\n{'='*60}")
    print("INICIANDO ENTRENAMIENTO")
    print(f"{'='*60}\n")
    
    for episode in range(episodes):
        game.new_episode()
        state = preprocess_frame(game.get_state().screen_buffer)
        
        total_reward = 0
        steps_in_episode = 0
        episode_loss = []
        
        while not game.is_episode_finished():
            # Seleccionar y ejecutar acción
            action = agent.select_action(state)
            reward = game.make_action([action == i for i in range(n_actions)])
            
            # Obtener siguiente estado
            if not game.is_episode_finished():
                next_state = preprocess_frame(game.get_state().screen_buffer)
                done = False
            else:
                next_state = np.zeros_like(state)
                done = True
            
            # Guardar experiencia
            agent.replay_buffer.push(state, action, reward, next_state, done)
            
            # Actualizar red
            loss = agent.update(batch_size)
            if loss is not None:
                episode_loss.append(loss)
            
            state = next_state
            total_reward += reward
            steps_in_episode += 1
        
        # Actualizar target network periódicamente
        if episode % target_update == 0:
            agent.update_target_network()
        
        # Registrar métricas
        episode_rewards.append(total_reward)
        episode_lengths.append(steps_in_episode)
        if episode_loss:
            losses.append(np.mean(episode_loss))
        
        # Imprimir progreso
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            avg_length = np.mean(episode_lengths[-10:])
            avg_loss = np.mean(losses[-10:]) if losses else 0
            elapsed = time.time() - start_time
            
            print(f"Episodio {episode + 1}/{episodes} | "
                  f"Reward: {total_reward:.1f} | "
                  f"Avg Reward (10): {avg_reward:.1f} | "
                  f"Steps: {steps_in_episode} | "
                  f"Epsilon: {agent.epsilon:.3f} | "
                  f"Loss: {avg_loss:.4f} | "
                  f"Tiempo: {elapsed:.1f}s")
        
        # Guardar modelo periódicamente
        if (episode + 1) % save_interval == 0:
            os.makedirs('models', exist_ok=True)
            torch.save({
                'episode': episode,
                'model_state_dict': agent.policy_net.state_dict(),
                'optimizer_state_dict': agent.optimizer.state_dict(),
                'rewards': episode_rewards,
            }, f'models/basic_cacodemon_ep{episode+1}.pth')
            print(f"✓ Modelo guardado en episodio {episode + 1}")
    
    game.close()
    
    # Guardar resultados finales
    print(f"\n{'='*60}")
    print("ENTRENAMIENTO COMPLETADO")
    print(f"{'='*60}")
    print(f"Tiempo total: {time.time() - start_time:.1f} segundos")
    print(f"Reward promedio final (últimos 100 ep): {np.mean(episode_rewards[-100:]):.2f}")
    print(f"Mejor reward: {max(episode_rewards):.2f}")
    
    # Guardar métricas
    os.makedirs('results', exist_ok=True)
    np.savez('results/basic_cacodemon_metrics.npz',
             rewards=episode_rewards,
             lengths=episode_lengths,
             losses=losses)
    
    return agent, episode_rewards, episode_lengths, losses

# ============================================
# Función principal
# ============================================
if __name__ == "__main__":
    print("\n" + "="*60)
    print("VizDoom - Experimento Basic (Cacodemon)")
    print("Deep Q-Learning Implementation")
    print("="*60 + "\n")
    
    # Mostrar información de VizDoom
    print(f"VizDoom version: {vzd.__version__}")
    print(f"Ruta de escenarios: {vzd.scenarios_path}")
    print(f"Archivos disponibles: {os.listdir(vzd.scenarios_path)[:5]}...")
    
    # Entrenar el agente
    result = train(
        episodes=1000,
        batch_size=32,
        target_update=10,
        save_interval=100
    )
    
    if result[0] is not None:
        print("\n✓ Experimento completado exitosamente")
        print("  - Modelos guardados en: ./models/")
        print("  - Métricas guardadas en: ./results/")
    else:
        print("\n✗ El experimento no pudo completarse")
        print("  Revise los mensajes de error anteriores")
