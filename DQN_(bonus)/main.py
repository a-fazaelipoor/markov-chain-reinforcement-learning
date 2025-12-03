import numpy as np
import pygame
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import matplotlib.pyplot as plt
from environment import WallE_RL, PygameInit, ACTIONS

# =============================================================================
# HYPERPARAMETERS
# =============================================================================
GAMMA = 0.99
ALPHA = 0.001
EPSILON_START = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.996
BATCH_SIZE = 64
REPLAY_MEMORY_SIZE = 50000
TARGET_UPDATE = 500
TRAIN_EPISODES = 2000
TEST_EPISODES = 5

STATE_DIM = 6
ACTION_DIM = 4

# DQN CLASS
class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        # Define the Fully Connected layers sequence becuase environment is small 5*5
        self.fc = nn.Sequential(
            # Input Layer (STATE_DIM=6) to Hidden Layer 1 (64 neurons)
            nn.Linear(STATE_DIM, 64), 
            nn.ReLU(), 
            
            # Hidden Layer 2 (64 neurons)
            nn.Linear(64, 64),
            nn.ReLU(),
            
            # Output Layer (ACTION_DIM=4 -> Q-values for each action)
            nn.Linear(64, ACTION_DIM)
        )

# HELPER FUNCTIONS
def state_to_vector(pos, power, trash_states):
    """
    Converts environment variables (position, power, trash status) into the 
    normalized state vector (STATE_DIM=6) for the DQN.
    """
    x, y = pos
    
    # Normalize position for faster convergence
    x_norm = x / 4.0 
    y_norm = y / 4.0

    # Normalize battery power
    power_norm = min(power, 30) / 30.0
    
    # Trash states (to Boolean states)
    trash_vec = np.array(trash_states, dtype=float)

    return np.array([x_norm, y_norm, power_norm, trash_vec[0], trash_vec[1], trash_vec[2]], dtype=np.float32)


if __name__ == "__main__":

    FPS = 2
    env = WallE_RL()
    screen, clock = PygameInit.initialization()

    NUM_TEST_EPISODES = 5

    episode_rewards = []

    print(f"Starting test run for {NUM_TEST_EPISODES} episodes...")

    # Main game loop
    for episode in range(NUM_TEST_EPISODES):

        state_pos, agent_power = env.reset()
        trash_state = env._WallE_RL__get_trash_state() 
        
        # New: Convert initial state to vector for eventual DQN use
        state_vec = state_to_vector(state_pos, agent_power, trash_state)

        running = True
        total_reward = 0

        while running:

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    exit()

            env.render(screen)

            # Random Policy (to be replaced by DQN logic)
            action = np.random.choice(ACTIONS)

            next_state_pos, reward, next_trash_state, done, next_agent_power = env.step(action)
            
            # New: Convert next state to vector
            next_state_vec = state_to_vector(next_state_pos, next_agent_power, next_trash_state)

            # Update the current state and total reward
            state_vec = next_state_vec
            total_reward += reward

            if done:
                print(f"Episode {episode + 1} finished with reward: {total_reward:.0f}")
                episode_rewards.append(total_reward)
                running = False
                break

            pygame.display.flip()
            clock.tick(FPS)

    print("-" * 30)
    print(f"MEAN REWARD (Random Policy): {np.mean(episode_rewards):.1f}")
    pygame.quit()