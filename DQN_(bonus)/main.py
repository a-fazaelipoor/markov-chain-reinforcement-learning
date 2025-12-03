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


if __name__ == "__main__":

    FPS = 2
    env = WallE_RL()
    screen, clock = PygameInit.initialization()

    # --- Project Parameters ---
    NUM_TEST_EPISODES = 5 # Used for the initial random test run

    episode_rewards = []

    print(f"Starting test run for {NUM_TEST_EPISODES} episodes...")

    # Main game loop (for 5 test episodes)
    for episode in range(NUM_TEST_EPISODES):

        state, agent_power = env.reset()
        
        # New: Get the full trash state for a complete observation 
        trash_state = env._WallE_RL__get_trash_state() 

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

            # Update the current state and total reward
            state = next_state_pos
            agent_power = next_agent_power
            trash_state = next_trash_state
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