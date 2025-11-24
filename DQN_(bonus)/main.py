import numpy as np
import pygame
import os
from environment import WallE_RL, PygameInit, ACTIONS

if __name__ == "__main__":

    FPS = 2
    env = WallE_RL()
    screen, clock = PygameInit.initialization()

    # --- Project Parameters ---
    # These parameters are part of the problem definition
    MAX_ACTIONS_PER_EPISODE = 200
    NUM_TEST_EPISODES = 5

    episode_rewards = []

    print(f"Starting test run for {NUM_TEST_EPISODES} episodes...")

    # Main game loop (for 5 test episodes)
    for episode in range(NUM_TEST_EPISODES):

        # ------------------------------------------------------------------
        # This section resets the environment for the start of each new episode
        state, agent_power = env.reset()
        # ------------------------------------------------------------------

        running = True
        total_reward = 0

        while running:

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    exit()

            # Render the environment
            env.render(screen)

            # -------------------------------- #
            #     Implement your logic here    #
            # -------------------------------- #
            #
            # Currently, this is a "Random Policy".
            # You must replace this section with your Q-Learning logic.
            # Your agent needs to select an 'action' based on the current 'state'.
            action = np.random.choice(ACTIONS)

            # -------------------------------- #
            #     Execute the action           #
            # -------------------------------- #
            #
            # The env.step() function returns the information needed for learning.
            # In this random template, we only need 3 of the 5 returned values.
            #
            next_state, reward, next_trash_state, done, next_agent_power = env.step(action)

            # Update the current state and total reward
            state = next_state
            total_reward += reward

            # Check if the episode is over
            if done:
                print(f"Episode {episode + 1} finished with reward: {total_reward:.0f}")
                episode_rewards.append(total_reward)
                running = False
                break

            pygame.display.flip()
            clock.tick(FPS)

    print("-" * 30)
    print(f"MEAN REWARD (Random Policy): {np.mean(episode_rewards)}")
    pygame.quit()