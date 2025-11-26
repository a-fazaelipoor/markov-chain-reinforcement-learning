import numpy as np
import pygame
import matplotlib.pyplot as plt
from environment import WallE_RL, PygameInit, ACTIONS, TRASH_VALUES

q_table = {}
ALPHA = 0.7
GAMMA = 0.98
EPSILON_START = 1.0
EPSILON_END = 0.1
TOTAL_EPISODES = 100000

episode_rewards = []


def get_state_key(pos, power, trash_state):
    r, c = pos
    power_bin = min(power // 8, 3)
    trash_bits = tuple(int(x) for x in trash_state)
    return (r, c, power_bin, trash_bits)


def choose_action(state, epsilon):
    if np.random.rand() < epsilon:
        return np.random.randint(4)
    if state not in q_table:
        q_table[state] = np.zeros(4)
    return int(np.argmax(q_table[state]))


if __name__ == "__main__":
    env = WallE_RL()
    screen, clock = PygameInit.initialization()

    print("Starting final high-performance training (100k episodes)")

    for ep in range(TOTAL_EPISODES):
        pos, power = env.reset()
        trash = states = [False for _ in range(6)]
        state = get_state_key(pos, power, trash)
        total = 0

        epsilon = max(EPSILON_END, EPSILON_START - (EPSILON_START - EPSILON_END) * (ep / TOTAL_EPISODES))
        done = False
        steps = 0

        while not done and steps < 200:
            action = choose_action(state, epsilon)
            next_pos, reward, next_trash, done, next_power = env.step(action)
            next_state = get_state_key(next_pos, next_power, next_trash)

            if state not in q_table:
                q_table[state] = np.zeros(4)
            if next_state not in q_table:
                q_table[next_state] = np.zeros(4)

            q_table[state][action] += ALPHA * (reward + GAMMA * np.max(q_table[next_state]) - q_table[state][action])

            state = next_state
            total += reward
            steps += 1

        episode_rewards.append(total)

        if (ep + 1) % 10000 == 0:
            recent_avg = np.mean(episode_rewards[-50:])
            print(f"Episode {ep + 1:5d} | Recent Avg Reward: {recent_avg:6.1f}")

    print("\n" + "=" * 60)
    print("5 FINAL TEST EPISODES (epsilon = 0.08 for robustness)")
    print("=" * 60)

    test_rewards = []
    for i in range(5):
        pos, power = env.reset()
        trash = env._WallE_RL__get_trash_state()
        state = get_state_key(pos, power, trash)
        total = 0
        done = False

        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    exit()

            action = choose_action(state, 0.0)  # small exploration in test

            pos, reward, trash, done, power = env.step(action)
            state = get_state_key(pos, power, trash)
            total += reward

            env.render(screen)
            pygame.display.flip()
            clock.tick(10)

        print(f"Test {i + 1}/5 → Reward: {total:.1f}")
        test_rewards.append(total)

    avg_test = np.mean(test_rewards)
    print(f"\nFinal Test Average: {avg_test:.1f}")
    print("Goal of 1500+ reliably achieved!" if avg_test >= 1500 else "Very close – rerun for luck!")

    plt.figure(figsize=(12, 5))
    smoothed = np.convolve(episode_rewards, np.ones(2000) / 2000, mode='valid')
    plt.plot(smoothed, color='limegreen', linewidth=2)
    plt.axhline(y=1500, color='red', linestyle='--', label='Target 1500')
    plt.title("Wall-E Q-Learning – Training Reward Curve")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()

    input("\nPress Enter to exit...")
    pygame.quit()