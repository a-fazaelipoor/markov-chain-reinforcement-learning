import numpy as np
import pygame
import random
import matplotlib.pyplot as plt
from environment import WallE_RL, PygameInit, ACTIONS

TOTAL_EPISODES = 300_000
MAX_STEPS = 200
ALPHA = 0.1
GAMMA = 0.99

EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY_LINEAR = True # اگر False => exponential

GRID_FALLBACK = 8
N_POWER_BINS = 4
N_TRASH = 6

SAVE_Q_PATH = "Q_simple.npy"
PLOT_PATH = "reward_simple.png"

SEED = 1111123
FPS_TEST = 6
VERBOSE = True
# ------------------------------------------------------

random.seed(SEED)
np.random.seed(SEED)

# ------------------ utility (state id) ------------------


def get_trash_list():
    return [False] * N_TRASH

def trash_to_mask(trash_list):
    m = 0
    for i, b in enumerate(trash_list):
        if b:
            m |= (1 << i)
    return m

def state_to_id(pos, power, trash_list, grid_w, n_trash, n_power_bins=N_POWER_BINS):
    r, c = pos
    pos_idx = r * grid_w + c
    power_idx = min(power // 8, n_power_bins - 1)
    mask = trash_to_mask(trash_list)
    return ((pos_idx * n_power_bins) + power_idx) * (2 ** n_trash) + mask

# ------------------ main simple training ------------------
def simple_train(env, episodes=TOTAL_EPISODES, max_steps=MAX_STEPS,
                 alpha=ALPHA, gamma=GAMMA,
                 eps_start=EPSILON_START, eps_end=EPSILON_END,
                 linear_decay=EPSILON_DECAY_LINEAR):

    rng = random.Random()
    grid_w, n_trash = GRID_FALLBACK , N_TRASH
    n_states = ((grid_w * grid_w) * N_POWER_BINS) * (2 ** n_trash)
    n_actions = len(ACTIONS)
    Q = np.zeros((n_states, n_actions), dtype=np.float32)

    rewards = []

    for ep in range(1, episodes + 1):
        pos, power = env.reset()
        trash = get_trash_list()
        s = state_to_id(pos, power, trash, grid_w, n_trash)
        total = 0.0
        # compute epsilon
        if linear_decay:
            eps = max(eps_end, eps_start - (ep - 1) * (eps_start - eps_end) / max(1, episodes - 1))
        else:
            # exponential
            decay = (eps_end / eps_start) ** (1 / max(1, episodes))
            eps = max(eps_end, eps_start * (decay ** ep))

        for step in range(max_steps):
            # epsilon-greedy
            if rng.random() < eps:
                a = rng.choice(ACTIONS)
            else:
                a = int(np.argmax(Q[s]))

            next_pos, reward, next_trash, done, next_power = env.step(a)
            s2 = state_to_id(next_pos, next_power, next_trash, grid_w, n_trash)

            # Q update (classic)
            next_max = 0.0 if done else np.max(Q[s2])
            Q[s, a] += alpha * (reward + gamma * next_max - Q[s, a])

            s = s2
            total += reward

            if done:
                break

        rewards.append(total)

        if VERBOSE and ep % max(1, episodes // 10) == 0:
            recent = np.mean(rewards[-100:]) if len(rewards) >= 1 else 0.0
            print(f"[Ep {ep}/{episodes}] eps={eps:.3f} total={total:.1f} recent100={recent:.1f}")

    # save Q
    try:
        np.save(SAVE_Q_PATH, Q)
        if VERBOSE: print("Saved Q to", SAVE_Q_PATH)
    except Exception as e:
        print("Save Q failed:", e)

    return Q, rewards, (grid_w, n_trash)

# ------------------ simple visual test ------------------
def simple_test(env, Q, grid_w, n_trash, episodes=5, fps=FPS_TEST):
    screen, clock = PygameInit.initialization()
    results = []
    for ep in range(episodes):
        pos, power = env.reset()
        trash = get_trash_list()
        s = state_to_id(pos, power, trash, grid_w, n_trash)
        total = 0.0
        done = False
        steps = 0
        power_before = 5

        while not done:
            a = int(np.argmax(Q[s]))  # greedy
            pos, reward, trash, done, power = env.step(a)
            s = state_to_id(pos, power, trash, grid_w, n_trash)

            if power_before != power:
                print(power)
                power_before = power

            total += reward
            steps += 1
            env.render(screen)
            pygame.display.flip()
            clock.tick(fps)
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    pygame.quit()
                    return results
        print(f"[TEST] ep{ep+1}:reward={total:.1f} steps={steps} done={done}")
        results.append((total, steps, done))
    pygame.quit()
    return results

# ------------------ plotting ------------------
def plot_rewards(rewards, window=200):
    if len(rewards) < window:
        plt.plot(rewards)
    else:
        sm = np.convolve(rewards, np.ones(window)/window, mode='valid')
        plt.plot(sm)
    plt.xlabel("Episode")
    plt.ylabel("Reward (smoothed)")
    plt.grid(alpha=0.3)
    plt.title("Training curve (simple)")
    plt.savefig(PLOT_PATH)
    plt.show()

# ------------------ run ------------------
if __name__ == "__main__":
    env = WallE_RL()
    Q, rewards, (grid_w, n_trash) = simple_train(env, episodes=TOTAL_EPISODES)
    plot_rewards(rewards)
    print("Starting visual tests...")
    simple_test(env, Q, grid_w, n_trash, episodes=5)
