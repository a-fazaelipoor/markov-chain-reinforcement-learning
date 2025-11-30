import numpy as np
import random
# Imports for other sections are omitted for this logical split
# import pygame
# import matplotlib.pyplot as plt
# from environment import WallE_RL, ACTIONS

# --- Hyperparameters ---
TOTAL_EPISODES = 10_000
MAX_STEPS = 200
ALPHA = 0.1
GAMMA = 0.99

EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY_LINEAR = True

GRID_FALLBACK = 8
N_POWER_BINS = 4
N_TRASH = 6

# ------------------------------------------------------

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

# ------------------ main training (Initial Structure) ------------------
def simple_train(env, episodes=TOTAL_EPISODES, max_steps=MAX_STEPS,
                 alpha=ALPHA, gamma=GAMMA,
                 eps_start=EPSILON_START, eps_end=EPSILON_END,
                 linear_decay=EPSILON_DECAY_LINEAR, VERBOSE=True):

    rng = random.Random()
    grid_w, n_trash = GRID_FALLBACK , N_TRASH
    n_states = ((grid_w * grid_w) * N_POWER_BINS) * (2 ** n_trash)
    n_actions = 4
    Q = np.zeros((n_states, n_actions), dtype=np.float32)

    rewards = []
    q_diffs = []

    prev_q = None
    for ep in range(1, episodes + 1):
        pos, power = env.reset()
        trash = get_trash_list()
        s = state_to_id(pos, power, trash, grid_w, n_trash)
        total = 0.0
        
        if linear_decay:
            eps = max(eps_end, eps_start - (ep - 1) * (eps_start - eps_end) / max(1, episodes - 1))
        else:
            decay = (eps_end / eps_start) ** (1 / max(1, episodes))
            eps = max(eps_end, eps_start * (decay ** ep))

        for step in range(max_steps):
            
            if rng.random() < eps:
                a = rng.choice([0, 1, 2, 3])
            else:
                a = int(np.argmax(Q[s]))

            if step > max_steps: 
                break 


    return Q, rewards, (grid_w, n_trash) , q_diffs