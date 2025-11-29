import numpy as np
import pygame
import random
import copy
import pprint

from mpmath.libmp import to_str

from environment import WallE, PygameInit



def build_trash_reward_maps(env):
    grid_size = env._WallE__grid_size
    trash_positions = env.get_trash_points()
    trash_reward_maps = {}

    for (tr, tc) in trash_positions:
        r_map = [[-1 for _ in range(grid_size)] for _ in range(grid_size)]

        for r in range(grid_size):
            for c in range(grid_size):
                tile, val = env.grid[r][c]

                if tile == 'B':
                    r_map[r][c] = -100
                elif tile == 'E':
                    r_map[r][c] = -100000
                elif tile == 'F':
                    r_map[r][c] = 0
                elif tile == 'S':
                    if (r, c) == (tr, tc):
                        r_map[r][c] = 5000
                    else:
                        r_map[r][c] = -2
                else:
                    r_map[r][c] = -1

        trash_reward_maps[(tr, tc)] = r_map

    return trash_reward_maps



def get_term(target_r, target_c, current_r, current_c, grid_size, grid, value, v_table, gamma):

    if 0 <= target_r < grid_size and 0 <= target_c < grid_size:
        if grid[target_r][target_c][0] != 'B':
            return value[target_r][target_c] + gamma * v_table[target_r][target_c]
        else :
            return -1 + gamma * v_table[current_r][current_c]

    return -1 + gamma * v_table[current_r][current_c]


def up_down_calc(move, value, move_prob, neigh_prob, grid_size, r, c, v_table, gamma, grid):
    term_intended = 0
    term_neigh1 = 0
    term_neigh2 = 0

    if move == "up":
        term_intended = get_term(r - 1, c, r, c, grid_size, grid, value, v_table, gamma)
        term_neigh1 = get_term(r, c - 1, r, c, grid_size, grid, value, v_table, gamma)
        term_neigh2 = get_term(r, c + 1, r, c, grid_size, grid, value, v_table, gamma)

    elif move == "down":
        term_intended = get_term(r + 1, c, r, c, grid_size, grid, value, v_table, gamma)
        term_neigh1 = get_term(r, c - 1, r, c, grid_size, grid, value, v_table, gamma)
        term_neigh2 = get_term(r, c + 1, r, c, grid_size, grid, value, v_table, gamma)

    val = (move_prob * term_intended) + (neigh_prob * term_neigh1) + (neigh_prob * term_neigh2)
    return val


def left_right_calc(move, value, move_prob, neigh_prob, grid_size, r, c, v_table, gamma, grid):
    term_intended = 0
    term_neigh1 = 0
    term_neigh2 = 0

    if move == "left":
        term_intended = get_term(r, c - 1, r, c, grid_size, grid, value, v_table, gamma)
        term_neigh1 = get_term(r - 1, c, r, c, grid_size, grid, value, v_table, gamma)
        term_neigh2 = get_term(r + 1, c, r, c, grid_size, grid, value, v_table, gamma)

    elif move == "right":
        term_intended = get_term(r, c + 1, r, c, grid_size, grid, value, v_table, gamma)
        term_neigh1 = get_term(r - 1, c, r, c, grid_size, grid, value, v_table, gamma)
        term_neigh2 = get_term(r + 1, c, r, c, grid_size, grid, value, v_table, gamma)

    val = (move_prob * term_intended) + (neigh_prob * term_neigh1) + (neigh_prob * term_neigh2)
    return val


# -------------------------------- #
#     Implement your code here     #
# -------------------------------- #
class MySolver(WallE):
    def actions_model(self, reward_map):
        gamma = 0.99
        policy_dictionary = {}
        transition = self._WallE__probability_dict
        for key, value in reward_map.items():
            grid_size = self._WallE__grid_size
            v_table = [[0 for _ in range(grid_size)] for _ in range(grid_size)]
            policy = [[0 for _ in range(grid_size)] for _ in range(grid_size)]
            for r in range(grid_size):
                for c in range(grid_size):
                    v_table[r][c] = 0

            max_iteration = 5000
            epsilon = 1e-10

            for iteration in range(max_iteration):
                v_new = [[0 for _ in range(grid_size)] for _ in range(grid_size)]
                delta = 0
                for r in range(grid_size):
                    for c in range(grid_size):
                        if self._WallE__base_grid[r][c][0] == 'B':
                            v_new[r][c] = 0
                            policy[r][c] = -1  # اکشنی وجود ندارد
                            continue

                        p =transition[(r,c)]

                        up = p[0]
                        down = p[1]
                        left = p[2]
                        right = p[3]

                        up_val = up_down_calc("up" , value , up["intended"] , up["neighbor"] , grid_size, r, c ,v_table , gamma , self._WallE__base_grid)
                        down_val = up_down_calc("down" , value , down["intended"],  down["neighbor"] , grid_size, r, c , v_table , gamma , self._WallE__base_grid)
                        left_val = left_right_calc("left" , value ,left["intended"], left["neighbor"] , grid_size, r, c , v_table , gamma , self._WallE__base_grid)
                        right_val = left_right_calc("right" , value , right["intended"] , right["neighbor"] , grid_size, r, c , v_table , gamma , self._WallE__base_grid)

                        vals = [up_val, down_val, left_val, right_val]
                        best_action = int(np.argmax(vals))  # 0: up, 1: down, 2: left, 3: right

                        v_new[r][c] = vals[best_action]
                        policy[r][c] = best_action

                        delta = max(delta, abs(v_new[r][c] - v_table[r][c]))

                v_table = v_new
                if delta < epsilon:
                    print(iteration)
                    break

            policy_dictionary[key] = policy

        return policy_dictionary








# -------------------------------- #
#     Implement your code here     #
# -------------------------------- #
# This function is needed to build the 'goal_table'
def build_reward_map(env):
    grid_size = env._WallE__grid_size
    reward_map = [[-1 for _ in range(grid_size)] for _ in range(grid_size)]
    for r in range(env._WallE__grid_size):
        for c in range(env._WallE__grid_size):
            tile, val = env.grid[r][c]
            if tile == 'F':
                reward_map[r][c] = +10000
            elif tile == "E":
                reward_map[r][c] = -10000
            elif tile == "S":
                reward_map[r][c] = -2
            elif tile == "B":
                reward_map[r][c] = -100
            else:
                reward_map[r][c] = -1

    return reward_map









# -------------------------------- #
#     Implement your code here     #
# -------------------------------- #
#
#  Based on the project document, you must now:
#  1. Implement your reward functions.
#  2. Implement your main policy generation logic (pre-computation).
#  3. Implement your policy selection logic.
#  4. Implement the main game loop to run 5 episodes and average the score.



if __name__ == "__main__":

    FPS = 3
    env = MySolver()
    state = env.reset()


    # --- parameters for precomputed table scheme ---
    # don't change these in your final code
    THRESHOLDS = [5, 10, 20, 30, 40]
    MAX_MOVES_PER_EPISODE = 150

    trash_reward_maps = build_trash_reward_maps(env)




    # collect trash positions (fixed ordering) and values from the base grid
    trash_positions = env.get_trash_points()

    # -------------------------------- #
    #     Implement your code here     #
    # -------------------------------- #
    # Build all V-Tables
    # (You need to implement all vtables and call it here)
    # v_tables_map = ...

    print("Building goal_table...")
    goal_reward_map = build_reward_map(env)
    reward_map = {}


    reward_map.update(trash_reward_maps)

    reward_map["goal"] = goal_reward_map

    policy_dictionary = env.actions_model(reward_map)



    print("Finished building goal_table.")
    print(policy_dictionary)
    # ---------------------------#

    screen, clock = PygameInit.initialization()
    sum_of_all = 0

    flower_position = (7, 7)

    # game loop
    for episode in range(5):
        running = True
        current = 0
        moves = 0
        state = env.reset()

        while running:

            base_power = env.agent_value

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            env.render(screen)
            trash_positions = env.get_trash_points()

            useful_trash_exists = False
            agent_current_power = env.agent_value

            for (tr, tc) in trash_positions:
                cell_data = env.grid[tr][tc]
                trash_value = cell_data[1]

                if agent_current_power >= trash_value:
                    useful_trash_exists = True
                    break

            trash_positions = env.get_trash_points()
            agent_power = env.agent_value

            f_r , f_c = flower_position

            dis_to_flower = abs(f_r - state[0]) + abs(f_c - state[1])
            moves_left = MAX_MOVES_PER_EPISODE - moves
            move_limit = 20
            if moves_left <= dis_to_flower + move_limit:
                action_table = policy_dictionary["goal"]
            else:
                best_target = None
                best_dist = 9999

                for (tr, tc) in trash_positions:
                    tile, val = env.grid[tr][tc]

                    if agent_power >= val:
                        dist = abs(tr - state[0]) + abs(tc - state[1])
                        if dist < best_dist:
                            best_dist = dist
                            best_target = (tr, tc)

                if best_target is not None:
                    action_table = policy_dictionary[best_target]

                else:
                    action_table = policy_dictionary["goal"]


            moves += 1
            r, c = state

            action = action_table[r][c]

            next_state, probability, reward_episode, done = env.step(action, THRESHOLDS)
            state = next_state
            current += reward_episode


            if done or moves > MAX_MOVES_PER_EPISODE:
                moves = 0
                print(f"episode {episode} current_score: {current}")
                print(f"agent_value at end: {env.agent_value}")
                sum_of_all += current
                break

            pygame.display.flip()
            clock.tick(FPS)

    print(f"Average score: {sum_of_all / 5}")
    pygame.quit()
