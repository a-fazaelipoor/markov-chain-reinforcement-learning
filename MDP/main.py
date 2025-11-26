import numpy as np
import pygame
import random
import copy
import pprint

from mpmath.libmp import to_str

from environment import WallE, PygameInit



def up_down_calc(move , value , move_prob , neigh_prob , grid_size , r , c):
    if move == "up":
        if r - 1 >= 0:
            if c - 1 >= 0 and c + 1 <= grid_size - 1:
                val = -1 + move_prob * value[r - 1][c] + neigh_prob * value[r][c - 1] + neigh_prob * value[r][c + 1]

            elif c - 1 >= 0 and c + 1 > grid_size - 1:
                val = -1 + move_prob * value[r - 1][c] + neigh_prob * value[r][c - 1]

            elif c - 1 < 0 and c + 1 <= grid_size - 1:
                val = -1 + move_prob * value[r - 1][c] + neigh_prob * value[r][c + 1]

            else:
                val = -1 + move_prob * value[r - 1][c]

        else:
            if c - 1 >= 0 and c + 1 <= grid_size - 1:
                val = -1 + neigh_prob * value[r][c - 1] + neigh_prob * value[r][c + 1]

            elif c - 1 >= 0 and c + 1 > grid_size - 1:
                val = -1 + neigh_prob * value[r][c - 1]

            elif c - 1 < 0 and c + 1 <= grid_size - 1:
                val = -1 + neigh_prob * value[r][c + 1]

    if move == "down":
        if r + 1 <= grid_size - 1:
            if c - 1 >= 0 and c + 1 <= grid_size - 1:
                val = -1 + move_prob * value[r + 1][c] + neigh_prob * value[r][c - 1] + neigh_prob * value[r][c + 1]

            elif c - 1 >= 0 and c + 1 > grid_size - 1:
                val = -1 + move_prob * value[r + 1][c] + neigh_prob * value[r][c - 1]

            elif c - 1 < 0 and c + 1 <= grid_size - 1:
                val = -1 + move_prob * value[r + 1][c] + neigh_prob * value[r][c + 1]

            else:
                val = -1 + move_prob * value[r + 1][c]

        else:
            if c - 1 >= 0 and c + 1 <= grid_size - 1:
                val = -1 + neigh_prob * value[r][c - 1] + neigh_prob * value[r][c + 1]

            elif c - 1 >= 0 and c + 1 > grid_size - 1:
                val = -1 + neigh_prob * value[r][c - 1]

            elif c - 1 < 0 and c + 1 <= grid_size - 1:
                val = -1 + neigh_prob * value[r][c + 1]

    return val


def left_right_calc(move , value , move_prob , neigh_prob , grid_size , r , c):
    if move == "left":
        if c - 1 >= 0:  # LEFT ممکن است
            if r - 1 >= 0 and r + 1 <= grid_size - 1:
                val = -1 + move_prob * value[r][c - 1] + neigh_prob * value[r - 1][c] + neigh_prob * value[r + 1][c]

            elif r - 1 >= 0 and r + 1 > grid_size - 1:
                val = -1 + move_prob * value[r][c - 1] + neigh_prob * value[r - 1][c]

            elif r - 1 < 0 and r + 1 <= grid_size - 1:
                val = -1 + move_prob * value[r][c - 1] + neigh_prob * value[r + 1][c]

            else:
                val = -1 + move_prob * value[r][c - 1]

        else:  # LEFT غیرممکن است
            if r - 1 >= 0 and r + 1 <= grid_size - 1:
                val = -1 + neigh_prob * value[r - 1][c] + neigh_prob * value[r + 1][c]

            elif r - 1 >= 0 and r + 1 > grid_size - 1:
                val = -1 + neigh_prob * value[r - 1][c]

            elif r - 1 < 0 and r + 1 <= grid_size - 1:
                val = -1 + neigh_prob * value[r + 1][c]

        if move == "right":
            if c + 1 <= grid_size - 1:
                if r - 1 >= 0 and r + 1 <= grid_size - 1:
                    val = -1 + move_prob * value[r][c + 1] + neigh_prob * value[r - 1][c] + neigh_prob * value[r + 1][c]

                elif r - 1 >= 0 and r + 1 > grid_size - 1:
                    val = -1 + move_prob * value[r][c + 1] + neigh_prob * value[r - 1][c]

                elif r - 1 < 0 and r + 1 <= grid_size - 1:
                    val = -1 + move_prob * value[r][c + 1] + neigh_prob * value[r + 1][c]

                else:
                    val = -1 + move_prob * value[r][c + 1]

            else:  # RIGHT غیرممکن است
                if r - 1 >= 0 and r + 1 <= grid_size - 1:
                    val = -1 + neigh_prob * value[r - 1][c] + neigh_prob * value[r + 1][c]

                elif r - 1 >= 0 and r + 1 > grid_size - 1:
                    val = -1 + neigh_prob * value[r - 1][c]

                elif r - 1 < 0 and r + 1 <= grid_size - 1:
                    val = -1 + neigh_prob * value[r + 1][c]

        return val


# -------------------------------- #
#     Implement your code here     #
# -------------------------------- #
class MySolver(WallE):
    def actions_model(self, reward_map):
        transition = self._WallE__probability_dict
        for key, value in reward_map.items():
            grid_size = self._WallE__grid_size
            v_table = [[-1 for _ in range(grid_size)] for _ in range(grid_size)]
            for r in range(grid_size):
                for c in range(grid_size):
                    v_table[r][c] = 0


            for r in range(grid_size):
                for c in range(grid_size):
                    p =transition[(r,c)]

                    up = p[0]
                    down = p[1]
                    left = p[2]
                    right = p[3]

                    up_up = up["intended"]
                    up_neighbors = up["neighbor"]

                    down_down = down["intended"]
                    down_neighbors = down["neighbor"]

                    left_left = left["intended"]
                    left_neighbors = left["neighbor"]

                    right_right = right["intended"]
                    right_neighbors = right["neighbor"]

                    up_val = up_down_calc("up" , value , up_up , up_neighbors , grid_size, r, c)
                    down_val = up_down_calc("down" , value , up_up , up_neighbors , grid_size, r, c)
                    left_val = left_right_calc("left" , value , up_up , up_neighbors , grid_size, r, c)
                    left_val = left_right_calc("right" , value , up_up , up_neighbors , grid_size, r, c)











        grid_size = self._WallE__grid_size
        actions_taken = np.full((grid_size, grid_size), -1, dtype=np.int8)
        for x in range(grid_size):
            for y in range(grid_size):
                if self.grid[x][y][0] == 'B':
                    continue
                actions_taken[x, y] = random.choice([0, 1, 2, 3])
        return actions_taken


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
                reward_map[r][c] = +1000
            else:
                reward_map[r][c] = 0
    return reward_map


def build_all_reward_map(thresholds, env):
    reward_map_dic = {
        "5" : [],
        "10" : [],
        "20" : [],
        "30" : [],
        "40" : []
    }

    for lim in thresholds:
        grid_size = env._WallE__grid_size
        reward_map = [[-1 for _ in range(grid_size)] for _ in range(grid_size)]
        for r in range(env._WallE__grid_size):
            for c in range(env._WallE__grid_size):
                tile, val = env.grid[r][c]
                if tile == "F":
                    reward_map[r][c] = 400
                elif tile == "E":
                    reward_map[r][c] = -400
                elif tile == "S":
                    if lim < val:
                        reward_map[r][c] = (lim * 250) // val
                    else:
                        reward_map[r][c] = 250
                else:
                    reward_map[r][c] = 0
        reward_map_dic[str(lim)] = reward_map
    return reward_map_dic

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
    MAX_MOVES_PER_EPISODE = 100

    reward_map_dic_without_goal_reward_map = build_all_reward_map(THRESHOLDS, env)




    # collect trash positions (fixed ordering) and values from the base grid
    trash_positions = env.get_trash_points()

    # -------------------------------- #
    #     Implement your code here     #
    # -------------------------------- #
    # Build all V-Tables
    # (You need to implement all vtables and call it here)
    # v_tables_map = ...

    print("Building goal_table...")
    reward_map = build_reward_map(env)
    reward_map_dic_without_goal_reward_map["goal"] = reward_map

    reward_map = reward_map_dic_without_goal_reward_map

    goal_table = env.actions_model(reward_map)
    print("Finished building goal_table.")

    # ---------------------------#

    screen, clock = PygameInit.initialization()
    sum_of_all = 0

    # game loop
    for episode in range(5):
        running = True
        current = 0
        moves = 0
        state = env.reset()

        while running:

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            env.render(screen)

            # --- Policy Selection ---
            # This is a simple implementation that ALWAYS uses the goal_table.
            # You MUST modify this section to implement the full logic
            action_table = goal_table
            # ------------------------

            moves += 1

            # Get action from the selected policy
            action = action_table[state]

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
