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

# REPLAY BUFFER CLASS
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, transition):
        """Stores a new experience in the buffer"""
        # transition: (state, action, reward, next_state, done)
        self.buffer.append(transition)

    def sample(self, batch_size):
        """Randomly samples a batch of transitions for training"""
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        """Returns the current size of the buffer"""
        return len(self.buffer)
    
# OPTIMIZATION FUNCTION
def optimize_model(policy_net, target_net, optimizer, replay_buffer, loss_fn):
    if len(replay_buffer) < BATCH_SIZE:
        return 0.0

    batch = replay_buffer.sample(BATCH_SIZE)
    states, actions, rewards, next_states, dones = zip(*batch)

    # Convert to Tensors
    states = torch.tensor(np.array(states))
    actions = torch.tensor(actions).unsqueeze(1)
    rewards = torch.tensor(rewards, dtype=torch.float32)
    next_states = torch.tensor(np.array(next_states))
    dones = torch.tensor(dones, dtype=torch.float32)

    # 1. Compute Q(s_t, a_t) - Q values for actions taken by the agent
    q_values = policy_net(states).gather(1, actions).squeeze(1)

    # 2. Compute Target Q-values: r_{t+1} + gamma * max_a Q_target(s_{t+1}, a)
    with torch.no_grad():
        # Get max Q from the target network for the next state
        next_q_values = target_net(next_states).max(1)[0]
        # Calculate target Q-value (set to reward only if episode is done)
        targets = rewards + GAMMA * next_q_values * (1 - dones)

    # 3. Compute Loss (Huber Loss is used for stability)
    loss = loss_fn(q_values, targets)
    
    # 4. Optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item()

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
    
    env = WallE_RL()
    screen, clock = PygameInit.initialization()
    
    #DQN Initialization
    policy_net = DQN()
    target_net = DQN()
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=ALPHA)
    # Using SmoothL1Loss (Huber Loss) for stability
    loss_fn = nn.SmoothL1Loss() 
    replay_buffer = ReplayBuffer(REPLAY_MEMORY_SIZE)

    epsilon = EPSILON_START
    global_step = 0
    losses = []
    episode_rewards = []
    print("Starting DQN training...")

    # TRAINING LOOP
    for episode in range(TRAIN_EPISODES):
        # Reset environment for new episode
        state_pos, agent_power = env.reset()
        trash_state = env._WallE_RL__get_trash_state() 
        state_vec = state_to_vector(state_pos, agent_power, trash_state)
        
        done = False
        total_reward = 0
        current_loss = 0

        while not done:            
            # 1. Epsilon-Greedy Action Selection
            if random.random() < epsilon:
                action = np.random.choice(ACTIONS) # Explore
            else:
                with torch.no_grad():
                    # Exploit: Choose action with max Q-Value
                    q_values = policy_net(torch.tensor(state_vec).unsqueeze(0))
                    action = q_values.argmax().item()

            # 2. Execute action and get results
            next_state_pos, reward, next_trash_state, done, next_agent_power = env.step(action)
            next_state_vec = state_to_vector(next_state_pos, next_agent_power, next_trash_state)

            # 3. Store transition in Replay Buffer
            replay_buffer.push((state_vec, action, reward, next_state_vec, done))

            # 4. Update state and counters
            state_vec = next_state_vec
            total_reward += reward
            global_step += 1

            # 5. Perform optimization step (Learn)
            loss = optimize_model(policy_net, target_net, optimizer, replay_buffer, loss_fn)
            if loss > 0.0:
                losses.append(loss)
                current_loss = loss

            # 6. Update target network periodically
            if global_step % TARGET_UPDATE == 0:
                target_net.load_state_dict(policy_net.state_dict())
        
        # 7. Epsilon Decay
        epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)
        episode_rewards.append(total_reward)

        if (episode + 1) % 200 == 0:
            avg_reward = np.mean(episode_rewards[-200:])
            print(f"Episode {episode + 1}/{TRAIN_EPISODES} | Avg Reward: {avg_reward:.1f} | Epsilon: {epsilon:.3f} | Last Loss: {current_loss:.4f}")

    print("Training finished!")
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            exit()
    print("Training finished!")
    
    # 1. PLOTTING RESULTS
    # Plotting Loss over training steps
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.title("Loss over Training Steps")
    plt.xlabel("Training Steps")
    plt.ylabel("Huber Loss")
    plt.grid(True)
    plt.savefig("loss_plot.png")
    print("Loss plot saved to loss_plot.png.")

    # Plotting Average Rewards
    # A simple moving average can better show learning trends
    window = 50 
    avg_rewards = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
    
    plt.figure(figsize=(10, 5))
    plt.plot(avg_rewards)
    plt.title(f"Average Reward over Episodes (Window={window})")
    plt.xlabel("Episode Index")
    plt.ylabel("Average Reward")
    plt.grid(True)
    plt.savefig("reward_plot.png")
    print("Reward plot saved to reward_plot.png.")

    # 2. FINAL TESTING
    print("\nStarting final test run...")
    test_rewards = []
    
    # Test Policy Net without exploration (epsilon = 0)
    for ep in range(TEST_EPISODES):
        state_pos, agent_power = env.reset()
        trash_state = env._WallE_RL__get_trash_state()
        state_vec = state_to_vector(state_pos, agent_power, trash_state)
        done = False
        total_reward = 0

        while not done:
            # Handle Pygame quit events during test rendering
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    exit()

            env.render(screen)

            # Action selection: Pure exploitation (argmax Q)
            with torch.no_grad():
                q_values = policy_net(torch.tensor(state_vec).unsqueeze(0))
                action = q_values.argmax().item()

            # Execute action
            next_state_pos, reward, next_trash_state, done, next_agent_power = env.step(action)
            next_state_vec = state_to_vector(next_state_pos, next_agent_power, next_trash_state)

            state_vec = next_state_vec
            total_reward += reward

            pygame.display.flip()
            # Slow down test run visualization (FPS=5)
            clock.tick(5) 

        test_rewards.append(total_reward)
        print(f"Test Episode {ep+1}/{TEST_EPISODES} | Final Reward: {total_reward:.1f}")

    mean_test_reward = np.mean(test_rewards)
    print("-" * 40)
    print(f"MEAN FINAL TEST REWARD: {mean_test_reward:.1f}")
    
    pygame.quit()