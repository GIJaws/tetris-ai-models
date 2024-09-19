import gymnasium as gym
import gym_simpletetris
import torch
import torch.optim as optim
import torch.nn.functional as F

from collections import namedtuple, deque
import random
import numpy as np
import logging
import sys
import os
import math

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)
from models.simple_dqn import SimpleDQN
from utils.helpful_utils import simplify_board, ACTION_COMBINATIONS
from utils.my_logging import (
    log_episode,
    log_q_values,
    log_action_distribution,
    log_loss,
    log_hardware_usage,
    aggregate_metrics,
    close_logging,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 1.0
EPS_END = 0.01
EPS_DECAY = 200000
TARGET_UPDATE = 10
MEMORY_SIZE = 10000
LEARNING_RATE = 1e-4
NUM_EPISODES = 10000

# Logging intervals
EPISODE_LOG_INTERVAL = 5
METRICS_AGGREGATE_INTERVAL = 10
HARDWARE_LOG_INTERVAL = 10
SAVE_MODEL_INTERVAL = 100


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))


class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


def train_dqn():
    env = gym.make("SimpleTetris-v0", render_mode=None)
    n_actions = env.action_space.n

    initial_state, _ = env.reset()
    state = simplify_board(initial_state)
    input_shape = (state.shape[0], state.shape[1])

    policy_net = SimpleDQN(input_shape, n_actions).to(device)
    target_net = SimpleDQN(input_shape, n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
    memory = ReplayMemory(MEMORY_SIZE)

    steps_done = 0
    episode_rewards = []
    # Metrics tracking
    steps_done = 0
    total_loss = 0
    loss_count = 0
    episode_q_values = []

    # Enhance logging by adding action distribution and loss tracking
    action_count = {action: 0 for action in ACTION_COMBINATIONS.keys()}
    try:
        for episode in range(NUM_EPISODES):
            state, _ = env.reset()
            state = simplify_board(state)
            state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
            episode_reward = 0
            episode_steps = 0
            lines_cleared = 0
            episode_q_values = []
            total_reward = 0

            while True:
                eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1.0 * steps_done / EPS_DECAY)
                action = select_action(state, policy_net, n_actions, eps_threshold)
                action_count[action] += 1  # Update action counts for logging
                observation, reward, terminated, truncated, info = env.step([action.item()])
                reward = calculate_reward(observation, info["lines_cleared"], terminated)
                total_reward += reward
                done = terminated or truncated

                if not done:
                    next_state = simplify_board(observation)
                    next_state = torch.tensor(next_state, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
                else:
                    next_state = None

                memory.push(state, action, next_state, torch.tensor([reward], device=device))
                state = next_state

                optimize_model(memory, policy_net, target_net, optimizer)

                if done:
                    episode_rewards.append(total_reward)
                    break

                steps_done += 1

                # Log hardware usage and update target network periodically...
                if episode % HARDWARE_LOG_INTERVAL == 0:
                    log_hardware_usage(episode)

                if episode % TARGET_UPDATE == 0:
                    target_net.load_state_dict(policy_net.state_dict())

                # Log episode stats and other metrics less frequently
                if episode % EPISODE_LOG_INTERVAL == 0:
                    avg_loss = total_loss / loss_count if loss_count > 0 else 0
                    log_episode(
                        episode,
                        episode_reward,
                        episode_steps,
                        lines_cleared,
                        eps_threshold,
                        avg_loss,
                        interval=EPISODE_LOG_INTERVAL,
                    )
                    total_loss = 0
                    loss_count = 0

                if episode % METRICS_AGGREGATE_INTERVAL == 0:
                    log_action_distribution(action_count, episode)
                    log_q_values(episode, episode_q_values, interval=METRICS_AGGREGATE_INTERVAL)
                    action_count = {action: 0 for action in ACTION_COMBINATIONS.keys()}
                    episode_q_values = []

                if episode % SAVE_MODEL_INTERVAL == 0:
                    # Save the trained model every SAVE_MODEL_INTERVAL
                    torch.save(policy_net.state_dict(), f"outputs/simple_dqn_{episode}_v1.pth")

            if episode % TARGET_UPDATE == 0:
                target_net.load_state_dict(policy_net.state_dict())

            if episode % 100 == 0:
                print(
                    f"Episode {episode}/{NUM_EPISODES}, Avg Reward (last 100): {np.mean(episode_rewards[-100:]):.2f}"
                )
    finally:
        print("Training complete")
        torch.save(policy_net.state_dict(), "outputs/simple_dqn_v1.pth")
        env.close()
        close_logging()
    return policy_net


def select_action(state, policy_net, n_actions, eps_threshold):
    sample = random.random()
    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)


def optimize_model(memory, policy_net, target_net, optimizer):
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(
        tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.bool
    )
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    state_action_values = policy_net(state_batch).gather(1, action_batch)

    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()


def calculate_reward(board, lines_cleared, game_over):
    reward = 0

    # Reward for each line cleared
    reward += lines_cleared  # +1 per line cleared

    # Bonus for clearing multiple lines at once
    if lines_cleared >= 2:
        reward += 1  # Additional +1 bonus

    # Penalty for game over
    if game_over:
        reward -= 1  # Modest penalty

    # Calculate heights
    heights = np.array([board.shape[0] - np.argmax(column) if np.any(column) else 0 for column in board.T])
    max_height = np.max(heights)

    # Height Penalty
    reward -= 0.01 * max_height

    # Holes Penalty
    holes = 0
    for x in range(board.shape[1]):
        column = board[:, x].astype(bool)
        filled = np.where(column)[0]
        if filled.size > 0:
            holes += np.sum(~column[filled[0] :])
    reward -= 0.1 * holes

    # Bumpiness Penalty
    bumpiness = np.sum(np.abs(np.diff(heights)))
    reward -= 0.05 * bumpiness

    # Reward for placing a piece
    reward += 0.1

    # Clip the reward to prevent extreme values
    reward = np.clip(reward, -1, 1)

    return reward


if __name__ == "__main__":
    trained_model = train_dqn()
    torch.save(trained_model.state_dict(), "simple_dqn_tetris_v1.pth")
